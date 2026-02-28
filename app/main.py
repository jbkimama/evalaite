"""
Evalaite - AI-Powered Exam Evaluator
"""
from fastapi import FastAPI, Request, Depends, HTTPException, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, StreamingResponse
import csv
import io
from sqlalchemy.orm import Session
from typing import Optional
from pydantic import BaseModel

from app.models import engine, Base, get_db, SavedPrompt, Review, ReviewResponse, ReviewListItem
from app.services import ReviewerService

# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Evalaite", version="1.0.0")
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")
templates = Jinja2Templates(directory="frontend/templates")

ALLOWED_TYPES = {
    'application/pdf': 'pdf',
    'image/png': 'image', 'image/jpeg': 'image', 'image/jpg': 'image',
    'text/plain': 'text'
}


class SavePromptRequest(BaseModel):
    name: str
    prompt: str


def get_file_type(content_type: str, filename: str) -> str | None:
    if content_type in ALLOWED_TYPES:
        return ALLOWED_TYPES[content_type]
    ext = filename.lower().split('.')[-1] if filename else ''
    return {'pdf': 'pdf', 'png': 'image', 'jpg': 'image', 'jpeg': 'image', 'txt': 'text'}.get(ext)


# Page routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/results/{review_id}", response_class=HTMLResponse)
async def results_page(request: Request, review_id: int):
    return templates.TemplateResponse("results.html", {"request": request, "review_id": review_id})


@app.get("/history", response_class=HTMLResponse)
async def history_page(request: Request):
    return templates.TemplateResponse("history.html", {"request": request})


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


# API routes
@app.post("/api/reviews/upload", response_model=ReviewResponse)
async def upload_file(
    file: UploadFile = File(...),
    marking_scheme: Optional[UploadFile] = File(None),
    custom_prompt: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    file_type = get_file_type(file.content_type, file.filename)
    if not file_type:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 10MB)")

    scheme_content = None
    if marking_scheme:
        scheme_bytes = await marking_scheme.read()
        scheme_type = get_file_type(marking_scheme.content_type, marking_scheme.filename)
        if scheme_type:
            scheme_content = (scheme_bytes, scheme_type)

    reviewer = ReviewerService(db)
    if file_type == 'pdf':
        return reviewer.create_review_from_pdf(content, file.filename, custom_prompt, scheme_content)
    elif file_type == 'image':
        return reviewer.create_review_from_image(content, file.filename, custom_prompt, scheme_content)
    else:
        return reviewer.create_review_from_text(content.decode('utf-8'), custom_prompt, scheme_content)


@app.get("/api/reviews/{review_id}", response_model=ReviewResponse)
async def get_review(review_id: int, db: Session = Depends(get_db)):
    review = ReviewerService(db).get_review(review_id)
    if not review:
        raise HTTPException(status_code=404, detail="Review not found")
    return review


@app.get("/api/reviews/", response_model=list[ReviewListItem])
async def list_reviews(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    return ReviewerService(db).get_reviews(skip=skip, limit=limit)


@app.get("/api/reviews/prompts/saved")
async def get_saved_prompts(db: Session = Depends(get_db)):
    prompts = db.query(SavedPrompt).all()
    return {"prompts": [{"name": p.name, "prompt": p.prompt} for p in prompts]}


@app.post("/api/reviews/prompts/save")
async def save_prompt(request: SavePromptRequest, db: Session = Depends(get_db)):
    existing = db.query(SavedPrompt).filter(SavedPrompt.name == request.name).first()
    if existing:
        existing.prompt = request.prompt
    else:
        db.add(SavedPrompt(name=request.name, prompt=request.prompt))
    db.commit()
    return {"success": True}


@app.delete("/api/reviews/prompts/{prompt_name}")
async def delete_prompt(prompt_name: str, db: Session = Depends(get_db)):
    prompt = db.query(SavedPrompt).filter(SavedPrompt.name == prompt_name).first()
    if not prompt:
        raise HTTPException(status_code=404, detail="Prompt not found")
    db.delete(prompt)
    db.commit()
    return {"success": True}


@app.delete("/api/reviews/{review_id}")
async def delete_review(review_id: int, db: Session = Depends(get_db)):
    review = db.query(Review).filter(Review.id == review_id).first()
    if not review:
        raise HTTPException(status_code=404, detail="Review not found")
    db.delete(review)
    db.commit()
    return {"success": True}


@app.get("/api/reviews/export/csv")
async def export_reviews_csv(db: Session = Depends(get_db)):
    reviews = ReviewerService(db).get_reviews(limit=10000)

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Rank', 'Date', 'Filename', 'Type', 'Score', 'Correct', 'Total Questions', 'Status'])

    sorted_reviews = sorted(reviews, key=lambda r: r.total_score or 0, reverse=True)

    for i, r in enumerate(sorted_reviews, 1):
        writer.writerow([
            i,
            r.created_at.strftime('%Y-%m-%d %H:%M'),
            r.original_filename or 'Text input',
            r.input_type.value,
            f"{r.total_score:.1f}" if r.total_score else '',
            r.correct_answers,
            r.total_questions,
            r.status.value
        ])

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=evalaite_results.csv"}
    )


@app.get("/api/reviews/{review_id}/export/csv")
async def export_review_csv(review_id: int, db: Session = Depends(get_db)):
    review = ReviewerService(db).get_review(review_id)
    if not review:
        raise HTTPException(status_code=404, detail="Review not found")

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Question #', 'Question', 'Student Answer', 'Correct Answer', 'Score', 'Is Correct', 'Feedback'])

    for q in review.questions:
        writer.writerow([
            q.question_number,
            q.question_text,
            q.student_answer,
            q.correct_answer or '',
            f"{q.score:.1f}" if q.score else '',
            'Yes' if q.is_correct else 'No' if q.is_correct is not None else '',
            q.feedback or ''
        ])

    output.seek(0)
    filename = f"review_{review_id}.csv"
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )
