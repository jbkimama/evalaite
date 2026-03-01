"""
Evalaite - AI-Powered Exam Evaluator
"""
from fastapi import FastAPI, Request, Depends, HTTPException, UploadFile, File, Form, Response, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse
import csv
import io
import secrets
from datetime import datetime, timedelta
from sqlalchemy.orm import Session as DBSession
from typing import Optional, List
from pydantic import BaseModel

from app.models import (
    engine, Base, get_db, SavedPrompt,
    Subject, SubjectCreate, SubjectResponse,
    ExamSeries, SeriesCreate, SeriesResponse,
    Exam, ExamCreate, ExamResponse,
    Script, ScriptResponse, ScriptDetailResponse, ScriptStatus, FileType,
    User, Session as UserSession, UserCreate, UserUpdate, UserLogin, UserResponse, UserRole
)
from app.services import ExamService, get_file_type

# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Evalaite", version="2.0.0")


@app.on_event("startup")
def create_default_users():
    """Create default users if no users exist."""
    db = next(get_db())
    try:
        if db.query(User).count() == 0:
            admin = User(
                username="admin",
                password_hash=User.hash_password("admin123"),
                role=UserRole.SUPERADMIN
            )
            db.add(admin)

            teacher = User(
                username="teacher",
                password_hash=User.hash_password("teacher123"),
                role=UserRole.TEACHER
            )
            db.add(teacher)

            db.commit()
            print("Default users created:")
            print("  Superadmin: admin / admin123")
            print("  Teacher: teacher / teacher123")
    finally:
        db.close()


app.mount("/static", StaticFiles(directory="frontend/static"), name="static")
templates = Jinja2Templates(directory="frontend/templates")

SESSION_EXPIRE_HOURS = 24


class SavePromptRequest(BaseModel):
    name: str
    prompt: str


def get_current_user(request: Request, db: DBSession = Depends(get_db)) -> Optional[User]:
    """Get current user from session cookie."""
    token = request.cookies.get("session_token")
    if not token:
        return None
    session = db.query(UserSession).filter(
        UserSession.token == token,
        UserSession.expires_at > datetime.utcnow()
    ).first()
    if not session:
        return None
    return session.user


def require_auth(request: Request, db: DBSession = Depends(get_db)) -> User:
    """Require authentication - redirect to login if not authenticated."""
    user = get_current_user(request, db)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


def require_superadmin(request: Request, db: DBSession = Depends(get_db)) -> User:
    user = get_current_user(request, db)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    if user.role != UserRole.SUPERADMIN:
        raise HTTPException(status_code=403, detail="Superadmin access required")
    return user


# ==================== Page Routes ====================

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, db: DBSession = Depends(get_db)):
    user = get_current_user(request, db)
    if user:
        return RedirectResponse(url="/", status_code=302)
    return templates.TemplateResponse("login.html", {"request": request})


@app.get("/admin", response_class=HTMLResponse)
async def admin_page(request: Request, db: DBSession = Depends(get_db)):
    user = get_current_user(request, db)
    if not user:
        return RedirectResponse(url="/login", status_code=302)
    if user.role != UserRole.SUPERADMIN:
        return RedirectResponse(url="/", status_code=302)
    return templates.TemplateResponse("admin.html", {"request": request, "user": user})


@app.get("/logout")
async def logout(request: Request, response: Response, db: DBSession = Depends(get_db)):
    token = request.cookies.get("session_token")
    if token:
        session = db.query(UserSession).filter(UserSession.token == token).first()
        if session:
            db.delete(session)
            db.commit()
    response = RedirectResponse(url="/login", status_code=302)
    response.delete_cookie("session_token")
    return response


@app.get("/", response_class=HTMLResponse)
async def home(request: Request, db: DBSession = Depends(get_db)):
    user = get_current_user(request, db)
    if not user:
        return RedirectResponse(url="/login", status_code=302)
    return templates.TemplateResponse("index.html", {"request": request, "user": user})


@app.get("/exam/{exam_id}", response_class=HTMLResponse)
async def exam_page(request: Request, exam_id: int, db: DBSession = Depends(get_db)):
    user = get_current_user(request, db)
    if not user:
        return RedirectResponse(url="/login", status_code=302)
    return templates.TemplateResponse("exam.html", {"request": request, "exam_id": exam_id, "user": user})


@app.get("/results", response_class=HTMLResponse)
async def results_page(request: Request, db: DBSession = Depends(get_db)):
    user = get_current_user(request, db)
    if not user:
        return RedirectResponse(url="/login", status_code=302)
    return templates.TemplateResponse("results.html", {"request": request, "user": user})


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


# ==================== Auth API ====================

@app.post("/api/auth/login")
async def login(credentials: UserLogin, response: Response, db: DBSession = Depends(get_db)):
    user = db.query(User).filter(User.username == credentials.username).first()
    if not user or not user.verify_password(credentials.password):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    token = secrets.token_hex(32)
    session = UserSession(
        token=token,
        user_id=user.id,
        expires_at=datetime.utcnow() + timedelta(hours=SESSION_EXPIRE_HOURS)
    )
    db.add(session)
    db.commit()

    response.set_cookie(
        key="session_token",
        value=token,
        httponly=True,
        max_age=SESSION_EXPIRE_HOURS * 3600,
        samesite="lax"
    )
    return {"success": True, "user": UserResponse.model_validate(user)}


@app.get("/api/auth/me")
async def get_me(request: Request, db: DBSession = Depends(get_db)):
    user = get_current_user(request, db)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return UserResponse.model_validate(user)


# ==================== Admin API (Superadmin only) ====================

@app.get("/api/admin/users")
async def list_users(request: Request, db: DBSession = Depends(get_db)):
    require_superadmin(request, db)
    users = db.query(User).order_by(User.created_at.desc()).all()
    return [UserResponse.model_validate(u) for u in users]


@app.post("/api/admin/users")
async def create_user(user_data: UserCreate, request: Request, db: DBSession = Depends(get_db)):
    require_superadmin(request, db)
    if db.query(User).filter(User.username == user_data.username).first():
        raise HTTPException(status_code=400, detail="Username already taken")

    role = UserRole.SUPERADMIN if user_data.role == "superadmin" else UserRole.TEACHER
    user = User(
        username=user_data.username,
        password_hash=User.hash_password(user_data.password),
        role=role
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return UserResponse.model_validate(user)


@app.put("/api/admin/users/{user_id}")
async def update_user(user_id: int, user_data: UserUpdate, request: Request, db: DBSession = Depends(get_db)):
    require_superadmin(request, db)
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if user_data.username and user_data.username != user.username:
        if db.query(User).filter(User.username == user_data.username).first():
            raise HTTPException(status_code=400, detail="Username already taken")
        user.username = user_data.username

    if user_data.password:
        user.password_hash = User.hash_password(user_data.password)
    if user_data.role:
        user.role = UserRole.SUPERADMIN if user_data.role == "superadmin" else UserRole.TEACHER
    if user_data.is_active is not None:
        user.is_active = user_data.is_active

    db.commit()
    db.refresh(user)
    return UserResponse.model_validate(user)


@app.delete("/api/admin/users/{user_id}")
async def delete_user(user_id: int, request: Request, db: DBSession = Depends(get_db)):
    admin = require_superadmin(request, db)
    if admin.id == user_id:
        raise HTTPException(status_code=400, detail="Cannot delete yourself")

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    db.query(UserSession).filter(UserSession.user_id == user_id).delete()
    db.delete(user)
    db.commit()
    return {"success": True}


# ==================== Subjects API ====================

@app.get("/api/subjects", response_model=List[SubjectResponse])
async def list_subjects(request: Request, db: DBSession = Depends(get_db)):
    user = require_auth(request, db)
    subjects = db.query(Subject).filter(Subject.user_id == user.id).order_by(Subject.name).all()
    return [SubjectResponse.model_validate(s) for s in subjects]


@app.post("/api/subjects", response_model=SubjectResponse)
async def create_subject(data: SubjectCreate, request: Request, db: DBSession = Depends(get_db)):
    user = require_auth(request, db)
    subject = Subject(name=data.name, user_id=user.id)
    db.add(subject)
    db.commit()
    db.refresh(subject)
    return SubjectResponse.model_validate(subject)


@app.delete("/api/subjects/{subject_id}")
async def delete_subject(subject_id: int, request: Request, db: DBSession = Depends(get_db)):
    user = require_auth(request, db)
    subject = db.query(Subject).filter(Subject.id == subject_id, Subject.user_id == user.id).first()
    if not subject:
        raise HTTPException(status_code=404, detail="Subject not found")
    db.delete(subject)
    db.commit()
    return {"success": True}


# ==================== Exam Series API ====================

@app.get("/api/series", response_model=List[SeriesResponse])
async def list_series(request: Request, db: DBSession = Depends(get_db)):
    user = require_auth(request, db)
    series_list = db.query(ExamSeries).filter(ExamSeries.user_id == user.id).order_by(ExamSeries.created_at.desc()).all()
    result = []
    for s in series_list:
        data = SeriesResponse.model_validate(s)
        data.exam_count = len(s.exams)
        result.append(data)
    return result


@app.post("/api/series", response_model=SeriesResponse)
async def create_series(data: SeriesCreate, request: Request, db: DBSession = Depends(get_db)):
    user = require_auth(request, db)
    series = ExamSeries(name=data.name, description=data.description, user_id=user.id)
    db.add(series)
    db.commit()
    db.refresh(series)
    return SeriesResponse.model_validate(series)


@app.delete("/api/series/{series_id}")
async def delete_series(series_id: int, request: Request, db: DBSession = Depends(get_db)):
    user = require_auth(request, db)
    series = db.query(ExamSeries).filter(ExamSeries.id == series_id, ExamSeries.user_id == user.id).first()
    if not series:
        raise HTTPException(status_code=404, detail="Series not found")
    db.delete(series)
    db.commit()
    return {"success": True}


# ==================== Exams API ====================

@app.get("/api/exams", response_model=List[ExamResponse])
async def list_exams(request: Request, series_id: Optional[int] = None, db: DBSession = Depends(get_db)):
    user = require_auth(request, db)
    query = db.query(Exam).filter(Exam.user_id == user.id)
    if series_id:
        query = query.filter(Exam.series_id == series_id)
    exams = query.order_by(Exam.created_at.desc()).all()

    result = []
    for e in exams:
        data = ExamResponse(
            id=e.id,
            series_id=e.series_id,
            subject_id=e.subject_id,
            form=e.form,
            subject_name=e.subject.name if e.subject else "",
            series_name=e.series.name if e.series else "",
            has_marking_scheme=e.marking_scheme_data is not None,
            marking_scheme_filename=e.marking_scheme_filename,
            marking_scheme_text=e.marking_scheme_text,
            evaluation_prompt=e.evaluation_prompt,
            script_count=len(e.scripts),
            evaluated_count=sum(1 for s in e.scripts if s.status == ScriptStatus.COMPLETED),
            average_score=sum(s.total_score or 0 for s in e.scripts if s.status == ScriptStatus.COMPLETED) / len([s for s in e.scripts if s.status == ScriptStatus.COMPLETED]) if any(s.status == ScriptStatus.COMPLETED for s in e.scripts) else None,
            created_at=e.created_at
        )
        result.append(data)
    return result


@app.post("/api/exams", response_model=ExamResponse)
async def create_exam(data: ExamCreate, request: Request, db: DBSession = Depends(get_db)):
    user = require_auth(request, db)

    # Verify series and subject belong to user
    series = db.query(ExamSeries).filter(ExamSeries.id == data.series_id, ExamSeries.user_id == user.id).first()
    if not series:
        raise HTTPException(status_code=404, detail="Series not found")

    subject = db.query(Subject).filter(Subject.id == data.subject_id, Subject.user_id == user.id).first()
    if not subject:
        raise HTTPException(status_code=404, detail="Subject not found")

    exam = Exam(
        user_id=user.id,
        series_id=data.series_id,
        subject_id=data.subject_id,
        evaluation_prompt=data.evaluation_prompt
    )
    db.add(exam)
    db.commit()
    db.refresh(exam)

    return ExamResponse(
        id=exam.id,
        series_id=exam.series_id,
        subject_id=exam.subject_id,
        form=exam.form,
        subject_name=subject.name,
        series_name=series.name,
        has_marking_scheme=False,
        created_at=exam.created_at
    )


@app.get("/api/exams/{exam_id}", response_model=ExamResponse)
async def get_exam(exam_id: int, request: Request, db: DBSession = Depends(get_db)):
    user = require_auth(request, db)
    exam = db.query(Exam).filter(Exam.id == exam_id, Exam.user_id == user.id).first()
    if not exam:
        raise HTTPException(status_code=404, detail="Exam not found")

    completed = [s for s in exam.scripts if s.status == ScriptStatus.COMPLETED]
    return ExamResponse(
        id=exam.id,
        series_id=exam.series_id,
        subject_id=exam.subject_id,
        form=exam.form,
        subject_name=exam.subject.name if exam.subject else "",
        series_name=exam.series.name if exam.series else "",
        has_marking_scheme=exam.marking_scheme_data is not None,
        marking_scheme_filename=exam.marking_scheme_filename,
        marking_scheme_text=exam.marking_scheme_text,
        evaluation_prompt=exam.evaluation_prompt,
        script_count=len(exam.scripts),
        evaluated_count=len(completed),
        average_score=sum(s.total_score or 0 for s in completed) / len(completed) if completed else None,
        created_at=exam.created_at
    )


@app.post("/api/transcribe-marking-scheme")
async def transcribe_marking_scheme_endpoint(
    request: Request,
    file: UploadFile = File(...),
    db: DBSession = Depends(get_db)
):
    """Transcribe a marking scheme file without creating an exam."""
    from app.services import transcribe_marking_scheme

    require_auth(request, db)

    file_type = get_file_type(file.content_type, file.filename)
    if not file_type:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 10MB)")

    transcription = transcribe_marking_scheme(content, file_type)
    return {"transcription": transcription}


@app.post("/api/exams/{exam_id}/marking-scheme")
async def upload_marking_scheme(
    exam_id: int,
    request: Request,
    file: UploadFile = File(...),
    db: DBSession = Depends(get_db)
):
    from app.services import transcribe_marking_scheme

    user = require_auth(request, db)
    exam = db.query(Exam).filter(Exam.id == exam_id, Exam.user_id == user.id).first()
    if not exam:
        raise HTTPException(status_code=404, detail="Exam not found")

    file_type = get_file_type(file.content_type, file.filename)
    if not file_type:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 10MB)")

    # Transcribe the marking scheme immediately
    transcription = transcribe_marking_scheme(content, file_type)

    exam.marking_scheme_filename = file.filename
    exam.marking_scheme_type = file_type
    exam.marking_scheme_data = content
    exam.marking_scheme_text = transcription
    db.commit()

    return {"success": True, "filename": file.filename, "transcription": transcription}


class MarkingSchemeTextUpdate(BaseModel):
    text: str


@app.put("/api/exams/{exam_id}/marking-scheme-text")
async def update_marking_scheme_text(
    exam_id: int,
    request: Request,
    data: MarkingSchemeTextUpdate,
    db: DBSession = Depends(get_db)
):
    user = require_auth(request, db)
    exam = db.query(Exam).filter(Exam.id == exam_id, Exam.user_id == user.id).first()
    if not exam:
        raise HTTPException(status_code=404, detail="Exam not found")

    exam.marking_scheme_text = data.text
    db.commit()
    return {"success": True}


@app.put("/api/exams/{exam_id}/prompt")
async def update_exam_prompt(
    exam_id: int,
    request: Request,
    db: DBSession = Depends(get_db)
):
    user = require_auth(request, db)
    exam = db.query(Exam).filter(Exam.id == exam_id, Exam.user_id == user.id).first()
    if not exam:
        raise HTTPException(status_code=404, detail="Exam not found")

    body = await request.json()
    exam.evaluation_prompt = body.get("prompt")
    db.commit()

    return {"success": True}


@app.delete("/api/exams/{exam_id}")
async def delete_exam(exam_id: int, request: Request, db: DBSession = Depends(get_db)):
    user = require_auth(request, db)
    exam = db.query(Exam).filter(Exam.id == exam_id, Exam.user_id == user.id).first()
    if not exam:
        raise HTTPException(status_code=404, detail="Exam not found")
    db.delete(exam)
    db.commit()
    return {"success": True}


# ==================== Scripts API ====================

@app.get("/api/exams/{exam_id}/scripts", response_model=List[ScriptResponse])
async def list_scripts(exam_id: int, request: Request, db: DBSession = Depends(get_db)):
    user = require_auth(request, db)
    exam = db.query(Exam).filter(Exam.id == exam_id, Exam.user_id == user.id).first()
    if not exam:
        raise HTTPException(status_code=404, detail="Exam not found")

    # Sort by score descending (completed first, then by score)
    scripts = sorted(exam.scripts, key=lambda s: (
        s.status != ScriptStatus.COMPLETED,
        -(s.total_score or 0)
    ))
    return [ScriptResponse.model_validate(s) for s in scripts]


@app.post("/api/exams/{exam_id}/scripts")
async def upload_scripts(
    exam_id: int,
    request: Request,
    files: List[UploadFile] = File(...),
    db: DBSession = Depends(get_db)
):
    user = require_auth(request, db)
    exam = db.query(Exam).filter(Exam.id == exam_id, Exam.user_id == user.id).first()
    if not exam:
        raise HTTPException(status_code=404, detail="Exam not found")

    uploaded = []
    for file in files:
        file_type = get_file_type(file.content_type, file.filename)
        if not file_type:
            continue

        content = await file.read()
        if len(content) > 10 * 1024 * 1024:
            continue

        script = Script(
            exam_id=exam_id,
            filename=file.filename,
            file_type=file_type,
            file_data=content,
            status=ScriptStatus.PENDING
        )
        db.add(script)
        uploaded.append(file.filename)

    db.commit()
    return {"success": True, "uploaded": uploaded, "count": len(uploaded)}


@app.get("/api/scripts/{script_id}", response_model=ScriptDetailResponse)
async def get_script(script_id: int, request: Request, db: DBSession = Depends(get_db)):
    user = require_auth(request, db)
    script = db.query(Script).filter(Script.id == script_id).first()
    if not script or script.exam.user_id != user.id:
        raise HTTPException(status_code=404, detail="Script not found")

    return ScriptDetailResponse.model_validate(script)


@app.delete("/api/scripts/{script_id}")
async def delete_script(script_id: int, request: Request, db: DBSession = Depends(get_db)):
    user = require_auth(request, db)
    script = db.query(Script).filter(Script.id == script_id).first()
    if not script or script.exam.user_id != user.id:
        raise HTTPException(status_code=404, detail="Script not found")
    db.delete(script)
    db.commit()
    return {"success": True}


class StudentInfoUpdate(BaseModel):
    student_name: Optional[str] = None
    student_number: Optional[str] = None


@app.put("/api/scripts/{script_id}/student-info")
async def update_student_info(script_id: int, request: Request, data: StudentInfoUpdate, db: DBSession = Depends(get_db)):
    user = require_auth(request, db)
    script = db.query(Script).filter(Script.id == script_id).first()
    if not script or script.exam.user_id != user.id:
        raise HTTPException(status_code=404, detail="Script not found")
    script.student_name = data.student_name
    script.student_number = data.student_number
    db.commit()
    return {"success": True}


# ==================== Evaluation API ====================

@app.post("/api/exams/{exam_id}/evaluate")
async def evaluate_exam(exam_id: int, request: Request, background_tasks: BackgroundTasks, db: DBSession = Depends(get_db)):
    user = require_auth(request, db)
    exam = db.query(Exam).filter(Exam.id == exam_id, Exam.user_id == user.id).first()
    if not exam:
        raise HTTPException(status_code=404, detail="Exam not found")

    # Run evaluation in background
    def run_evaluation(exam_id: int):
        import traceback
        from app.models import get_db, Exam
        from app.services import ExamService
        db_session = next(get_db())
        try:
            exam = db_session.query(Exam).filter(Exam.id == exam_id).first()
            if exam:
                service = ExamService(db_session)
                service.evaluate_all_scripts(exam)
                print(f"[Background] Evaluation completed for exam {exam_id}")
        except Exception as e:
            print(f"[Background] Evaluation error for exam {exam_id}: {e}")
            traceback.print_exc()
        finally:
            db_session.close()

    background_tasks.add_task(run_evaluation, exam_id)

    return {
        "success": True,
        "message": "Evaluation started in background"
    }


@app.post("/api/scripts/{script_id}/evaluate")
async def evaluate_script(script_id: int, request: Request, db: DBSession = Depends(get_db)):
    user = require_auth(request, db)
    script = db.query(Script).filter(Script.id == script_id).first()
    if not script or script.exam.user_id != user.id:
        raise HTTPException(status_code=404, detail="Script not found")

    service = ExamService(db)
    script = service.evaluate_script(script)

    return ScriptDetailResponse.model_validate(script)


# ==================== Prompts API ====================

@app.get("/api/prompts/saved")
async def get_saved_prompts(request: Request, db: DBSession = Depends(get_db)):
    user = require_auth(request, db)
    prompts = db.query(SavedPrompt).filter(SavedPrompt.user_id == user.id).all()
    return {"prompts": [{"name": p.name, "prompt": p.prompt} for p in prompts]}


@app.post("/api/prompts/save")
async def save_prompt(request: Request, prompt_data: SavePromptRequest, db: DBSession = Depends(get_db)):
    user = require_auth(request, db)
    existing = db.query(SavedPrompt).filter(SavedPrompt.name == prompt_data.name, SavedPrompt.user_id == user.id).first()
    if existing:
        existing.prompt = prompt_data.prompt
    else:
        db.add(SavedPrompt(name=prompt_data.name, prompt=prompt_data.prompt, user_id=user.id))
    db.commit()
    return {"success": True}


@app.delete("/api/prompts/{prompt_name}")
async def delete_prompt(prompt_name: str, request: Request, db: DBSession = Depends(get_db)):
    user = require_auth(request, db)
    prompt = db.query(SavedPrompt).filter(SavedPrompt.name == prompt_name, SavedPrompt.user_id == user.id).first()
    if not prompt:
        raise HTTPException(status_code=404, detail="Prompt not found")
    db.delete(prompt)
    db.commit()
    return {"success": True}


# ==================== Export API ====================

@app.get("/api/exams/{exam_id}/export/csv")
async def export_exam_csv(exam_id: int, request: Request, db: DBSession = Depends(get_db)):
    user = require_auth(request, db)
    exam = db.query(Exam).filter(Exam.id == exam_id, Exam.user_id == user.id).first()
    if not exam:
        raise HTTPException(status_code=404, detail="Exam not found")

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Rank', 'Student Name', 'Student Number', 'Filename', 'Score', 'Correct', 'Total Questions', 'Status'])

    # Sort by score descending
    scripts = sorted(exam.scripts, key=lambda s: -(s.total_score or 0))

    for i, s in enumerate(scripts, 1):
        writer.writerow([
            i,
            s.student_name or '',
            s.student_number or '',
            s.filename or '',
            f"{s.total_score:.1f}" if s.total_score else '',
            s.correct_answers,
            s.total_questions,
            s.status.value
        ])

    output.seek(0)
    filename = f"exam_{exam_id}_results.csv"
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@app.get("/api/scripts/{script_id}/export/csv")
async def export_script_csv(script_id: int, request: Request, db: DBSession = Depends(get_db)):
    user = require_auth(request, db)
    script = db.query(Script).filter(Script.id == script_id).first()
    if not script or script.exam.user_id != user.id:
        raise HTTPException(status_code=404, detail="Script not found")

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Question #', 'Question', 'Student Answer', 'Correct Answer', 'Score', 'Is Correct', 'Feedback'])

    for q in script.questions:
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
    filename = f"script_{script_id}.csv"
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )
