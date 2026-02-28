"""
Evalaite - Services
Gemini AI, file parsing, and review orchestration.
"""
import json
import io
import re
import hashlib
from google import genai
from google.genai import types
from PIL import Image
import pdfplumber
from sqlalchemy.orm import Session

import os
from dotenv import load_dotenv
from app.models import (
    Review, Question, SavedPrompt, InputType, ReviewStatus,
    QuestionInput, GeminiEvaluation
)

load_dotenv()
_client = None


def get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.getenv("GEMINI_API_KEY", ""))
    return _client


# --- Parser Functions ---
def parse_pdf(content: bytes) -> tuple[str, list[QuestionInput]]:
    text_parts = []
    with pdfplumber.open(io.BytesIO(content)) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)
    raw_text = "\n".join(text_parts)
    return raw_text, extract_questions(raw_text)


def parse_image(content: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(content))
    if image.mode not in ('RGB', 'L'):
        image = image.convert('RGB')
    return image


def is_scanned_pdf(content: bytes) -> bool:
    with pdfplumber.open(io.BytesIO(content)) as pdf:
        if not pdf.pages:
            return True
        text = pdf.pages[0].extract_text()
        return not text or len(text.strip()) < 50


def pdf_to_images(content: bytes) -> list[Image.Image]:
    images = []
    with pdfplumber.open(io.BytesIO(content)) as pdf:
        for page in pdf.pages:
            img = page.to_image(resolution=150).original
            if img.mode not in ('RGB', 'L'):
                img = img.convert('RGB')
            images.append(img)
    return images


def extract_questions(text: str) -> list[QuestionInput]:
    pattern = r'Q(\d+)[.:]\s*(.+?)\s*(?:A[.:]\s*|Answer[.:]\s*)(.+?)(?=Q\d+[.:]|$)'
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    if matches:
        return [QuestionInput(question_text=q.strip(), student_answer=a.strip()) for _, q, a in matches]

    pattern = r'(\d+)[.)]\s*(.+?)\s*(?:Answer[.:]\s*)(.+?)(?=\d+[.)]|$)'
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    if matches:
        return [QuestionInput(question_text=q.strip(), student_answer=a.strip()) for _, q, a in matches]
    return []


# --- Gemini AI Functions ---
MODEL = "gemini-2.5-flash"


def image_to_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    return buf.getvalue()


def evaluate_answer(question: QuestionInput, custom_prompt: str = None, marking_scheme: str = None) -> GeminiEvaluation:
    template = custom_prompt or """Evaluate the answer.
Question: {question}
Student Answer: {student_answer}
{correct_answer_section}

Respond in YAML:
is_correct: true/false
score: 0-100
feedback: explanation"""

    correct = f"Correct Answer: {question.correct_answer}" if question.correct_answer else ""
    prompt = template.format(question=question.question_text, student_answer=question.student_answer, correct_answer_section=correct)
    if marking_scheme:
        prompt += f"\n\nMarking Scheme:\n{marking_scheme}"

    try:
        response = get_client().models.generate_content(model=MODEL, contents=prompt)
        return parse_evaluation(response.text)
    except Exception as e:
        return GeminiEvaluation(is_correct=False, score=0, feedback=f"Error: {e}")


def evaluate_batch(questions: list[QuestionInput], custom_prompt: str = None, marking_scheme: str = None) -> list[GeminiEvaluation]:
    return [evaluate_answer(q, custom_prompt, marking_scheme) for q in questions]


def extract_and_evaluate_image(image: Image.Image, custom_prompt: str = None, scheme_text: str = None, scheme_image: Image.Image = None):
    try:
        img_bytes = image_to_bytes(image)
        contents = [types.Part.from_bytes(data=img_bytes, mime_type="image/png")]

        if scheme_image:
            contents.append(types.Part.from_bytes(data=image_to_bytes(scheme_image), mime_type="image/png"))

        if custom_prompt:
            transcript_prompt = "Transcribe all text in this image exactly as written. Return only the transcription."
            transcript_resp = get_client().models.generate_content(model=MODEL, contents=[types.Part.from_bytes(data=img_bytes, mime_type="image/png"), transcript_prompt])
            transcript = transcript_resp.text.strip()

            eval_prompt = custom_prompt.replace("{question}", "Evaluate the student's work")
            eval_prompt = eval_prompt.replace("{student_answer}", transcript)
            eval_prompt = eval_prompt.replace("{correct_answer_section}", f"\nMarking Scheme:\n{scheme_text}" if scheme_text else "")

            contents.append(eval_prompt)
            response = get_client().models.generate_content(model=MODEL, contents=contents)
            evaluation = parse_evaluation(response.text)
            question = QuestionInput(question_text="Document Evaluation", student_answer=transcript)
            return [question], [evaluation]

        prefix = "First image is student work. Second image is marking scheme. " if scheme_image else ""
        scheme = f"\nMarking Scheme:\n{scheme_text}" if scheme_text else ""
        prompt = f"""{prefix}Extract questions and answers from this image.{scheme}
Respond in JSON:
{{"questions": [{{"question_text": "...", "student_answer": "...", "correct_answer": "..."}}]}}"""

        contents.append(prompt)
        response = get_client().models.generate_content(model=MODEL, contents=contents)
        questions = parse_extraction(response.text)
        if questions:
            return questions, evaluate_batch(questions, None, scheme_text)
        return [], []
    except Exception:
        return [], []


def extract_and_evaluate_text(text: str, marking_scheme: str = None):
    prompt = f"""Analyze this exam content and extract questions with answers.
{f"Marking Scheme: {marking_scheme}" if marking_scheme else ""}

Content:
{text}

Respond in JSON:
{{"results": [{{"question_text": "...", "student_answer": "...", "correct_answer": "...", "is_correct": true/false, "score": 0-100, "feedback": "..."}}]}}"""

    try:
        response = get_client().models.generate_content(model=MODEL, contents=prompt)
        return parse_combined(response.text)
    except Exception:
        return [], []


def parse_evaluation(text: str) -> GeminiEvaluation:
    try:
        if '```' in text:
            text = text.split('```')[1].split('```')[0]
            if text.startswith('yaml') or text.startswith('json'):
                text = text[4:]
        text = text.strip()

        try:
            data = json.loads(text)
            return GeminiEvaluation(is_correct=data.get('is_correct', False), score=float(data.get('score', 0)), feedback=data.get('feedback', ''))
        except json.JSONDecodeError:
            pass

        is_correct, score, feedback_lines, in_feedback = False, 0.0, [], False
        for line in text.split('\n'):
            lower = line.lower().strip()
            if lower.startswith('is_correct:'):
                is_correct = 'true' in lower or 'yes' in lower
                in_feedback = False
            elif lower.startswith('score:'):
                try:
                    s = line.split(':', 1)[1].strip()
                    if '/' in s:
                        s = s.split('/')[0]
                    score = float(s)
                    if score <= 40:
                        score *= 2.5
                except ValueError:
                    pass
                in_feedback = False
            elif lower.startswith('feedback:'):
                rest = line.split(':', 1)[1].strip()
                if rest and rest != '|':
                    feedback_lines.append(rest)
                in_feedback = True
            elif in_feedback:
                feedback_lines.append(line)

        return GeminiEvaluation(is_correct=is_correct, score=score, feedback='\n'.join(feedback_lines).strip() or 'No feedback')
    except Exception:
        return GeminiEvaluation(is_correct=False, score=0, feedback=text[:500])


def parse_extraction(text: str) -> list[QuestionInput]:
    try:
        if '```' in text:
            text = text.split('```')[1].split('```')[0]
        data = json.loads(text.strip())
        return [QuestionInput(question_text=q.get('question_text', ''), student_answer=q.get('student_answer', ''), correct_answer=q.get('correct_answer')) for q in data.get('questions', [])]
    except Exception:
        return []


def parse_combined(text: str):
    try:
        if '```' in text:
            text = text.split('```')[1].split('```')[0]
        data = json.loads(text.strip())
        questions, evaluations = [], []
        for r in data.get('results', []):
            questions.append(QuestionInput(question_text=r.get('question_text', ''), student_answer=r.get('student_answer', ''), correct_answer=r.get('correct_answer')))
            evaluations.append(GeminiEvaluation(is_correct=r.get('is_correct', False), score=float(r.get('score', 0)), feedback=r.get('feedback', '')))
        return questions, evaluations
    except Exception:
        return [], []


# --- Reviewer Service ---
class ReviewerService:
    def __init__(self, db: Session):
        self.db = db

    def create_review_from_pdf(self, content: bytes, filename: str, custom_prompt: str = None, marking_scheme=None) -> Review:
        review = self._create_review(InputType.PDF, content, filename)
        try:
            scheme_text, scheme_image = self._get_scheme(marking_scheme)
            if is_scanned_pdf(content):
                self._process_images(review, pdf_to_images(content), custom_prompt, scheme_text, scheme_image)
            else:
                raw_text, questions = parse_pdf(content)
                self._process_text(review, raw_text, questions, custom_prompt, scheme_text)
            self._finalize(review)
        except Exception as e:
            review.status = ReviewStatus.FAILED
            review.error_message = str(e)
            self.db.commit()
        return review

    def create_review_from_image(self, content: bytes, filename: str, custom_prompt: str = None, marking_scheme=None) -> Review:
        review = self._create_review(InputType.IMAGE, content, filename)
        try:
            scheme_text, scheme_image = self._get_scheme(marking_scheme)
            self._process_images(review, [parse_image(content)], custom_prompt, scheme_text, scheme_image)
            self._finalize(review)
        except Exception as e:
            review.status = ReviewStatus.FAILED
            review.error_message = str(e)
            self.db.commit()
        return review

    def create_review_from_text(self, content: str, custom_prompt: str = None, marking_scheme=None) -> Review:
        review = self._create_review(InputType.TEXT, content.encode())
        try:
            scheme_text = self._get_scheme(marking_scheme)[0] if marking_scheme else None
            if custom_prompt:
                q = QuestionInput(question_text="Document Evaluation", student_answer=content)
                e = evaluate_answer(q, custom_prompt, scheme_text)
                self._save_results(review, [q], [e])
            else:
                raw_text, parsed = content, extract_questions(content)
                self._process_text(review, raw_text, parsed, custom_prompt, scheme_text)
            self._finalize(review)
        except Exception as e:
            review.status = ReviewStatus.FAILED
            review.error_message = str(e)
            self.db.commit()
        return review

    def get_review(self, review_id: int) -> Review:
        return self.db.query(Review).filter(Review.id == review_id).first()

    def get_reviews(self, skip: int = 0, limit: int = 100) -> list[Review]:
        return self.db.query(Review).order_by(Review.created_at.desc()).offset(skip).limit(limit).all()

    def _create_review(self, input_type: InputType, content: bytes, filename: str = None) -> Review:
        review = Review(input_type=input_type, original_filename=filename, content_hash=hashlib.sha256(content).hexdigest(), status=ReviewStatus.PROCESSING)
        self.db.add(review)
        self.db.commit()
        self.db.refresh(review)
        return review

    def _get_scheme(self, scheme):
        if not scheme:
            return None, None
        data, stype = scheme
        text, image = None, None
        if stype == 'pdf':
            text, _ = parse_pdf(data)
            if is_scanned_pdf(data):
                imgs = pdf_to_images(data)
                image = imgs[0] if imgs else None
        elif stype == 'image':
            image = parse_image(data)
        elif stype == 'text':
            text = data.decode('utf-8')
        return text, image

    def _process_images(self, review, images, prompt, scheme_text, scheme_image):
        all_q, all_e = [], []
        for img in images:
            q, e = extract_and_evaluate_image(img, prompt, scheme_text, scheme_image)
            all_q.extend(q)
            all_e.extend(e)
        if all_q:
            self._save_results(review, all_q, all_e)
        else:
            review.error_message = "Could not extract content from image"

    def _process_text(self, review, raw_text, parsed, prompt, scheme_text):
        if parsed:
            self._save_results(review, parsed, evaluate_batch(parsed, prompt, scheme_text))
        else:
            q, e = extract_and_evaluate_text(raw_text, scheme_text)
            if q:
                self._save_results(review, q, e)
            else:
                review.error_message = "Could not extract questions from text"

    def _save_results(self, review, questions, evaluations):
        for i, (q, e) in enumerate(zip(questions, evaluations), 1):
            self.db.add(Question(review_id=review.id, question_number=i, question_text=q.question_text, student_answer=q.student_answer, correct_answer=q.correct_answer, is_correct=e.is_correct, score=e.score, feedback=e.feedback))
        self.db.commit()

    def _finalize(self, review):
        self.db.refresh(review)
        if review.questions:
            review.total_questions = len(review.questions)
            review.correct_answers = sum(1 for q in review.questions if q.is_correct)
            review.total_score = sum(q.score or 0 for q in review.questions) / review.total_questions
            review.status = ReviewStatus.COMPLETED
        else:
            review.status = ReviewStatus.FAILED
            review.error_message = review.error_message or "No questions evaluated"
        self.db.commit()
