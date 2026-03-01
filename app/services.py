"""
Evalaite - Services
Gemini AI, file parsing, and exam evaluation logic.
"""
import json
import io
import re
from datetime import datetime
from google import genai
from google.genai import types
from PIL import Image
import pdfplumber
from sqlalchemy.orm import Session

import os
from dotenv import load_dotenv
from app.models import (
    Exam, Script, ScriptQuestion, FileType, ScriptStatus,
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


def get_file_type(content_type: str, filename: str) -> FileType | None:
    type_map = {
        'application/pdf': FileType.PDF,
        'image/png': FileType.IMAGE, 'image/jpeg': FileType.IMAGE, 'image/jpg': FileType.IMAGE,
        'text/plain': FileType.TEXT
    }
    if content_type in type_map:
        return type_map[content_type]
    ext = filename.lower().split('.')[-1] if filename else ''
    ext_map = {'pdf': FileType.PDF, 'png': FileType.IMAGE, 'jpg': FileType.IMAGE, 'jpeg': FileType.IMAGE, 'txt': FileType.TEXT}
    return ext_map.get(ext)


# --- Gemini AI Functions ---
MODEL = "gemini-2.5-flash"


def image_to_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    return buf.getvalue()


def transcribe_marking_scheme(file_data: bytes, file_type: FileType) -> str:
    """Transcribe marking scheme from PDF or image using Gemini AI."""
    prompt = """Transcribe this marking scheme document completely and accurately.

Extract ALL questions, answers, and point allocations exactly as they appear.
Format the output clearly with:
- Question numbers
- Expected answers or answer guidelines
- Point values for each question/part

Be thorough and include every detail from the marking scheme.
If there are multiple acceptable answers, include all of them.
Preserve the structure (e.g., Question 1a, 1b, 2a, etc.)

Output the transcription in a clean, readable format."""

    try:
        if file_type == FileType.PDF:
            if is_scanned_pdf(file_data):
                images = pdf_to_images(file_data)
                contents = [prompt]
                for img in images:
                    contents.append(types.Part.from_bytes(data=image_to_bytes(img), mime_type="image/png"))
            else:
                text, _ = parse_pdf(file_data)
                contents = f"{prompt}\n\nDocument content:\n{text}"
        else:  # Image
            img = parse_image(file_data)
            contents = [
                prompt,
                types.Part.from_bytes(data=image_to_bytes(img), mime_type="image/png")
            ]

        response = get_client().models.generate_content(model=MODEL, contents=contents)
        return response.text.strip()
    except Exception as e:
        return f"Error transcribing marking scheme: {str(e)}"


def extract_student_info(content, content_type: str) -> dict:
    """Extract student name and number from answer booklet using Gemini."""
    try:
        prompt = """Look at this document and extract the student's information.
Find the student name and student number/ID if present.

Respond in JSON format:
{"student_name": "name or null", "student_number": "number or null"}

Only return the JSON, nothing else."""

        if content_type == 'image':
            img_bytes = image_to_bytes(content) if isinstance(content, Image.Image) else content
            contents = [
                types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
                prompt
            ]
        elif content_type == 'images':
            img_bytes = image_to_bytes(content[0]) if content else None
            if not img_bytes:
                return {"student_name": None, "student_number": None}
            contents = [
                types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
                prompt
            ]
        else:
            contents = f"{prompt}\n\nDocument content:\n{content[:2000]}"

        response = get_client().models.generate_content(model=MODEL, contents=contents)
        text = response.text.strip()

        if '```' in text:
            text = text.split('```')[1].split('```')[0]
            if text.startswith('json'):
                text = text[4:]
        data = json.loads(text.strip())
        return {
            "student_name": data.get("student_name") if data.get("student_name") not in [None, "null", ""] else None,
            "student_number": data.get("student_number") if data.get("student_number") not in [None, "null", ""] else None
        }
    except Exception:
        return {"student_name": None, "student_number": None}


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


# --- Exam Service ---
class ExamService:
    def __init__(self, db: Session):
        self.db = db

    def get_scheme_content(self, exam: Exam) -> tuple[str | None, Image.Image | None]:
        """Get marking scheme content from exam."""
        if not exam.marking_scheme_data:
            return None, None

        text, image = None, None
        if exam.marking_scheme_type == FileType.PDF:
            text, _ = parse_pdf(exam.marking_scheme_data)
            if is_scanned_pdf(exam.marking_scheme_data):
                imgs = pdf_to_images(exam.marking_scheme_data)
                image = imgs[0] if imgs else None
        elif exam.marking_scheme_type == FileType.IMAGE:
            image = parse_image(exam.marking_scheme_data)
        elif exam.marking_scheme_type == FileType.TEXT:
            text = exam.marking_scheme_data.decode('utf-8')
        return text, image

    def evaluate_script(self, script: Script) -> Script:
        """Evaluate a single script."""
        exam = script.exam
        script.status = ScriptStatus.PROCESSING
        self.db.commit()

        try:
            scheme_text, scheme_image = self.get_scheme_content(exam)
            custom_prompt = exam.evaluation_prompt

            # Parse script content based on type
            if script.file_type == FileType.PDF:
                if is_scanned_pdf(script.file_data):
                    images = pdf_to_images(script.file_data)
                    # Extract student info
                    if images:
                        student_info = extract_student_info(images, 'images')
                        script.student_name = student_info.get("student_name")
                        script.student_number = student_info.get("student_number")
                    self._process_images(script, images, custom_prompt, scheme_text, scheme_image)
                else:
                    raw_text, questions = parse_pdf(script.file_data)
                    student_info = extract_student_info(raw_text, 'text')
                    script.student_name = student_info.get("student_name")
                    script.student_number = student_info.get("student_number")
                    self._process_text(script, raw_text, questions, custom_prompt, scheme_text)

            elif script.file_type == FileType.IMAGE:
                image = parse_image(script.file_data)
                student_info = extract_student_info(image, 'image')
                script.student_name = student_info.get("student_name")
                script.student_number = student_info.get("student_number")
                self._process_images(script, [image], custom_prompt, scheme_text, scheme_image)

            elif script.file_type == FileType.TEXT:
                content = script.file_data.decode('utf-8')
                student_info = extract_student_info(content, 'text')
                script.student_name = student_info.get("student_name")
                script.student_number = student_info.get("student_number")
                if custom_prompt:
                    q = QuestionInput(question_text="Document Evaluation", student_answer=content)
                    e = evaluate_answer(q, custom_prompt, scheme_text)
                    self._save_results(script, [q], [e])
                else:
                    raw_text, parsed = content, extract_questions(content)
                    self._process_text(script, raw_text, parsed, custom_prompt, scheme_text)

            self._finalize(script)

        except Exception as e:
            script.status = ScriptStatus.FAILED
            script.error_message = str(e)
            self.db.commit()

        return script

    def _process_images(self, script, images, prompt, scheme_text, scheme_image):
        all_q, all_e = [], []
        for img in images:
            q, e = extract_and_evaluate_image(img, prompt, scheme_text, scheme_image)
            all_q.extend(q)
            all_e.extend(e)
        if all_q:
            self._save_results(script, all_q, all_e)
        else:
            script.error_message = "Could not extract content from image"

    def _process_text(self, script, raw_text, parsed, prompt, scheme_text):
        if parsed:
            self._save_results(script, parsed, evaluate_batch(parsed, prompt, scheme_text))
        else:
            q, e = extract_and_evaluate_text(raw_text, scheme_text)
            if q:
                self._save_results(script, q, e)
            else:
                script.error_message = "Could not extract questions from text"

    def _save_results(self, script, questions, evaluations):
        for i, (q, e) in enumerate(zip(questions, evaluations), 1):
            self.db.add(ScriptQuestion(
                script_id=script.id,
                question_number=i,
                question_text=q.question_text,
                student_answer=q.student_answer,
                correct_answer=q.correct_answer,
                is_correct=e.is_correct,
                score=e.score,
                feedback=e.feedback
            ))
        self.db.commit()

    def _finalize(self, script):
        self.db.refresh(script)
        if script.questions:
            script.total_questions = len(script.questions)
            script.correct_answers = sum(1 for q in script.questions if q.is_correct)
            script.total_score = sum(q.score or 0 for q in script.questions) / script.total_questions
            script.status = ScriptStatus.COMPLETED
        else:
            script.status = ScriptStatus.FAILED
            script.error_message = script.error_message or "No questions evaluated"
        script.evaluated_at = datetime.utcnow()
        self.db.commit()

    def evaluate_all_scripts(self, exam: Exam) -> list[Script]:
        """Evaluate all pending scripts for an exam."""
        pending_scripts = [s for s in exam.scripts if s.status == ScriptStatus.PENDING]
        for script in pending_scripts:
            self.evaluate_script(script)
        return pending_scripts
