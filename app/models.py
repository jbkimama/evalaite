"""
Evalaite - Models
Database setup, SQLAlchemy models, and Pydantic schemas.
"""
from sqlalchemy import create_engine, Column, Integer, String, Text, Boolean, Float, DateTime, ForeignKey, Enum, LargeBinary
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List
import enum
import os
import hashlib
import secrets

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./evalaite.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Enums
class FileType(str, enum.Enum):
    PDF = "pdf"
    IMAGE = "image"
    TEXT = "text"


class ScriptStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class UserRole(str, enum.Enum):
    SUPERADMIN = "superadmin"
    TEACHER = "teacher"


# SQLAlchemy Models
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    role = Column(Enum(UserRole), default=UserRole.TEACHER, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)

    @staticmethod
    def hash_password(password: str) -> str:
        salt = secrets.token_hex(16)
        hash_obj = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        return f"{salt}${hash_obj.hex()}"

    def verify_password(self, password: str) -> bool:
        try:
            salt, hash_value = self.password_hash.split('$')
            hash_obj = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
            return hash_obj.hex() == hash_value
        except:
            return False


class Session(Base):
    __tablename__ = "sessions"
    id = Column(Integer, primary_key=True, index=True)
    token = Column(String(64), unique=True, nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)
    user = relationship("User")


class Subject(Base):
    """Subject/Course (e.g., Mathematics, English)"""
    __tablename__ = "subjects"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    name = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    user = relationship("User")
    exams = relationship("Exam", back_populates="subject", cascade="all, delete-orphan")


class ExamSeries(Base):
    """Groups exams done at the same time (e.g., Mid-term 2024, Final Exam March)"""
    __tablename__ = "exam_series"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    user = relationship("User")
    exams = relationship("Exam", back_populates="series", cascade="all, delete-orphan")


class Exam(Base):
    """An exam for a specific subject in a series"""
    __tablename__ = "exams"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    series_id = Column(Integer, ForeignKey("exam_series.id"), nullable=False)
    subject_id = Column(Integer, ForeignKey("subjects.id"), nullable=False)
    form = Column(String(20), nullable=False)  # Form 1, Form 2, Form 3, Form 4

    # Marking scheme storage
    marking_scheme_filename = Column(String(255), nullable=True)
    marking_scheme_type = Column(Enum(FileType), nullable=True)
    marking_scheme_data = Column(LargeBinary, nullable=True)
    marking_scheme_text = Column(Text, nullable=True)  # Transcribed/editable marking scheme

    # Evaluation prompt
    evaluation_prompt = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User")
    series = relationship("ExamSeries", back_populates="exams")
    subject = relationship("Subject", back_populates="exams")
    scripts = relationship("Script", back_populates="exam", cascade="all, delete-orphan")


class Script(Base):
    """Student answer script/booklet"""
    __tablename__ = "scripts"
    id = Column(Integer, primary_key=True, index=True)
    exam_id = Column(Integer, ForeignKey("exams.id"), nullable=False)

    # Student info (extracted from script)
    student_name = Column(String(255), nullable=True)
    student_number = Column(String(100), nullable=True)

    # Script file storage
    filename = Column(String(255), nullable=True)
    file_type = Column(Enum(FileType), nullable=False)
    file_data = Column(LargeBinary, nullable=True)

    # Evaluation results
    status = Column(Enum(ScriptStatus), default=ScriptStatus.PENDING)
    total_score = Column(Float, nullable=True)
    total_questions = Column(Integer, default=0)
    correct_answers = Column(Integer, default=0)
    error_message = Column(Text, nullable=True)

    evaluated_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    exam = relationship("Exam", back_populates="scripts")
    questions = relationship("ScriptQuestion", back_populates="script", cascade="all, delete-orphan")


class ScriptQuestion(Base):
    """Individual question evaluation from a script"""
    __tablename__ = "script_questions"
    id = Column(Integer, primary_key=True, index=True)
    script_id = Column(Integer, ForeignKey("scripts.id"), nullable=False)
    question_number = Column(Integer, nullable=False)
    question_text = Column(Text, nullable=False)
    student_answer = Column(Text, nullable=False)
    correct_answer = Column(Text, nullable=True)
    is_correct = Column(Boolean, nullable=True)
    score = Column(Float, nullable=True)
    feedback = Column(Text, nullable=True)
    script = relationship("Script", back_populates="questions")


class SavedPrompt(Base):
    """Saved evaluation prompts"""
    __tablename__ = "saved_prompts"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    name = Column(String(255), nullable=False)
    prompt = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    user = relationship("User")


# Pydantic Schemas

# Subject Schemas
class SubjectCreate(BaseModel):
    name: str


class SubjectResponse(BaseModel):
    id: int
    name: str
    created_at: datetime

    class Config:
        from_attributes = True


# Series Schemas
class SeriesCreate(BaseModel):
    name: str
    description: Optional[str] = None


class SeriesResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    created_at: datetime
    exam_count: int = 0

    class Config:
        from_attributes = True


# Exam Schemas
class ExamCreate(BaseModel):
    series_id: int
    subject_id: int
    form: str  # Form 1, Form 2, Form 3, Form 4
    evaluation_prompt: Optional[str] = None


class ExamResponse(BaseModel):
    id: int
    series_id: int
    subject_id: int
    form: Optional[str] = None
    subject_name: str = ""
    series_name: str = ""
    has_marking_scheme: bool = False
    marking_scheme_filename: Optional[str] = None
    marking_scheme_text: Optional[str] = None
    evaluation_prompt: Optional[str] = None
    script_count: int = 0
    evaluated_count: int = 0
    average_score: Optional[float] = None
    created_at: datetime

    class Config:
        from_attributes = True


# Script Schemas
class ScriptResponse(BaseModel):
    id: int
    exam_id: int
    student_name: Optional[str]
    student_number: Optional[str]
    filename: Optional[str]
    file_type: FileType
    status: ScriptStatus
    total_score: Optional[float]
    total_questions: int
    correct_answers: int
    error_message: Optional[str]
    evaluated_at: Optional[datetime]
    created_at: datetime

    class Config:
        from_attributes = True


class ScriptDetailResponse(ScriptResponse):
    questions: List["QuestionResult"] = []


# Question Schemas
class QuestionInput(BaseModel):
    question_text: str
    student_answer: str
    correct_answer: Optional[str] = None


class GeminiEvaluation(BaseModel):
    is_correct: bool
    score: float
    feedback: str


class QuestionResult(BaseModel):
    id: int
    question_number: int
    question_text: str
    student_answer: str
    correct_answer: Optional[str]
    is_correct: Optional[bool]
    score: Optional[float]
    feedback: Optional[str]

    class Config:
        from_attributes = True


# Auth Schemas
class UserCreate(BaseModel):
    username: str
    password: str
    role: str = "teacher"


class UserUpdate(BaseModel):
    username: Optional[str] = None
    password: Optional[str] = None
    role: Optional[str] = None
    is_active: Optional[bool] = None


class UserLogin(BaseModel):
    username: str
    password: str


class UserResponse(BaseModel):
    id: int
    username: str
    role: UserRole
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


# Update forward references
ScriptDetailResponse.model_rebuild()
