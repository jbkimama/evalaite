"""
Evalaite - Models
Database setup, SQLAlchemy models, and Pydantic schemas.
"""
from sqlalchemy import create_engine, Column, Integer, String, Text, Boolean, Float, DateTime, ForeignKey, Enum
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from pydantic import BaseModel
from datetime import datetime
from typing import Optional
import enum
import os

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
class InputType(str, enum.Enum):
    PDF = "pdf"
    IMAGE = "image"
    TEXT = "text"


class ReviewStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# SQLAlchemy Models
class Review(Base):
    __tablename__ = "reviews"
    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    input_type = Column(Enum(InputType), nullable=False)
    original_filename = Column(String(255), nullable=True)
    content_hash = Column(String(64), nullable=True)
    status = Column(Enum(ReviewStatus), default=ReviewStatus.PENDING)
    total_score = Column(Float, nullable=True)
    total_questions = Column(Integer, default=0)
    correct_answers = Column(Integer, default=0)
    error_message = Column(Text, nullable=True)
    questions = relationship("Question", back_populates="review", cascade="all, delete-orphan")


class Question(Base):
    __tablename__ = "questions"
    id = Column(Integer, primary_key=True, index=True)
    review_id = Column(Integer, ForeignKey("reviews.id"), nullable=False)
    question_number = Column(Integer, nullable=False)
    question_text = Column(Text, nullable=False)
    student_answer = Column(Text, nullable=False)
    correct_answer = Column(Text, nullable=True)
    is_correct = Column(Boolean, nullable=True)
    score = Column(Float, nullable=True)
    feedback = Column(Text, nullable=True)
    review = relationship("Review", back_populates="questions")


class SavedPrompt(Base):
    __tablename__ = "saved_prompts"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), unique=True, nullable=False)
    prompt = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


# Pydantic Schemas
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


class ReviewResponse(BaseModel):
    id: int
    created_at: datetime
    input_type: InputType
    original_filename: Optional[str]
    status: ReviewStatus
    total_score: Optional[float]
    total_questions: int
    correct_answers: int
    error_message: Optional[str]
    questions: list[QuestionResult] = []

    class Config:
        from_attributes = True


class ReviewListItem(BaseModel):
    id: int
    created_at: datetime
    input_type: InputType
    original_filename: Optional[str]
    status: ReviewStatus
    total_score: Optional[float]
    total_questions: int
    correct_answers: int

    class Config:
        from_attributes = True
