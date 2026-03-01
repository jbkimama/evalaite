# Evalaite

AI-powered exam evaluation system using Google Gemini. Built as a proof of concept for high school student assessment.

## Features

- Evaluate student work (letters, essays, compositions, math) using AI
- Support for PDF and image uploads
- Automatic extraction of student name and number from booklets
- Custom evaluation prompts saved to database
- Batch evaluation with results ranked by score
- Export results as CSV
- Multi-user authentication with role-based access (Superadmin, Teacher)
- Admin panel for user management

## Default Users

On first run, the following default users are created:

| Role | Username | Password |
|------|----------|----------|
| Superadmin | `admin` | `admin123` |
| Teacher | `teacher` | `teacher123` |

**Important:** Change these passwords after first login!

## Quick Start

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- Google Gemini API key

### Installation

```bash
cd evalaite

# Install dependencies
uv sync

# Create .env file with your API key
echo GEMINI_API_KEY=your_api_key_here > .env
```

### Run

**Linux/Mac:**
```bash
uv run uvicorn app.main:app --reload --port 8000
```

**Windows (PowerShell):**
```powershell
uv run uvicorn app.main:app --reload --port 8000
```

**Windows (CMD):**
```cmd
uv run uvicorn app.main:app --reload --port 8000
```

Open http://localhost:8000 in your browser.

## Usage

1. **Select or create an evaluation prompt** - Choose from presets or write custom criteria
2. **Upload marking scheme** (optional) - Add teacher's marking guide
3. **Upload student booklets** - Add one or more student answer files
4. **Click "Review Exams"** - AI evaluates each submission
5. **View results** - See scores and detailed feedback

## Project Structure

```
evalaite/
├── app/
│   ├── main.py           # FastAPI app & routes
│   ├── models.py         # Database & Pydantic models
│   └── services.py       # Gemini AI & review logic
├── frontend/
│   ├── templates/        # HTML templates
│   └── static/css/       # Styles
├── pyproject.toml        # Dependencies
└── README.md
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/reviews/upload` | Upload file for evaluation |
| GET | `/api/reviews/{id}` | Get evaluation result |
| GET | `/api/reviews/` | List all evaluations |
| GET | `/api/reviews/prompts/saved` | Get saved prompts |
| POST | `/api/reviews/prompts/save` | Save custom prompt |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `GEMINI_API_KEY` | Google Gemini API key (required) |
| `DATABASE_URL` | SQLite database path (default: `sqlite:///./evalaite.db`) |

## Tech Stack

- **FastAPI** - Web framework
- **Google Gemini** - AI evaluation (gemini-2.0-flash)
- **SQLite + SQLAlchemy** - Database
- **Jinja2** - Templates
- **pdfplumber** - PDF processing
- **Pillow** - Image processing

## License

MIT
