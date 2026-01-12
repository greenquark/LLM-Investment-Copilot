# UX Path A Backend

FastAPI backend for the Smart Trading Copilot web chat application.

## Setup

1. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment:**
   ```bash
   # Create .env file (copy from .env.example if it exists)
   # Edit .env and set:
   # - OPENAI_API_KEY=your-key-here
   # - OPENAI_MODEL=gpt-5-mini (or your preferred model)
   # - DEBUG=true (to enable debug endpoints and verbose logging)
   # - LOG_LEVEL=DEBUG (for maximum logging verbosity)
   ```
   
   **Quick setup (Windows PowerShell):**
   ```powershell
   # Create .env file with debug enabled
   @"
   OPENAI_API_KEY=your-key-here
   OPENAI_MODEL=gpt-5-mini
   DEBUG=true
   LOG_LEVEL=DEBUG
   "@ | Out-File -FilePath .env -Encoding utf8
   ```

4. **Run the server:**
   ```bash
   # Option 1: Using uvicorn directly (recommended for development)
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   
   # Option 2: Using Python (uses config settings)
   python main.py
   ```

Server will run on `http://localhost:8000` (or `http://127.0.0.1:8000`)

## API Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Database

The database is automatically created on first startup. Tables are created using SQLAlchemy.

To run migrations manually:
```bash
alembic upgrade head
```

## Testing

```bash
pytest ../tests/test_end_to_end.py -v
```

## Project Structure

```
backend/
├── api/              # API endpoints (auth, chat, health)
├── core/             # Core logic
│   ├── orchestrator.py    # LLM orchestration
│   ├── tools/            # Tool implementations
│   ├── guardrails.py     # Safety and cost controls
│   ├── models.py         # Database models
│   ├── database.py       # Database configuration
│   └── prompts.py        # System prompts
├── alembic/          # Database migrations
└── main.py           # FastAPI application
```
