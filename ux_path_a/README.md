# UX Path A - Web Chat App (Smart Trading Copilot)

This is the UX Path A implementation - a standalone ChatGPT-style web application for conversational market analysis.

## Architecture

- **Frontend**: Next.js 14+ (React) - `ux_path_a/frontend/`
- **Backend**: FastAPI (Python) - `ux_path_a/backend/`
- **Shared**: Common types/constants - `ux_path_a/shared/`

## Quick Start

### Backend Setup

```bash
cd ux_path_a/backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

### Frontend Setup

```bash
cd ux_path_a/frontend
npm install
npm run dev
```

## Project Structure

See `plan/UX_PATH_A_PROJECT_PLAN.md` for detailed architecture and implementation plan.

## Development Status

- ✅ Project plan created
- ✅ Checklist created
- 🚧 Project structure setup (in progress)
- ⏳ Frontend foundation
- ⏳ Backend foundation
- ⏳ Tool integration
