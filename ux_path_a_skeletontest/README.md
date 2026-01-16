# UX Path A Skeleton Test

Purpose: **minimal backend + frontend** with *only* health endpoints, to debug external accessibility (Railway Edge) without any DB/migrations/LLM/tools.

## Backend (Railway)
- Entrypoint: `ux_path_a_skeletontest/backend/main.py`
- Health: `GET /api/health` (and `/api/health/`)
- Start script respects Railway `PORT` env var.

## Frontend (Vercel)
- Next.js app that calls `/api/health` (proxied via rewrite to the backend).
- Configure `NEXT_PUBLIC_BACKEND_URL` in Vercel (example below).

### Vercel env var
- `NEXT_PUBLIC_BACKEND_URL=https://<your-railway-backend-domain>`

## Local run

Backend:
```bash
cd ux_path_a_skeletontest/backend
python -m pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Frontend:
```bash
cd ux_path_a_skeletontest/frontend
npm install
npm run dev
```

