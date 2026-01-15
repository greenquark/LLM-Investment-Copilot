# Quick Deployment Checklist

## Pre-Deployment

- [ ] All code committed to GitHub
- [ ] Environment variables documented
- [ ] Database migrations tested locally
- [ ] Backend runs locally without errors
- [ ] Frontend builds successfully (`npm run build`)

## Railway Backend Setup (5 minutes)

1. **Create Railway Account**
   - Go to https://railway.app
   - Sign in with GitHub

2. **Deploy Backend**
   - Click "New Project" → "Deploy from GitHub repo"
   - Select your repository
   - **IMPORTANT**: Go to Settings → Root Directory
   - Set root directory to: `.` (repository root)
   - This allows access to project root's `core/models/`, `core/data/`, etc.

3. **Add PostgreSQL**
   - Click "New" → "Database" → "Add PostgreSQL"
   - Railway provides `DATABASE_URL` automatically

4. **Set Environment Variables**
   ```
   OPENAI_API_KEY=sk-...
   OPENAI_MODEL=gpt-5-mini
   CORS_ORIGINS=http://localhost:3000
   SECRET_KEY=<generate-random-32-char-string>
   DEBUG=false
   LOG_LEVEL=INFO
   ```

5. **Configure Dockerfile**
   - Go to Settings → Deploy
   - Set Dockerfile path to: `ux_path_a/backend/Dockerfile`
   - Or select "Dockerfile" as builder
   - Start command should be: `./start.sh` (from `railway.toml`)

6. **Get Backend URL**
   - Settings → Networking → Generate Domain
   - Copy URL: `https://your-backend.railway.app`

## Vercel Frontend Setup (3 minutes)

1. **Create Vercel Account**
   - Go to https://vercel.com
   - Sign in with GitHub

2. **Deploy Frontend**
   - Click "Add New Project"
   - Import GitHub repository
   - Root Directory: `ux_path_a/frontend`
   - Framework: Next.js (auto-detected)

3. **Set Environment Variable**
   ```
   NEXT_PUBLIC_API_URL=https://your-backend.railway.app
   ```
   (Use your Railway URL from step above)

4. **Deploy**
   - Click "Deploy"
   - Get frontend URL: `https://your-app.vercel.app`

## Final Step

1. **Update CORS in Railway**
   - Go back to Railway backend settings
   - Update `CORS_ORIGINS`:
     ```
     CORS_ORIGINS=https://your-app.vercel.app,http://localhost:3000
     ```
   - Railway auto-redeploys

## Test

1. Visit your Vercel URL
2. Login/Register
3. Send a test message
4. Verify charts render

## Local Development

To develop locally while keeping cloud deployments:

**Backend** (`ux_path_a/backend/.env`):
```env
DATABASE_URL=sqlite:///./ux_path_a.db
CORS_ORIGINS=http://localhost:3000
DEBUG=true
```

**Frontend** (`ux_path_a/frontend/.env.local`):
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

Run locally:
```bash
# Backend
cd ux_path_a/backend
uvicorn main:app --reload

# Frontend (new terminal)
cd ux_path_a/frontend
npm run dev
```

## Troubleshooting

- **CORS errors**: Check `CORS_ORIGINS` includes your Vercel URL
- **Database errors**: Verify `DATABASE_URL` is set in Railway
- **API connection**: Check `NEXT_PUBLIC_API_URL` matches Railway URL
- **Build errors**: Check Railway/Vercel logs

See `DEPLOYMENT.md` for detailed instructions.
