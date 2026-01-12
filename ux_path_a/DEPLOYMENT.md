# Deployment Guide: Vercel + Railway

This guide will help you deploy the Investment Copilot application to Vercel (frontend) and Railway (backend).

## Prerequisites

- GitHub account
- Vercel account (free tier available)
- Railway account (free trial available)
- OpenAI API key

## Architecture

- **Frontend**: Next.js app deployed on Vercel
- **Backend**: FastAPI app deployed on Railway
- **Database**: PostgreSQL (provided by Railway)

## Step 1: Prepare Your Repository

1. Ensure all code is committed and pushed to GitHub:
   ```bash
   git add .
   git commit -m "Prepare for deployment"
   git push origin main
   ```

## Step 2: Deploy Backend to Railway

### 2.1 Create Railway Project

1. Go to [Railway](https://railway.app) and sign in
2. Click "New Project"
3. Select "Deploy from GitHub repo"
4. Choose your repository
5. Railway will detect the backend directory automatically

### 2.2 Configure Backend Service

1. In Railway dashboard, click on your service
2. Go to "Settings" → "Root Directory"
3. Set root directory to: `ux_path_a/backend`
4. Go to "Settings" → "Deploy"
5. Set start command to: `uvicorn main:app --host 0.0.0.0 --port $PORT`

### 2.3 Add PostgreSQL Database

1. In Railway dashboard, click "New" → "Database" → "Add PostgreSQL"
2. Railway will automatically create a PostgreSQL database
3. Note the connection string (Railway provides it as `DATABASE_URL`)

### 2.4 Set Environment Variables

In Railway service settings, add these environment variables:

```env
# OpenAI
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-5-mini

# CORS (will be updated after frontend deployment)
CORS_ORIGINS=http://localhost:3000

# Security
SECRET_KEY=generate-a-random-secret-key-here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Server
DEBUG=false
LOG_LEVEL=INFO

# Database (Railway automatically provides DATABASE_URL)
# No need to set manually - Railway injects it
```

**Important**: Generate a secure SECRET_KEY:
```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### 2.5 Deploy and Get Backend URL

1. Railway will automatically deploy when you push to GitHub
2. Go to "Settings" → "Networking"
3. Click "Generate Domain" to get your backend URL
4. Copy the URL (e.g., `https://your-backend.railway.app`)

### 2.6 Run Database Migrations

Railway will automatically run migrations on startup (configured in Dockerfile).
If you need to run manually:

1. Go to Railway service → "Deployments"
2. Click on the latest deployment
3. Open "View Logs" to see migration output

## Step 3: Deploy Frontend to Vercel

### 3.1 Create Vercel Project

1. Go to [Vercel](https://vercel.com) and sign in
2. Click "Add New Project"
3. Import your GitHub repository
4. Configure project:
   - **Framework Preset**: Next.js
   - **Root Directory**: `ux_path_a/frontend`
   - **Build Command**: `npm run build` (default)
   - **Output Directory**: `.next` (default)

### 3.2 Set Environment Variables

In Vercel project settings → Environment Variables, add:

```env
NEXT_PUBLIC_API_URL=https://your-backend.railway.app
```

Replace `your-backend.railway.app` with your actual Railway backend URL.

### 3.3 Deploy

1. Click "Deploy"
2. Vercel will build and deploy your frontend
3. Note your frontend URL (e.g., `https://your-app.vercel.app`)

## Step 4: Update CORS Configuration

After both deployments are complete:

1. Go back to Railway backend settings
2. Update `CORS_ORIGINS` environment variable:
   ```env
   CORS_ORIGINS=https://your-app.vercel.app,http://localhost:3000
   ```
3. Railway will automatically redeploy

## Step 5: Verify Deployment

1. Visit your Vercel frontend URL
2. Test the application:
   - Login/Register
   - Send a chat message
   - Verify charts render correctly
3. Check Railway logs for any errors

## Local Development Setup

To continue developing locally while using cloud deployments:

### Backend (.env file in `ux_path_a/backend/`)

```env
# Use SQLite for local development
DATABASE_URL=sqlite:///./ux_path_a.db

# Use local server
HOST=0.0.0.0
PORT=8000
DEBUG=true
LOG_LEVEL=DEBUG

# CORS for local frontend
CORS_ORIGINS=http://localhost:3000,http://localhost:3001

# Your OpenAI key
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-5-mini

# Security (use a different key for local)
SECRET_KEY=local-dev-secret-key
```

### Frontend (.env.local file in `ux_path_a/frontend/`)

```env
# Point to local backend
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### Running Locally

**Backend:**
```bash
cd ux_path_a/backend
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Frontend:**
```bash
cd ux_path_a/frontend
npm install
npm run dev
```

## Troubleshooting

### Backend Issues

1. **Database Connection Errors**
   - Verify `DATABASE_URL` is set correctly in Railway
   - Check Railway PostgreSQL service is running
   - Review migration logs in Railway

2. **CORS Errors**
   - Verify `CORS_ORIGINS` includes your Vercel URL
   - Check frontend `NEXT_PUBLIC_API_URL` matches Railway URL
   - Ensure no trailing slashes in URLs

3. **Port Issues**
   - Railway sets `PORT` automatically - don't override it
   - Backend should listen on `0.0.0.0`, not `127.0.0.1`

### Frontend Issues

1. **API Connection Errors**
   - Verify `NEXT_PUBLIC_API_URL` is set in Vercel
   - Check backend is accessible (visit Railway URL in browser)
   - Review browser console for CORS errors

2. **Build Errors**
   - Check Node.js version (should be 18+)
   - Review build logs in Vercel dashboard
   - Ensure all dependencies are in `package.json`

## Environment Variable Reference

### Backend (Railway)

| Variable | Description | Example |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | Auto-provided by Railway |
| `OPENAI_API_KEY` | OpenAI API key | `sk-...` |
| `OPENAI_MODEL` | LLM model to use | `gpt-5-mini` |
| `CORS_ORIGINS` | Allowed frontend origins | `https://app.vercel.app,http://localhost:3000` |
| `SECRET_KEY` | JWT secret key | Random 32+ character string |
| `PORT` | Server port | Auto-set by Railway |
| `DEBUG` | Debug mode | `false` for production |
| `LOG_LEVEL` | Logging level | `INFO` for production |

### Frontend (Vercel)

| Variable | Description | Example |
|----------|-------------|---------|
| `NEXT_PUBLIC_API_URL` | Backend API URL | `https://backend.railway.app` |

## Cost Estimates

- **Vercel**: Free tier includes 100GB bandwidth/month
- **Railway**: $5/month starter plan (includes PostgreSQL)
- **Total**: ~$5/month for basic deployment

## Security Checklist

- [ ] Use strong `SECRET_KEY` (32+ random characters)
- [ ] Set `DEBUG=false` in production
- [ ] Use HTTPS (automatic with Vercel/Railway)
- [ ] Limit `CORS_ORIGINS` to your domains only
- [ ] Keep `OPENAI_API_KEY` secret (never commit to Git)
- [ ] Use environment variables for all secrets
- [ ] Enable Railway's automatic backups for database

## Monitoring

### Railway Logs
- View real-time logs in Railway dashboard
- Check deployment logs for errors
- Monitor database connection status

### Vercel Analytics
- View deployment logs
- Monitor build times
- Check function execution logs

## Updating Deployments

### Backend Updates
1. Push changes to GitHub
2. Railway automatically redeploys
3. Check logs for any errors

### Frontend Updates
1. Push changes to GitHub
2. Vercel automatically redeploys
3. Preview deployments available for PRs

## Rollback

### Railway
1. Go to "Deployments" tab
2. Find previous successful deployment
3. Click "Redeploy"

### Vercel
1. Go to "Deployments" tab
2. Find previous deployment
3. Click "..." → "Promote to Production"

## Support

- Railway Docs: https://docs.railway.app
- Vercel Docs: https://vercel.com/docs
- Project Issues: Check GitHub issues
