# Next Steps After Railway Deployment

Based on your Railway dashboard, your backend is successfully deployed! Here's what to do next:

## ✅ Current Status
- ✅ Backend service is deployed and online
- ✅ Postgres database is running
- ✅ Deployment successful

## 🔴 Action Required: Expose Your Service

Your service shows as **"Unexposed service"** - you need to expose it to get a public URL.

### Steps to Expose:

1. In Railway dashboard, click on your **LLM-Investment-Copilot** service
2. Go to **Settings** → **Networking**
3. Scroll down to **"Public Networking"** section
4. Click **"Generate Domain"** button
5. Railway will create a public URL like: `https://your-service-name.railway.app`
6. **Copy this URL** - you'll need it for the frontend!

## 📋 Next Steps Checklist

### 1. Set Environment Variables (Railway)
Go to your service → **Variables** tab and add:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-5-mini
CORS_ORIGINS=http://localhost:3000
SECRET_KEY=<generate-random-32-char-string>
DEBUG=false
LOG_LEVEL=INFO
```

**Generate SECRET_KEY:**
```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### 2. Verify Backend is Working
1. Click **"View logs"** on your active deployment
2. Check for:
   - ✅ "Running database migrations..."
   - ✅ "Starting server on port..."
   - ✅ "Application startup complete"
3. Visit your Railway URL in browser - you should see API docs at `/docs`

### 3. Deploy Frontend to Vercel
1. Go to [Vercel](https://vercel.com)
2. Import your GitHub repository
3. Set **Root Directory**: `ux_path_a/frontend`
4. Add environment variable:
   ```
   NEXT_PUBLIC_API_URL=https://your-railway-url.railway.app
   ```
   (Use the URL from step 1 above)
5. Deploy

### 4. Update CORS in Railway
After Vercel deployment, update Railway's `CORS_ORIGINS`:
```
CORS_ORIGINS=https://your-vercel-app.vercel.app,http://localhost:3000
```

### 5. Test Everything
1. Visit your Vercel frontend URL
2. Try logging in
3. Send a test message
4. Verify charts render

## 🐛 Troubleshooting

**Service won't start?**
- Check logs: Click "View logs" on deployment
- Verify root directory is set to: `.` (repository root) in Settings → Root Directory
- Verify Dockerfile path is: `ux_path_a/backend/Dockerfile` in Settings → Deploy
- Verify start command is set: Settings → Deploy → Start Command
- Should be: `./start.sh` (from `railway.toml`)

**Can't access backend?**
- Make sure service is exposed (see "Expose Your Service" above)
- Check that `DATABASE_URL` is set (Railway provides this automatically)

**Database errors?**
- Verify Postgres service is running (green dot)
- Check logs for migration errors
- Ensure `DATABASE_URL` environment variable exists

## 📚 Reference
- Full deployment guide: `ux_path_a/DEPLOYMENT.md`
- Quick deploy checklist: `ux_path_a/QUICK_DEPLOY.md`
- Railway troubleshooting: `ux_path_a/RAILWAY_FIX.md`
