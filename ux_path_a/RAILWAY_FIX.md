# Railway "No Start Command Found" - Quick Fix

## Immediate Solution

Railway is detecting Python but not auto-detecting FastAPI. Here's how to fix it:

### Option 1: Manual Start Command (Recommended)

1. Go to your Railway project dashboard
2. Click on your backend service
3. Go to **Settings** → **Deploy**
4. Scroll down to **Start Command**
5. Enter this command:
   ```
   alembic upgrade head && uvicorn main:app --host 0.0.0.0 --port $PORT
   ```
6. Click **Save**
7. Railway will automatically redeploy

### Option 2: Use Dockerfile

1. Go to **Settings** → **Deploy**
2. Under **Builder**, select **Dockerfile**
3. Railway will use the Dockerfile instead of Nixpacks
4. The Dockerfile already has the correct start command

### Option 3: Verify Root Directory

**IMPORTANT**: With Solution 2, Railway root directory must be the repository root:

1. Go to **Settings** → **Root Directory**
2. Set it to: `.` (repository root, NOT `ux_path_a/backend`)
3. This allows Dockerfile to access project root's `core/models/`, `core/data/`, etc.
4. Set Dockerfile path to: `ux_path_a/backend/Dockerfile`
5. Save and redeploy

## Why This Happens

Railway's Nixpacks auto-detection looks for:
- `main.py` or `app.py` in the root
- FastAPI imports in those files
- Common FastAPI patterns

**With Solution 2**: We use repository root (`.`) as Railway root directory so the Dockerfile can access the entire project, including project root's `core/models/`, `core/data/`, etc. This eliminates duplication and naming conflicts.

If the root directory isn't set correctly, or if Nixpacks can't parse the file structure, it won't auto-detect.

## Verification

After setting the start command, check the deployment logs:
- You should see: "Running database migrations..."
- Then: "Starting server on port..."
- Finally: "Application startup complete"

If you see errors, check:
- `DATABASE_URL` is set (Railway provides this automatically for PostgreSQL)
- All environment variables are configured
- Root directory is correct

## Still Having Issues?

1. Check Railway logs for specific errors
2. Verify `main.py` exists in `ux_path_a/backend/`
3. Ensure `requirements.txt` includes `fastapi` and `uvicorn`
4. Try using Dockerfile instead of Nixpacks
