# Railway Solution 2: Repository Root Configuration

## Overview

This solution changes Railway's root directory to the repository root, allowing the Dockerfile to access the entire project including `core/models/`, `core/data/`, `core/strategy/`, etc. This eliminates duplication and naming conflicts.

## Changes Made

### 1. Dockerfile Updated
- **WORKDIR**: Changed to `/app/ux_path_a/backend`
- **COPY**: Now copies entire repository (`COPY . /app/`)
- **PYTHONPATH**: Set to `/app:/app/ux_path_a/backend`
- This ensures project root's `core/` directory is available in Docker

### 2. Imports Simplified
- `data_tools.py`: Simplified imports since project root is now in Docker image
- All imports from `core.models.bar`, `core.data.factory`, etc. should work directly

### 3. Path Calculations
- All existing path calculations still work (they already go up to project root)
- `alembic/env.py` updated with correct comments

## Railway Configuration Required

### Step 1: Change Root Directory

1. Go to Railway dashboard
2. Click on your backend service
3. Go to **Settings** → **Root Directory**
4. Change from: `ux_path_a/backend`
5. Change to: `.` (repository root)
6. Click **Save**

### Step 2: Verify Dockerfile Path

Railway should automatically detect the Dockerfile at:
- `ux_path_a/backend/Dockerfile`

If not, go to **Settings** → **Deploy** → **Dockerfile Path** and set it to:
- `ux_path_a/backend/Dockerfile`

### Step 3: Verify Start Command

The start command should be:
- `./start.sh`

This is already set in `railway.toml`, but verify in **Settings** → **Deploy** → **Start Command**

## What This Solves

✅ **No Duplication**: Single source of truth for all models  
✅ **No Naming Conflicts**: Project root's `core/models/` and backend's `core/models.py` can coexist  
✅ **Natural Imports**: All imports work as expected  
✅ **Future-Proof**: New models in project root automatically available  

## Verification

After deployment, check logs for:
- ✅ "✓ Imported using relative imports" or "✓ Imported using absolute imports"
- ✅ "✓ Base tables created/verified"
- ✅ "Starting server on port..."
- ✅ No `ModuleNotFoundError: No module named 'core.models.bar'` errors

## Rollback

If needed, you can rollback by:
1. Change Railway root directory back to `ux_path_a/backend`
2. Revert the Dockerfile changes
