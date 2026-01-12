# How to Enable Debug Mode

## Quick Fix

The debug endpoint is returning 404 because `DEBUG=false` by default. Follow these steps:

### Step 1: Create/Edit `.env` file

In `ux_path_a/backend/` directory, create or edit `.env` file:

```bash
DEBUG=true
LOG_LEVEL=DEBUG
OPENAI_API_KEY=your-key-here
OPENAI_MODEL=gpt-5-mini
```

**PowerShell command:**
```powershell
cd ux_path_a\backend
@"
DEBUG=true
LOG_LEVEL=DEBUG
OPENAI_API_KEY=your-key-here
OPENAI_MODEL=gpt-5-mini
"@ | Out-File -FilePath .env -Encoding utf8
```

### Step 2: Restart the Backend Server

**Important:** The server must be restarted to load the `.env` file!

1. Stop the current server (Ctrl+C in the terminal running uvicorn)
2. Start it again:
   ```bash
   cd ux_path_a\backend
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

### Step 3: Verify Debug Mode is Enabled

After restarting, you should see in the logs:
- More verbose logging
- Request logging for each API call

Test the debug endpoint:
```bash
curl http://localhost:8000/api/health/debug
```

You should now get either:
- **200 OK** with debug information (if DEBUG=true)
- **403 Forbidden** with a helpful error message (if DEBUG=false but route exists)

If you still get **404 Not Found**, the server hasn't reloaded - try restarting again.

## Alternative: Set Environment Variables Directly

Instead of using `.env` file, you can set environment variables when starting the server:

**PowerShell:**
```powershell
$env:DEBUG="true"; $env:LOG_LEVEL="DEBUG"; uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Linux/Mac:**
```bash
DEBUG=true LOG_LEVEL=DEBUG uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Verify It's Working

1. Check backend logs - you should see request logging like:
   ```
   📥 GET /api/health/debug
   📤 GET /api/health/debug - 200 (0.003s)
   ```

2. Test the endpoint:
   ```bash
   curl http://localhost:8000/api/health/debug
   ```

3. Check Swagger UI: Visit `http://localhost:8000/docs` and look for the `/api/health/debug` endpoint
