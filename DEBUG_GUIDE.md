# Debugging Guide

This guide provides tools and techniques for debugging the UX Path A application.

## Frontend Debugging

### Browser DevTools Utilities

When running in development mode, the following utilities are available in the browser console:

```javascript
// Test backend connection
window.__API_DEBUG__.testConnection()

// Enable/disable API request logging
window.__API_DEBUG__.enableLogging()
window.__API_DEBUG__.disableLogging()

// Get current API base URL
window.__API_DEBUG__.getBaseUrl()

// Set API base URL (for testing)
window.__API_DEBUG__.setBaseUrl('http://localhost:8000')

// Check if token is set (masked)
window.__API_DEBUG__.getToken()
```

### Connection Test Component

A connection test component is available at `components/debug/ConnectionTest.tsx`. You can add it to your page to test backend connectivity:

```tsx
import ConnectionTest from '@/components/debug/ConnectionTest'

// In your component
<ConnectionTest />
```

### Enable API Logging

API request/response logging is automatically enabled in development mode. To enable it manually:

```javascript
window.__API_DEBUG__.enableLogging()
```

This will log:
- Request URL, method, headers, and body
- Response status, duration, and data
- Error details

## Backend Debugging

### Enable Debug Mode

**Option 1: Create/Edit `.env` file in `ux_path_a/backend/`:**

```bash
# Copy the example file
cp .env.example .env

# Edit .env and set:
DEBUG=true
LOG_LEVEL=DEBUG
```

**Option 2: Set environment variables when starting the server:**

```bash
# Windows PowerShell
$env:DEBUG="true"; $env:LOG_LEVEL="DEBUG"; uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Linux/Mac
DEBUG=true LOG_LEVEL=DEBUG uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Note:** The debug endpoint (`/api/health/debug`) will return a helpful error message if DEBUG is not enabled, instead of just "Not Found".

### Request Logging

When `DEBUG=true`, the backend will log:
- All incoming requests (method, path, query params)
- Request headers (with sensitive data masked)
- Response status and processing time

### Debug Endpoint

Access debug information at `/api/health/debug` (only available when `DEBUG=true`):

```bash
curl http://localhost:8000/api/health/debug
```

Returns:
- Python version
- Platform information
- CORS origins
- Database URL (masked)
- OpenAI model configuration
- Debug mode status

### Log Levels

Available log levels (set via `LOG_LEVEL` environment variable):
- `DEBUG` - Most verbose, includes all debug information
- `INFO` - Standard logging (default)
- `WARNING` - Only warnings and errors
- `ERROR` - Only errors

## Common Debugging Scenarios

### 1. Frontend Can't Connect to Backend

**Symptoms:**
- "Failed to fetch" errors
- Network errors in browser console

**Debug Steps:**
1. Check if backend is running:
   ```bash
   curl http://localhost:8000/api/health
   ```

2. Test connection from browser console:
   ```javascript
   window.__API_DEBUG__.testConnection()
   ```

3. Check CORS configuration:
   - Verify backend CORS origins include frontend URL
   - Check browser console for CORS errors

4. Verify backend host binding:
   ```bash
   # Backend should be started with:
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

### 2. API Requests Failing

**Debug Steps:**
1. Enable API logging:
   ```javascript
   window.__API_DEBUG__.enableLogging()
   ```

2. Check browser Network tab:
   - Look for failed requests
   - Check request/response headers
   - Verify request payload

3. Check backend logs:
   - Look for request logs
   - Check for error messages
   - Verify endpoint is being called

### 3. Authentication Issues

**Debug Steps:**
1. Check token:
   ```javascript
   window.__API_DEBUG__.getToken()
   ```

2. Check localStorage:
   ```javascript
   localStorage.getItem('auth_token')
   ```

3. Test auth endpoint:
   ```bash
   curl -X POST http://localhost:8000/api/auth/token \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d "username=test&password=test"
   ```

### 4. Database Issues

**Debug Steps:**
1. Check database file exists:
   ```bash
   ls -la ux_path_a/backend/ux_path_a.db
   ```

2. Run migrations:
   ```bash
   cd ux_path_a/backend
   alembic upgrade head
   ```

3. Check database connection in debug endpoint

## Quick Debug Checklist

- [ ] Backend is running and accessible
- [ ] Frontend can reach backend (test with `window.__API_DEBUG__.testConnection()`)
- [ ] CORS is properly configured
- [ ] API logging is enabled (check browser console)
- [ ] Backend debug mode is enabled (check logs)
- [ ] Network tab shows requests/responses
- [ ] No console errors in browser
- [ ] Backend logs show incoming requests

## Tips

1. **Always enable logging first** - It provides the most information
2. **Check both frontend and backend logs** - Issues can be on either side
3. **Use the Network tab** - Browser DevTools Network tab shows actual HTTP traffic
4. **Test endpoints directly** - Use `curl` or Postman to test backend independently
5. **Check environment variables** - Many issues are configuration-related
