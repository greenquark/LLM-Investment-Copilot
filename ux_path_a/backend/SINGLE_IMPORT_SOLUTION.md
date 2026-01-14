# Single Import Path Solution

## Problem

The backend was experiencing `InvalidRequestError: Multiple classes found for path "ChatMessage"` errors due to duplicate SQLAlchemy model registrations. This occurred because models were being imported through different paths:
- Absolute: `ux_path_a.backend.backend_core.models`
- Relative: `backend_core.models`

Python treats these as different modules, causing SQLAlchemy to register the same classes twice.

## Root Cause

The try/except import pattern was attempting both absolute and relative imports:
```python
try:
    from ux_path_a.backend.backend_core.models import ChatSession  # Absolute
except ImportError:
    from backend_core.models import ChatSession  # Relative
```

Even if only one path succeeded, the mere attempt to import through different paths could cause issues, especially with Python's module caching.

## Solution: Single Absolute Import Path

Since the Dockerfile already sets `PYTHONPATH=/app` (project root), absolute imports work in both:
- **Local development**: Running from project root
- **Railway deployment**: `PYTHONPATH=/app` makes absolute imports work

### Changes Made

1. **Removed all try/except import blocks** for backend_core modules
2. **Standardized on absolute imports**: `from ux_path_a.backend.backend_core.*`
3. **Fixed order_by**: Changed from string `"ChatMessage.created_at"` to direct reference `ChatMessage.created_at`
4. **Updated relationships**: Using `backref` instead of `back_populates` (already done)

### Files Updated

All backend files now use single absolute imports:
- ✅ `main.py`
- ✅ `api/chat.py`
- ✅ `api/auth.py`
- ✅ `api/health.py`
- ✅ `backend_core/models.py`
- ✅ `backend_core/database.py`
- ✅ `backend_core/config.py`
- ✅ `backend_core/orchestrator.py`
- ✅ `backend_core/guardrails.py`
- ✅ `backend_core/prompts.py`
- ✅ `backend_core/tools/*.py`
- ✅ `alembic/env.py`
- ✅ `start.sh` (simplified to single import)

## Why This Works

1. **Single module registration**: SQLAlchemy sees only one module path
2. **No duplicate registrations**: Each class is registered exactly once
3. **Works everywhere**: `PYTHONPATH=/app` ensures absolute imports work in all environments
4. **Simpler code**: No try/except blocks needed

## Verification

After clearing Python's bytecode cache (`__pycache__`), the backend should:
- ✅ Start without errors
- ✅ Create chat sessions successfully
- ✅ Work in both local and Railway environments

## Next Steps

1. **Restart the backend server** (if running)
2. **Test creating a chat session** - should work without errors
3. **Deploy to Railway** - should work with the same code

## Benefits

- ✅ Eliminates duplicate registration errors
- ✅ Simpler, more maintainable code
- ✅ Consistent import style across all files
- ✅ Works in both local and production environments
- ✅ No need for complex try/except fallbacks
