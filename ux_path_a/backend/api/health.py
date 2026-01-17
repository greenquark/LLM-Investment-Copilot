"""
Health check endpoints.
"""

from fastapi import APIRouter
from datetime import datetime
# Use absolute imports (works in both local and Railway with PYTHONPATH=/app)
from ux_path_a.backend.backend_core.config import settings
import os

router = APIRouter()

def _get_build_info() -> dict:
    # Prefer Vercel vars if present (useful for parity), but support Railway/GitHub too.
    sha = (
        os.getenv("RAILWAY_GIT_COMMIT_SHA")
        or os.getenv("VERCEL_GIT_COMMIT_SHA")
        or os.getenv("GITHUB_SHA")
        or os.getenv("COMMIT_SHA")
        or ""
    )
    ref = (
        os.getenv("RAILWAY_GIT_BRANCH")
        or os.getenv("VERCEL_GIT_COMMIT_REF")
        or os.getenv("GITHUB_REF_NAME")
        or os.getenv("BRANCH")
        or ""
    )
    build_time = os.getenv("BUILD_TIME") or os.getenv("VERCEL_BUILD_TIME") or ""
    return {
        "commit_sha": sha,
        "commit_short": sha[:7] if sha else "",
        "branch": ref,
        "build_time": build_time,
    }


@router.get("")
@router.get("/")
async def health_check():
    """Basic health check."""
    build = _get_build_info()
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": os.getenv("APP_VERSION") or build.get("commit_short") or "0.1.0",
        "build": build,
    }


@router.get("/ready")
async def readiness_check():
    """Readiness check - verifies dependencies."""
    checks = {
        "database": "ok",  # TODO: Add actual DB check
        "openai": "ok" if settings.OPENAI_API_KEY else "missing",
    }
    
    all_ready = all(status == "ok" for status in checks.values())
    
    return {
        "status": "ready" if all_ready else "not_ready",
        "checks": checks,
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/config")
async def get_config():
    """Get application configuration (non-sensitive)."""
    return {
        "llm_model": settings.OPENAI_MODEL,
    }


@router.get("/debug", name="debug_info")
async def debug_info():
    """Debug information (only in development)."""
    from fastapi import HTTPException
    import sys
    import platform
    
    if not settings.DEBUG:
        raise HTTPException(
            status_code=403, 
            detail={
                "error": "Debug endpoint is disabled",
                "message": "Set DEBUG=true in your .env file or environment variables to enable this endpoint",
                "current_debug": settings.DEBUG,
                "current_log_level": settings.LOG_LEVEL
            }
        )
    
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "cors_origins": settings.CORS_ORIGINS,
        "database_url": settings.DATABASE_URL.split('@')[-1] if '@' in settings.DATABASE_URL else "local",
        "openai_model": settings.OPENAI_MODEL,
        "openai_key_set": bool(settings.OPENAI_API_KEY),
        "debug_mode": settings.DEBUG,
        "log_level": settings.LOG_LEVEL,
    }
