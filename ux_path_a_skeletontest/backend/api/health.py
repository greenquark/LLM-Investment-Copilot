from fastapi import APIRouter
from datetime import datetime, timezone

router = APIRouter()


@router.get("")
@router.get("/")
async def health():
    return {
        "status": "healthy",
        "service": "ux_path_a_skeletontest-backend",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/ready")
async def ready():
    # No external dependencies in skeleton test
    return {
        "status": "ready",
        "checks": {"runtime": "ok"},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

