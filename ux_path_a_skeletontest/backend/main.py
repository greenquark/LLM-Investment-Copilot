import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.health import router as health_router

log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ux_path_a_skeletontest.backend")

app = FastAPI(title="UX Path A Skeleton Test Backend", version="0.0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router, prefix="/api/health", tags=["health"])


@app.get("/")
async def root():
    return {"ok": True, "service": "ux_path_a_skeletontest-backend"}


@app.on_event("startup")
async def _startup():
    logger.info("Skeleton backend starting (PORT=%s)", os.getenv("PORT"))

