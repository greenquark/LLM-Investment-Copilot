"""
Database configuration and session management.

Uses SQLAlchemy for ORM and Alembic for migrations.
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from typing import Generator

# Use absolute imports (works in both local and Railway with PYTHONPATH=/app)
from ux_path_a.backend.backend_core.config import settings

# Build connect_args with sane timeouts so Railway can still serve /api/health even if DB is down.
db_url = settings.DATABASE_URL
db_url_lower = db_url.lower() if isinstance(db_url, str) else str(db_url).lower()

connect_args = {}
if "sqlite" in db_url_lower:
    connect_args = {"check_same_thread": False}
elif db_url_lower.startswith("postgres"):
    # psycopg2 connect timeout is in seconds.
    # Prevent long hangs during container startup/migrations when DB is unavailable.
    connect_timeout = int(os.getenv("DB_CONNECT_TIMEOUT", "5"))
    connect_args = {"connect_timeout": connect_timeout}

# Create database engine
engine = create_engine(
    db_url,
    connect_args=connect_args,
    echo=settings.DEBUG,
    pool_pre_ping=True,
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """
    Dependency for getting database session.
    
    Usage:
        @app.get("/items")
        def get_items(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
