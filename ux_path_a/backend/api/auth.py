"""
Authentication endpoints.

Implements JWT-based authentication for UX Path A.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr
import logging

# Use absolute imports (works in both local and Railway with PYTHONPATH=/app)
from ux_path_a.backend.backend_core.config import settings
from ux_path_a.backend.backend_core.database import get_db
from ux_path_a.backend.backend_core.models import User
from sqlalchemy.orm import Session

router = APIRouter()
logger = logging.getLogger(__name__)

# Password hashing
# NOTE: bcrypt is notoriously finicky on Windows and has a 72-byte password limit.
# For MVP (we don't verify passwords yet), use a stable built-in hash to keep local dev unblocked.
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token")


# Pydantic models
class UserCreate(BaseModel):
    """User registration model."""
    email: EmailStr
    password: str
    username: str


class UserResponse(BaseModel):
    """User response model."""
    id: int
    email: str
    username: str
    created_at: datetime


class Token(BaseModel):
    """Token response model."""
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    """Token data model."""
    user_id: Optional[int] = None
    username: Optional[str] = None


# TODO: Replace with actual database
# This is a placeholder for MVP
_users_db: dict = {}


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme)):
    """Get current authenticated user from token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        sub = payload.get("sub")
        if sub is None:
            raise credentials_exception
        try:
            user_id = int(sub)
        except Exception:
            raise credentials_exception
        username = payload.get("username")
        token_data = TokenData(user_id=user_id, username=username)
    except JWTError:
        raise credentials_exception
    
    # TODO: Get user from database
    # For now, return token data
    return token_data


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(user: UserCreate):
    """Register a new user."""
    # TODO: Check if user exists in database
    # TODO: Create user in database
    # For MVP, just return success
    return UserResponse(
        id=1,
        email=user.email,
        username=user.username,
        created_at=datetime.utcnow(),
    )


@router.post("/token", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    """Login and get access token."""
    # MVP behavior: accept any credentials, but back it with a real DB user row
    # so FK constraints (chat_sessions.user_id -> users.id) are satisfied.
    username = (form_data.username or "").strip()
    if not username:
        raise HTTPException(status_code=400, detail="Username is required")

    user = db.query(User).filter(User.username == username).first()
    if not user:
        # Create a deterministic placeholder email for MVP; can be replaced by real registration later.
        email = f"{username}@mvp.local"

        user = User(
            email=email,
            username=username,
            hashed_password=get_password_hash(form_data.password or "mvp"),
        )
        db.add(user)
        try:
            db.commit()
            db.refresh(user)
        except Exception as e:
            db.rollback()
            raise HTTPException(status_code=500, detail=f"Failed to create user: {e}")
        logger.info("Created MVP user username=%s id=%s", user.username, user.id)
    else:
        logger.info("Using existing user username=%s id=%s", user.username, user.id)
    
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": str(user.id), "username": user.username},
        expires_delta=access_token_expires,
    )
    
    return Token(access_token=access_token)


@router.get("/me", response_model=TokenData)
async def get_current_user_info(current_user: TokenData = Depends(get_current_user)):
    """Get current user information."""
    return current_user
