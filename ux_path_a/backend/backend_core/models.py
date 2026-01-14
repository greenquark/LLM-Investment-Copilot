"""
Database models for UX Path A.

Defines SQLAlchemy models for users, sessions, messages, and audit logs.
"""

from sqlalchemy import Column, Integer, String, DateTime, Text, JSON, ForeignKey, Float, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime

# Use absolute imports (works in both local and Railway with PYTHONPATH=/app)
from ux_path_a.backend.backend_core.database import Base

# Get the actual module name to use in relationships
_MODULE_NAME = __name__


class User(Base):
    """User model for authentication."""
    __tablename__ = "users"
    __table_args__ = {'extend_existing': True}
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    is_active = Column(Boolean, default=True)
    
    # Relationships - will be set after all classes are defined
    sessions = None  # type: ignore


class ChatSession(Base):
    """Chat session model."""
    __tablename__ = "chat_sessions"
    __table_args__ = {'extend_existing': True}
    
    id = Column(String, primary_key=True, index=True)  # UUID string
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    title = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Token usage tracking (INV-LLM-03)
    total_tokens_used = Column(Integer, default=0)
    
    # Relationships - will be set after all classes are defined
    user = None  # type: ignore
    messages = None  # type: ignore
    audit_logs = None  # type: ignore


class ChatMessage(Base):
    """Chat message model."""
    __tablename__ = "chat_messages"
    __table_args__ = {'extend_existing': True}
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, ForeignKey("chat_sessions.id"), nullable=False, index=True)
    role = Column(String, nullable=False)  # "user" or "assistant"
    content = Column(Text, nullable=False)
    thinking_content = Column(Text, nullable=True)  # Reasoning/thinking process from models like o1
    tool_calls = Column(JSON, nullable=True)  # Store tool calls as JSON
    tool_results = Column(JSON, nullable=True)  # Store tool results as JSON
    token_usage = Column(JSON, nullable=True)  # Store token usage stats
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships - will be set after all classes are defined
    session = None  # type: ignore


class AuditLog(Base):
    """
    Audit log for all interactions (INV-AUDIT-01, INV-AUDIT-02).
    
    Logs every interaction for reproducibility and compliance.
    """
    __tablename__ = "audit_logs"
    __table_args__ = {'extend_existing': True}
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, ForeignKey("chat_sessions.id"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    
    # Prompt and response tracking
    prompt_version = Column(String, nullable=True)  # System prompt version
    user_message = Column(Text, nullable=True)
    assistant_response = Column(Text, nullable=True)
    
    # Tool tracking
    tools_called = Column(JSON, nullable=True)  # List of tool names called
    tool_inputs = Column(JSON, nullable=True)  # Tool inputs
    tool_outputs = Column(JSON, nullable=True)  # Tool outputs (may be truncated)
    
    # Token usage
    prompt_tokens = Column(Integer, nullable=True)
    completion_tokens = Column(Integer, nullable=True)
    total_tokens = Column(Integer, nullable=True)
    
    # Strategy versions (if applicable)
    strategy_versions = Column(JSON, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships - will be set after all classes are defined
    session = None  # type: ignore
    user = None  # type: ignore


class TokenBudget(Base):
    """Token budget tracking per session (INV-LLM-03)."""
    __tablename__ = "token_budgets"
    __table_args__ = {'extend_existing': True}
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, ForeignKey("chat_sessions.id"), nullable=False, unique=True, index=True)
    budget_limit = Column(Integer, nullable=False)  # Max tokens for session
    tokens_used = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


# Set up relationships after all classes are defined to avoid circular imports
# Using backref instead of back_populates to avoid string resolution issues
# Using direct column references (ChatMessage.created_at) instead of strings for order_by
# All imports use absolute paths (ux_path_a.backend.backend_core.*) to ensure single module registration
User.sessions = relationship(ChatSession, backref="user", cascade="all, delete-orphan")
ChatSession.messages = relationship(ChatMessage, backref="session", cascade="all, delete-orphan", order_by=ChatMessage.created_at)
ChatSession.audit_logs = relationship(AuditLog, backref="session", cascade="all, delete-orphan")
AuditLog.user = relationship(User, backref="audit_logs")
