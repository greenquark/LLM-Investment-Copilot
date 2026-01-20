"""
Chat API endpoints.

Handles chat messages, session management, and LLM orchestration.
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timezone
import uuid
import logging

# Use absolute imports (works in both local and Railway with PYTHONPATH=/app)
from ux_path_a.backend.api.auth import get_current_user, TokenData
from ux_path_a.backend.backend_core.orchestrator import ChatOrchestrator
from ux_path_a.backend.backend_core.database import get_db
from ux_path_a.backend.backend_core.guardrails import TokenBudgetTracker, SafetyControls
from ux_path_a.backend.backend_core.models import ChatSession, ChatMessage as DBChatMessage, AuditLog
from ux_path_a.backend.backend_core.config import settings
from sqlalchemy.orm import Session

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize orchestrator
orchestrator = ChatOrchestrator()

def _require_user_id(current_user: TokenData) -> int:
    """
    Ensure we have a concrete user_id from the auth token.

    We treat missing user_id as an auth failure (should never happen in normal flows).
    """
    if current_user.user_id is None:
        raise HTTPException(status_code=401, detail="Invalid auth token: missing user_id")
    return int(current_user.user_id)


# Pydantic models
class Message(BaseModel):
    """Chat message model."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime
    thinking_content: Optional[str] = None  # Reasoning/thinking process from models like o1
    tool_calls: Optional[List[dict]] = None
    tool_results: Optional[List[dict]] = None


class ChatMessageRequest(BaseModel):
    """Incoming chat message request."""
    content: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    """Chat response."""
    message: Message
    session_id: str
    token_usage: Optional[dict] = None


class SessionCreate(BaseModel):
    """Create new session."""
    title: Optional[str] = None


class SessionResponse(BaseModel):
    """Session response."""
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int


# Database models are used instead of in-memory dict


@router.post("/sessions", response_model=SessionResponse)
async def create_session(
    session: SessionCreate,
    current_user: TokenData = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Create a new chat session."""
    session_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    
    # Create session in database
    db_session = ChatSession(
        id=session_id,
        user_id=current_user.user_id if current_user.user_id is not None else None,
        title=session.title or f"Chat {now.strftime('%Y-%m-%d %H:%M')}",
    )
    if db_session.user_id is None:
        raise HTTPException(status_code=401, detail="Invalid auth token: missing user_id")
    db.add(db_session)
    db.commit()
    db.refresh(db_session)
    
    return SessionResponse(
        id=session_id,
        title=db_session.title,
        created_at=db_session.created_at,
        updated_at=db_session.updated_at,
        message_count=0,
    )


@router.get("/sessions", response_model=List[SessionResponse])
async def list_sessions(
    current_user: TokenData = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List all sessions for current user."""
    user_id = _require_user_id(current_user)

    db_sessions = (
        db.query(ChatSession)
        .filter(ChatSession.user_id == user_id)
        .order_by(ChatSession.updated_at.desc())
        .all()
    )
    
    sessions = []
    for db_session in db_sessions:
        message_count = db.query(DBChatMessage).filter(
            DBChatMessage.session_id == db_session.id
        ).count()
        
        sessions.append(SessionResponse(
            id=db_session.id,
            title=db_session.title,
            created_at=db_session.created_at,
            updated_at=db_session.updated_at,
            message_count=message_count,
        ))
    return sessions


@router.post("/messages", response_model=ChatResponse)
async def send_message(
    request: Request,
    message: ChatMessageRequest,
    current_user: TokenData = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Send a chat message and get LLM response."""
    railway_request_id = request.headers.get("x-railway-request-id") or request.headers.get("x-request-id")
    user_id = _require_user_id(current_user)

    # Get or create session
    session_id = message.session_id
    if not session_id:
        # Fail fast with a clear error if LLM is not configured.
        # Avoid creating empty sessions when the LLM is unavailable.
        if not settings.OPENAI_API_KEY:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "LLM not configured",
                    "message": "Missing OPENAI_API_KEY on server. Set it in Railway Variables and redeploy.",
                    "request_id": railway_request_id,
                },
            )
        # Create new session
        session_data = await create_session(
            SessionCreate(),
            current_user,
            db,
        )
        session_id = session_data.id
    
    # Verify session exists in database
    db_session = (
        db.query(ChatSession)
        .filter(ChatSession.id == session_id, ChatSession.user_id == user_id)
        .first()
    )
    if not db_session:
        # 404 (not 403) to avoid leaking existence of other users' sessions.
        raise HTTPException(status_code=404, detail="Session not found")

    # Fail fast with a clear error if LLM is not configured.
    # Otherwise the OpenAI SDK raises deep inside the orchestrator and becomes a generic 500.
    if not settings.OPENAI_API_KEY:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "LLM not configured",
                "message": "Missing OPENAI_API_KEY on server. Set it in Railway Variables and redeploy.",
                "request_id": railway_request_id,
            },
        )
    
    # Load conversation history from database
    db_messages = db.query(DBChatMessage).filter(
        DBChatMessage.session_id == session_id
    ).order_by(DBChatMessage.created_at).all()
    
    conversation_history = [
        {
            "role": msg.role,
            "content": msg.content,
            "timestamp": msg.created_at.isoformat(),
            "thinking_content": msg.thinking_content,
            "tool_calls": msg.tool_calls,
            "tool_results": msg.tool_results,
        }
        for msg in db_messages
    ]
    
    # Add user message to database
    try:
        user_db_msg = DBChatMessage(
            session_id=session_id,
            role="user",
            content=message.content,
        )
        db.add(user_db_msg)
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error("DB write failed rid=%s: %s", railway_request_id, e, exc_info=True)
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Database error",
                "message": str(e) or "Failed to write chat message",
                "request_id": railway_request_id,
            },
        )
    
    # Get LLM response via orchestrator
    try:
        logger.info("Chat orchestration start rid=%s session_id=%s", railway_request_id, session_id)
        response = await orchestrator.process_message(
            message=message.content,
            session_id=session_id,
            conversation_history=conversation_history,
        )
        
        # Serialize tool_calls and tool_results for database storage
        import json
        
        def make_json_serializable(obj):
            """Recursively convert objects to JSON-serializable format."""
            # Check if it's a Function-like object first (has name and arguments attributes)
            # This check must come before checking __dict__ because Function might not expose __dict__
            try:
                if hasattr(obj, 'name') and hasattr(obj, 'arguments'):
                    # This looks like a Function object from OpenAI SDK
                    func_dict = {
                        'name': getattr(obj, 'name', None),
                        'arguments': getattr(obj, 'arguments', None),
                    }
                    # Add optional attributes
                    if hasattr(obj, 'id'):
                        func_dict['id'] = getattr(obj, 'id', None)
                    if hasattr(obj, 'type'):
                        func_dict['type'] = getattr(obj, 'type', None)
                    return func_dict
            except:
                pass
            
            if isinstance(obj, (str, int, float, bool, type(None))):
                return obj
            elif isinstance(obj, dict):
                return {k: make_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_json_serializable(item) for item in obj]
            elif hasattr(obj, '__dict__'):
                # Convert object to dict, but handle special cases
                obj_dict = {}
                for k, v in obj.__dict__.items():
                    # Skip private attributes
                    if not k.startswith('_'):
                        obj_dict[k] = make_json_serializable(v)
                # Also check for attributes that might not be in __dict__ (like properties)
                for attr in ['name', 'arguments', 'id', 'type', 'function']:
                    if hasattr(obj, attr) and attr not in obj_dict:
                        try:
                            attr_value = getattr(obj, attr)
                            obj_dict[attr] = make_json_serializable(attr_value)
                        except:
                            pass
                return obj_dict
            elif callable(obj) and not isinstance(obj, type):
                # Convert callable objects (functions, methods) to string
                return f"<function: {getattr(obj, '__name__', str(obj))}>"
            else:
                # Convert any other type to string
                return str(obj)
        
        tool_calls_serialized = None
        if response.get("tool_calls"):
            try:
                tool_calls_serialized = make_json_serializable(response["tool_calls"])
                # Test serialization
                json.dumps(tool_calls_serialized)
            except Exception as e:
                logger.warning(f"Could not serialize tool_calls: {e}", exc_info=True)
                # Fallback: convert everything to string
                try:
                    tool_calls_serialized = [str(tc) for tc in response["tool_calls"]]
                except:
                    tool_calls_serialized = None
        
        # Serialize tool_results - ensure all objects are JSON serializable
        tool_results_serialized = None
        if response.get("tool_results"):
            try:
                tool_results_serialized = make_json_serializable(response["tool_results"])
                # Test serialization
                json.dumps(tool_results_serialized)
            except Exception as e:
                logger.warning(f"Could not serialize tool_results: {e}", exc_info=True)
                # Fallback: convert everything to string
                try:
                    tool_results_serialized = [str(tr) for tr in response["tool_results"]]
                except:
                    tool_results_serialized = None
        
        # Add assistant message to database
        assistant_db_msg = DBChatMessage(
            session_id=session_id,
            role="assistant",
            content=response["content"],
            thinking_content=response.get("thinking_content"),
            tool_calls=tool_calls_serialized,
            tool_results=tool_results_serialized,
            token_usage=response.get("token_usage"),
        )
        db.add(assistant_db_msg)
        
        # Update session timestamp
        from datetime import timezone
        db_session.updated_at = datetime.now(timezone.utc)
        if response.get("token_usage"):
            db_session.total_tokens_used = (db_session.total_tokens_used or 0) + response["token_usage"].get("total_tokens", 0)
        
        db.commit()
        db.refresh(assistant_db_msg)
        
        # Create response message
        assistant_msg = Message(
            role="assistant",
            content=response["content"],
            timestamp=datetime.now(timezone.utc),
            thinking_content=response.get("thinking_content"),
            tool_calls=response.get("tool_calls"),
            tool_results=response.get("tool_results"),
        )
        
        return ChatResponse(
            message=assistant_msg,
            session_id=session_id,
            token_usage=response.get("token_usage"),
        )
    except Exception as e:
        db.rollback()
        # Categorize common upstream (OpenAI) errors so the frontend can show actionable messages.
        error_cls = e.__class__.__name__
        error_msg = str(e) or "Unknown error"
        logger.error("Error processing chat message rid=%s (%s): %s", railway_request_id, error_cls, e, exc_info=True)

        # Best-effort classification without relying on specific OpenAI SDK exception classes.
        # (OpenAI SDK exception names vary by version.)
        lowered = f"{error_cls} {error_msg}".lower()
        if "authentication" in lowered or "invalid api key" in lowered or "api key" in lowered and "invalid" in lowered:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": "OpenAI authentication failed",
                    "message": error_msg,
                    "request_id": railway_request_id,
                },
            )
        if "rate limit" in lowered or "429" in lowered:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "OpenAI rate limited",
                    "message": error_msg,
                    "request_id": railway_request_id,
                },
            )
        if "model" in lowered and ("not found" in lowered or "does not exist" in lowered):
            raise HTTPException(
                status_code=502,
                detail={
                    "error": "OpenAI model unavailable",
                    "message": error_msg,
                    "request_id": railway_request_id,
                },
            )

        raise HTTPException(
            status_code=500,
            detail={
                "error": "Error processing message",
                "message": error_msg,
                "exception": error_cls,
                "request_id": railway_request_id,
            },
        )


@router.get("/sessions/{session_id}/messages", response_model=List[Message])
async def get_messages(
    session_id: str,
    current_user: TokenData = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get all messages for a session."""
    user_id = _require_user_id(current_user)

    db_session = (
        db.query(ChatSession)
        .filter(ChatSession.id == session_id, ChatSession.user_id == user_id)
        .first()
    )
    if not db_session:
        # 404 (not 403) to avoid leaking existence of other users' sessions.
        raise HTTPException(status_code=404, detail="Session not found")
    
    db_messages = db.query(DBChatMessage).filter(
        DBChatMessage.session_id == session_id
    ).order_by(DBChatMessage.created_at).all()
    
    return [
        Message(
            role=msg.role,
            content=msg.content,
            timestamp=msg.created_at,
            thinking_content=msg.thinking_content,
            tool_calls=msg.tool_calls,
            tool_results=msg.tool_results,
        )
        for msg in db_messages
    ]
