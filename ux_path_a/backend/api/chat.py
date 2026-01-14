"""
Chat API endpoints.

Handles chat messages, session management, and LLM orchestration.
"""

from fastapi import APIRouter, Depends, HTTPException
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
from sqlalchemy.orm import Session

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize orchestrator
orchestrator = ChatOrchestrator()


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
        user_id=current_user.user_id or 1,  # TODO: Get from token
        title=session.title or f"Chat {now.strftime('%Y-%m-%d %H:%M')}",
    )
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
    # TODO: Filter by user_id from token
    db_sessions = db.query(ChatSession).order_by(ChatSession.updated_at.desc()).all()
    
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
    message: ChatMessageRequest,
    current_user: TokenData = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Send a chat message and get LLM response."""
    # Get or create session
    session_id = message.session_id
    if not session_id:
        # Create new session
        session_data = await create_session(
            SessionCreate(),
            current_user,
            db,
        )
        session_id = session_data.id
    
    # Verify session exists in database
    db_session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if not db_session:
        raise HTTPException(status_code=404, detail="Session not found")
    
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
    user_db_msg = DBChatMessage(
        session_id=session_id,
        role="user",
        content=message.content,
    )
    db.add(user_db_msg)
    db.commit()
    
    # Get LLM response via orchestrator
    try:
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
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")


@router.get("/sessions/{session_id}/messages", response_model=List[Message])
async def get_messages(
    session_id: str,
    current_user: TokenData = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get all messages for a session."""
    db_session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if not db_session:
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
