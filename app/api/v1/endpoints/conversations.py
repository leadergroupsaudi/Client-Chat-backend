from fastapi import APIRouter, Depends, Header, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel
import datetime

from app.core.dependencies import get_db, get_current_active_user, get_current_company, require_permission
from app.services import chat_service, agent_service, conversation_session_service, summary_service
from app.schemas import chat_message as schemas_chat_message, session as schemas_session, conversation_session as schemas_conversation_session
from app.models import user as models_user, conversation_session as models_conversation_session, chat_message as models_chat_message

router = APIRouter()

class NoteCreate(BaseModel):
    message: str

class StatusUpdate(BaseModel):
    status: str

class AssigneeUpdate(BaseModel):
    user_id: int

class FeedbackUpdate(BaseModel):
    feedback_rating: int
    feedback_notes: Optional[str] = None

class PriorityUpdate(BaseModel):
    priority: int  # 0=None, 1=Low, 2=Medium, 3=High, 4=Urgent

@router.get("/sessions/counts", dependencies=[Depends(require_permission("conversation:read"))])
def get_session_counts(db: Session = Depends(get_db), current_user: models_user.User = Depends(get_current_active_user)):
    """
    Get counts of sessions by status for the company.
    Returns: { "open": count, "resolved": count, "all": count }
    """
    all_sessions = chat_service.get_sessions_with_details(db, company_id=current_user.company_id)

    open_count = sum(1 for s in all_sessions if s.status not in ['resolved', 'archived'])
    resolved_count = sum(1 for s in all_sessions if s.status in ['resolved', 'archived'])

    return {
        "open": open_count,
        "resolved": resolved_count,
        "all": len(all_sessions)
    }

@router.get("/analytics/reopens", dependencies=[Depends(require_permission("conversation:read"))])
async def get_reopen_analytics(
    db: Session = Depends(get_db),
    current_user: models_user.User = Depends(get_current_active_user),
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """
    Get conversation reopen analytics including counts, rates, and average time to reopen.
    Query Parameters:
    - start_date: Optional date filter in YYYY-MM-DD format
    - end_date: Optional date filter in YYYY-MM-DD format
    """
    query = db.query(models_conversation_session.ConversationSession).filter(
        models_conversation_session.ConversationSession.company_id == current_user.company_id,
        models_conversation_session.ConversationSession.reopen_count > 0
    )

    if start_date:
        start_datetime = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        query = query.filter(models_conversation_session.ConversationSession.last_reopened_at >= start_datetime)
    if end_date:
        end_datetime = datetime.datetime.strptime(end_date, '%Y-%m-%d').replace(hour=23, minute=59, second=59)
        query = query.filter(models_conversation_session.ConversationSession.last_reopened_at <= end_datetime)

    sessions = query.all()

    total_reopens = sum(s.reopen_count for s in sessions)
    avg_reopen_count = total_reopens / len(sessions) if sessions else 0

    # Calculate average time to reopen
    times_to_reopen = []
    for s in sessions:
        if s.resolved_at and s.last_reopened_at:
            delta = (s.last_reopened_at - s.resolved_at).total_seconds()
            times_to_reopen.append(delta)

    avg_time_to_reopen = sum(times_to_reopen) / len(times_to_reopen) if times_to_reopen else 0

    # Get total conversations for reopen rate
    total_query = db.query(models_conversation_session.ConversationSession).filter(
        models_conversation_session.ConversationSession.company_id == current_user.company_id
    )
    if start_date:
        total_query = total_query.filter(models_conversation_session.ConversationSession.created_at >= start_datetime)
    if end_date:
        total_query = total_query.filter(models_conversation_session.ConversationSession.created_at <= end_datetime)

    total_conversations = total_query.count()
    reopen_rate = (len(sessions) / total_conversations * 100) if total_conversations > 0 else 0

    return {
        "total_conversations_reopened": len(sessions),
        "total_reopen_events": total_reopens,
        "average_reopens_per_conversation": round(avg_reopen_count, 2),
        "average_time_to_reopen_hours": round(avg_time_to_reopen / 3600, 2) if avg_time_to_reopen > 0 else 0,
        "reopen_rate_percentage": round(reopen_rate, 2),
    }


@router.get("/sessions", response_model=List[schemas_session.Session], dependencies=[Depends(require_permission("conversation:read"))])
def get_all_sessions(
    status_filter: Optional[str] = None,
    workflow_id: Optional[int] = None,
    db: Session = Depends(get_db),
    current_user: models_user.User = Depends(get_current_active_user)
):
    """
    Get sessions for the company, enriched with contact and channel info.

    Query Parameters:
    - status_filter: Filter by status category ('open', 'resolved', or None for all)
      - 'open' returns: active, inactive, assigned, pending
      - 'resolved' returns: resolved, archived
      - None returns: all sessions
    - workflow_id: Filter sessions by the workflow they are currently running
    """
    from app.services.connection_manager import manager

    sessions_from_db = chat_service.get_sessions_with_details(db, company_id=current_user.company_id, workflow_id=workflow_id)

    # Apply status filter
    if status_filter == 'open':
        sessions_from_db = [s for s in sessions_from_db if s.status not in ['resolved', 'archived']]
    elif status_filter == 'resolved':
        sessions_from_db = [s for s in sessions_from_db if s.status in ['resolved', 'archived']]

    sessions = []
    for s in sessions_from_db:
        first_message = chat_service.get_first_message_for_session(db, s.conversation_id, current_user.company_id)
        contact_info = chat_service.get_contact_for_session(db, s.conversation_id, current_user.company_id)

        # Get real-time connection status from ConnectionManager
        real_time_connected = manager.get_connection_status(s.conversation_id)

        # If real-time status differs from DB, update DB
        if real_time_connected != s.is_client_connected:
            print(f"[get_all_sessions] Syncing connection status for {s.conversation_id}: DB={s.is_client_connected} -> Real-time={real_time_connected}")
            s.is_client_connected = real_time_connected
            db.commit()

        sessions.append(schemas_session.Session(
            session_id=s.conversation_id,
            status=s.status,
            assignee_id=s.assignee_id,
            last_message_timestamp=s.updated_at.isoformat(),
            first_message_content=first_message.message if first_message else "",
            channel=s.channel,
            contact_name=contact_info.name if contact_info else "Unknown",
            contact_phone=contact_info.phone_number if contact_info else None,
            is_client_connected=real_time_connected,
            priority=s.priority,
            workflow_id=s.workflow_id,
        ))
    return sessions


# Summary endpoints - must be defined BEFORE /{agent_id}/{session_id} routes to avoid route conflicts
@router.get("/{session_id}/summary", response_model=schemas_conversation_session.ConversationSummaryResponse, dependencies=[Depends(require_permission("conversation:read"))])
def get_conversation_summary(
    session_id: str,
    db: Session = Depends(get_db),
    current_user: models_user.User = Depends(get_current_active_user)
):
    """
    Get the existing summary for a conversation session.
    Returns summary text, generation timestamp, and whether a summary exists.
    """
    result = summary_service.get_session_summary(
        db=db,
        session_id=session_id,
        company_id=current_user.company_id
    )
    return schemas_conversation_session.ConversationSummaryResponse(
        summary=result["summary"],
        generated_at=result["generated_at"],
        exists=result["exists"]
    )


@router.post("/{session_id}/summary", response_model=schemas_conversation_session.ConversationSummaryResponse, dependencies=[Depends(require_permission("conversation:read"))])
async def generate_conversation_summary_endpoint(
    session_id: str,
    db: Session = Depends(get_db),
    current_user: models_user.User = Depends(get_current_active_user)
):
    """
    Generate a new AI summary for a conversation session.
    Uses the session's agent LLM or company's first agent as fallback.
    """
    try:
        summary = await summary_service.generate_conversation_summary(
            db=db,
            session_id=session_id,
            company_id=current_user.company_id
        )
        return schemas_conversation_session.ConversationSummaryResponse(
            summary=summary,
            generated_at=datetime.datetime.utcnow(),
            exists=True
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate summary: {str(e)}")


@router.get("/{agent_id}/sessions", response_model=List[schemas_session.Session], deprecated=True, dependencies=[Depends(require_permission("conversation:read"))])
def get_sessions_by_agent(agent_id: int, db: Session = Depends(get_db), current_user: models_user.User = Depends(get_current_active_user)):
    # This endpoint is deprecated in favor of the company-wide /sessions endpoint
    sessions_from_db = chat_service.get_sessions_with_details(db, agent_id=agent_id, company_id=current_user.company_id)
    
    sessions = []
    for s in sessions_from_db:
        first_message = chat_service.get_first_message_for_session(db, s.conversation_id, current_user.company_id)
        sessions.append(schemas_session.Session(
            session_id=s.conversation_id,
            status=s.status,
            assignee_id=s.agent_id,
            last_message_timestamp=s.updated_at.isoformat(),
            first_message_content=first_message.message if first_message else ""
        ))
    return sessions


@router.get("/{agent_id}/sessions/{session_id:path}", response_model=schemas_session.Session, dependencies=[Depends(require_permission("conversation:read"))])
def get_session_detial_by_agent_id_session_id(agent_id: int, session_id: str, db: Session = Depends(get_db), current_user: models_user.User = Depends(get_current_active_user)):
    # This endpoint is deprecated in favor of the company-wide /sessions endpoint
    sessions_from_db = chat_service.get_session_details(db, agent_id=agent_id, broadcast_session_id=session_id, company_id=current_user.company_id)

    if not sessions_from_db:
        raise HTTPException(status_code=404, detail="Session not found")

    # Include contact information if available
    contact_info = None
    if sessions_from_db.contact:
        contact_info = schemas_session.ContactInfo(
            id=sessions_from_db.contact.id,
            name=sessions_from_db.contact.name,
            email=sessions_from_db.contact.email,
            phone_number=sessions_from_db.contact.phone_number
        )

    return schemas_session.Session(
            session_id=sessions_from_db.conversation_id,
            status=sessions_from_db.status,
            assignee_id=sessions_from_db.assignee_id,  # Use assignee_id instead of agent_id
            last_message_timestamp=sessions_from_db.updated_at.isoformat(),
            first_message_content="",
            contact=contact_info,
            is_ai_enabled=sessions_from_db.is_ai_enabled,
            priority=sessions_from_db.priority
        )

@router.get("/{agent_id}/{session_id}", response_model=List[schemas_chat_message.ChatMessage], dependencies=[Depends(require_permission("conversation:read"))])
def get_messages(
    agent_id: int,
    session_id: str,
    limit: int = 20,
    before_id: int = None,
    db: Session = Depends(get_db),
    current_user: models_user.User = Depends(get_current_active_user)
):
    """
    Get messages for a conversation session with pagination support.

    Args:
        agent_id: The agent ID
        session_id: The conversation session ID
        limit: Maximum number of messages to return (default: 20, max: 100)
        before_id: Return messages before this message ID (for loading older messages)

    Returns:
        List of messages in chronological order (oldest to newest)
    """
    # Cap the limit to prevent excessive data retrieval
    limit = min(limit, 100)

    return chat_service.get_chat_messages(
        db,
        agent_id=agent_id,
        session_id=session_id,
        company_id=current_user.company_id,
        limit=limit,
        before_id=before_id
    )

@router.post("/{agent_id}/{session_id}/notes", response_model=schemas_chat_message.ChatMessage, dependencies=[Depends(require_permission("conversation:update"))])
def create_note(
    agent_id: int,
    session_id: str,
    note: NoteCreate,
    db: Session = Depends(get_db),
    current_user: models_user.User = Depends(get_current_active_user)
):
    # We can consider adding author_id to notes in the future
    # For now, sender will be 'agent'
    message_schema = schemas_chat_message.ChatMessageCreate(message=note.message, message_type="note")
    return chat_service.create_chat_message(
        db=db,
        message=message_schema,
        agent_id=agent_id,
        session_id=session_id,
        company_id=current_user.company_id,
        sender="agent" 
    )

@router.put("/{session_id}/status", dependencies=[Depends(require_permission("conversation:update"))])
async def update_status(
    session_id: str,
    status_update: StatusUpdate,
    db: Session = Depends(get_db),
    current_user: models_user.User = Depends(get_current_active_user)
):
    success = await chat_service.update_conversation_status(db, session_id=session_id, status=status_update.status, company_id=current_user.company_id)
    if not success:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"message": "Status updated successfully"}

@router.put("/{session_id}/assignee", dependencies=[Depends(require_permission("conversation:update"))])
async def update_assignee(
    session_id: str,
    assignee_update: AssigneeUpdate,
    db: Session = Depends(get_db),
    current_user: models_user.User = Depends(get_current_active_user)
):
    success = await chat_service.update_conversation_assignee(db, session_id=session_id, user_id=assignee_update.user_id, company_id=current_user.company_id)
    if not success:
        raise HTTPException(status_code=404, detail="Conversation not found or assignee update failed")
    return {"message": "Assignee updated successfully"}

class AIToggleUpdate(BaseModel):
    is_ai_enabled: bool

@router.put("/{session_id}/toggle-ai", response_model=schemas_conversation_session.ConversationSession, dependencies=[Depends(require_permission("conversation:update"))])
def toggle_ai(
    session_id: str,
    update: AIToggleUpdate,
    db: Session = Depends(get_db),
    current_user: models_user.User = Depends(get_current_active_user)
):
    """
    Toggles the AI automated response for a specific conversation session.
    """
    session = conversation_session_service.toggle_ai_for_session(
        db, 
        conversation_id=session_id, 
        company_id=current_user.company_id, 
        is_enabled=update.is_ai_enabled
    )
    if not session:
        raise HTTPException(status_code=404, detail="Conversation session not found")
    return session


@router.put("/{session_id}/feedback", dependencies=[Depends(require_permission("conversation:update"))])
def update_feedback(
    session_id: str,
    feedback_update: FeedbackUpdate,
    db: Session = Depends(get_db),
    current_user: models_user.User = Depends(get_current_active_user)
):
    success = chat_service.update_conversation_feedback(
        db=db,
        session_id=session_id,
        feedback_rating=feedback_update.feedback_rating,
        feedback_notes=feedback_update.feedback_notes,
        company_id=current_user.company_id
    )
    if not success:
        raise HTTPException(status_code=404, detail="Conversation not found or feedback update failed")
    return {"message": "Feedback submitted successfully"}

@router.put("/{session_id}/priority", dependencies=[Depends(require_permission("conversation:update"))])
def update_priority(
    session_id: str,
    priority_update: PriorityUpdate,
    db: Session = Depends(get_db),
    current_user: models_user.User = Depends(get_current_active_user)
):
    """
    Update the priority level of a conversation.
    Priority levels: 0=None, 1=Low, 2=Medium, 3=High, 4=Urgent
    """
    if priority_update.priority < 0 or priority_update.priority > 4:
        raise HTTPException(status_code=400, detail="Priority must be between 0 and 4")

    session = conversation_session_service.update_session(
        db,
        conversation_id=session_id,
        session_update=schemas_conversation_session.ConversationSessionUpdate(priority=priority_update.priority)
    )
    if not session:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"message": "Priority updated successfully", "priority": session.priority}


# Public endpoint for widget (no authentication required)
@router.post("/widget/{session_id}/reset-workflow")
def reset_widget_workflow(session_id: str, db: Session = Depends(get_db)):
    """
    Public endpoint for widget to reset workflow state.
    Clears workflow_id, next_step_id, context, and subworkflow_stack.
    This allows the user to start fresh with a new conversation flow.
    """
    session = db.query(models_conversation_session.ConversationSession).filter(
        models_conversation_session.ConversationSession.conversation_id == session_id
    ).first()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Clear workflow state
    session.workflow_id = None
    session.next_step_id = None
    session.context = {}
    session.subworkflow_stack = None

    # Add reset message to database so it persists after refresh
    reset_message = models_chat_message.ChatMessage(
        session_id=session.id,
        agent_id=session.agent_id,
        company_id=session.company_id,
        sender="agent",
        message="Conversation reset. How can I help you?",
        message_type="message",
        timestamp=datetime.datetime.utcnow()
    )
    db.add(reset_message)
    db.commit()

    return {"success": True, "message": "Workflow reset successfully"}