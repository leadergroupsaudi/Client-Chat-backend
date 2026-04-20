from sqlalchemy.orm import Session
from sqlalchemy import func, desc, or_
from app.models import chat_message as models_chat_message, conversation_session as models_conversation_session, contact as models_contact
from app.schemas import chat_message as schemas_chat_message
from app.services import contact_service

def get_sessions_with_details(db: Session, company_id: int, agent_id: int = None, status: str = None, workflow_id: int = None):
    """
    Gets all unique sessions for a company, optionally filtered by agent, status, or workflow.
    It retrieves the latest message for ordering and context.
    """
    query = db.query(models_conversation_session.ConversationSession).filter(
        models_conversation_session.ConversationSession.company_id == company_id
    )

    if agent_id:
        query = query.filter(models_conversation_session.ConversationSession.agent_id == agent_id)

    if status:
        query = query.filter(models_conversation_session.ConversationSession.status == status)

    if workflow_id:
        query = query.filter(models_conversation_session.ConversationSession.workflow_id == workflow_id)

    # Order by the most recently updated session
    sessions = query.order_by(desc(models_conversation_session.ConversationSession.updated_at)).all()
    return sessions


def get_session_details(db: Session, company_id: int, agent_id: int = None, broadcast_session_id: str = None,  status: str = None):
    """
    Gets session details by conversation_id for a company.
    Note: agent_id is kept for backward compatibility but not used in filtering
    since conversations can be viewed across different agents.
    """
    session = db.query(models_conversation_session.ConversationSession).filter(
        models_conversation_session.ConversationSession.conversation_id == str(broadcast_session_id),
        models_conversation_session.ConversationSession.company_id == company_id
    ).first()

    return session


def get_contact_for_session(db: Session, session_id: str, company_id: int):
    """
    Retrieves the contact associated with a given session.
    """
    session = db.query(models_conversation_session.ConversationSession).filter(
        models_conversation_session.ConversationSession.conversation_id == session_id,
        models_conversation_session.ConversationSession.company_id == company_id
    ).first()
    
    if session:
        return session.contact
    return None


def get_first_message_for_session(db: Session, session_id: str, company_id: int):
    # First lookup the session by conversation_id to get the actual session.id (bigint)
    session = db.query(models_conversation_session.ConversationSession).filter(
        models_conversation_session.ConversationSession.conversation_id == session_id,
        models_conversation_session.ConversationSession.company_id == company_id
    ).first()
 
    if not session:
        return None
 
    return db.query(models_chat_message.ChatMessage).filter(
        models_chat_message.ChatMessage.session_id == session.id,
        models_chat_message.ChatMessage.company_id == company_id
    ).order_by(models_chat_message.ChatMessage.timestamp).first()

def create_chat_message(db: Session, message: schemas_chat_message.ChatMessageCreate, agent_id: int, session_id: str, company_id: int, sender: str, assignee_id: int = None, attachments: list = None, options: list = None):

    # Get the session to retrieve the contact_id (if available)
    session = db.query(models_conversation_session.ConversationSession).filter(
        models_conversation_session.ConversationSession.conversation_id == session_id,
        models_conversation_session.ConversationSession.company_id == company_id
    ).first()

    if not session:
        raise ValueError(f"Session {session_id} not found.")

    # issue = None
    # if sender == 'user':
    #     issue = tag_issue(message.message)

    # Clean attachments for storage (remove base64 file_data, keep only metadata)
    cleaned_attachments = None
    if attachments:
        cleaned_attachments = []
        for att in attachments:
            clean_att = {
                'file_name': att.get('file_name'),
                'file_type': att.get('file_type'),
                'file_size': att.get('file_size'),
            }
            # Include file_url if available (uploaded to S3)
            if att.get('file_url'):
                clean_att['file_url'] = att['file_url']
            # Include location data if present
            if att.get('location'):
                clean_att['location'] = att['location']
            cleaned_attachments.append(clean_att)

    # Contact_id can be NULL for anonymous conversations
    db_message = models_chat_message.ChatMessage(
        message=message.message,
        sender=sender,
        agent_id=agent_id,
        session_id=session.id,
        company_id=company_id,
        message_type=message.message_type,
        token=message.token, # Add the token here
        contact_id=session.contact_id,  # Can be NULL for anonymous chats
        assignee_id=assignee_id,  # Store the agent/user who sent this message
        attachments=cleaned_attachments,  # Store attachment metadata
        options=options,  # Store prompt options for message_type='prompt'
        # issue=issue
    )
    db.add(db_message)
    db.commit()
    db.refresh(db_message)
    return db_message

def get_chat_messages(db: Session, agent_id: int, session_id: str, company_id: int, skip: int = 0, limit: int = 50, before_id: int = None):
    """
    Get chat messages with pagination support.

    Args:
        db: Database session
        agent_id: Agent ID (kept for backward compatibility, not used in filtering)
        session_id: Conversation session ID
        company_id: Company ID
        skip: Number of messages to skip (used for offset-based pagination, deprecated)
        limit: Maximum number of messages to return
        before_id: Get messages before this message ID (cursor-based pagination)

    Returns:
        List of messages in chronological order (oldest to newest)

    Note:
        - When before_id is provided, it returns messages older than the specified message
        - When before_id is None, it returns the latest messages
        - Messages are always returned in chronological order (oldest first)
    """
    session = db.query(models_conversation_session.ConversationSession).filter(
        models_conversation_session.ConversationSession.conversation_id == session_id,
        models_conversation_session.ConversationSession.company_id == company_id
    ).first()

    if not session:
        return []

    # Base query for all messages in this session
    query = db.query(models_chat_message.ChatMessage).filter(
        models_chat_message.ChatMessage.session_id == session.id,
        models_chat_message.ChatMessage.company_id == company_id
    )

    if before_id is not None:
        # Cursor-based pagination: Get messages before the specified message ID
        # We order by ID descending to get older messages, then reverse
        query = query.filter(models_chat_message.ChatMessage.id < before_id)
        messages = query.order_by(desc(models_chat_message.ChatMessage.id)).limit(limit).all()
        # Reverse to return in chronological order (oldest to newest)
        return list(reversed(messages))
    else:
        # Get the latest messages (most recent)
        # Order by descending to get latest first, then reverse
        messages = query.order_by(desc(models_chat_message.ChatMessage.id)).limit(limit).all()
        # Reverse to return in chronological order (oldest to newest)
        return list(reversed(messages))

async def update_conversation_status(db: Session, session_id: str, status: str, company_id: int):
    """
    Updates the status of a conversation session (e.g., 'resolved', 'active', 'pending').
    """
    from app.services.connection_manager import manager
    import json

    # Get the session to retrieve current connection status
    session = db.query(models_conversation_session.ConversationSession).filter(
        models_conversation_session.ConversationSession.conversation_id == session_id,
        models_conversation_session.ConversationSession.company_id == company_id
    ).first()

    if not session:
        return False

    # Update status
    session.status = status

    # Track resolution timestamp
    if status == 'resolved':
        import datetime
        session.resolved_at = datetime.datetime.utcnow()

    db.commit()
    db.refresh(session)

    # Broadcast status change to all connected agents in real-time, including connection status and assignee
    status_update_message = json.dumps({
        "type": "session_status_update",
        "session_id": session_id,
        "status": status,
        "assignee_id": session.assignee_id,
        "is_client_connected": session.is_client_connected,
        "updated_at": session.updated_at.isoformat()
    })
    await manager.broadcast(status_update_message, str(company_id))

    return True

async def update_conversation_assignee(db: Session, session_id: str, user_id: int, company_id: int):
    """
    Assigns a conversation to a user and updates the session status to 'assigned'.
    Sends a notification to the assigned user.
    """
    from app.services.connection_manager import manager
    from app.models.user import User
    import json

    # First get the session to retrieve current connection status
    session = db.query(models_conversation_session.ConversationSession).filter(
        models_conversation_session.ConversationSession.conversation_id == session_id,
        models_conversation_session.ConversationSession.company_id == company_id
    ).first()

    if not session:
        return False

    # Get the assigned user's information
    assigned_user = db.query(User).filter(User.id == user_id).first()

    # Get contact information for the notification
    contact = session.contact
    contact_name = contact.name if contact else "Unknown Contact"

    # Update the session
    session.status = "assigned"
    session.assignee_id = user_id
    db.commit()
    db.refresh(session)

    # Broadcast status change to all connected agents in real-time, including connection status and assignee
    status_update_message = json.dumps({
        "type": "session_status_update",
        "session_id": session_id,
        "status": "assigned",
        "assignee_id": user_id,
        "is_client_connected": session.is_client_connected,
        "updated_at": session.updated_at.isoformat()
    })
    await manager.broadcast(status_update_message, str(company_id))

    # Send targeted notification to the assigned user
    if assigned_user:
        notification_message = json.dumps({
            "type": "conversation_assigned",
            "session_id": session_id,
            "contact_name": contact_name,
            "channel": session.channel,
            "is_client_connected": session.is_client_connected,
            "assigned_to": assigned_user.email,
            "assigned_to_id": user_id,
            "message": f"You've been assigned to handle conversation with {contact_name}",
            "timestamp": session.updated_at.isoformat()
        })
        print(f"[update_conversation_assignee] Sending assignment notification to user {user_id} ({assigned_user.email})")
        print(f"[update_conversation_assignee] Notification payload: {notification_message}")
        # Broadcast to the entire company (frontend will filter for the assigned user)
        await manager.broadcast(notification_message, str(company_id))
        print(f"[update_conversation_assignee] Notification broadcasted to company {company_id}")

    return True

async def send_proactive_message(db: Session, session_id: str, message_text: str):
    """
    Sends a proactive message to a user in a specific session.
    """
    # Note: Proactive messages are sent from the "agent"
    # We can enhance this later to allow specifying the sender.
    message_data = {"message": message_text, "message_type": "message", "sender": "agent"}
    await manager.broadcast_to_session(session_id, json.dumps(message_data), sender_type='agent')
