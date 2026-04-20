from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query, status
from sqlalchemy.orm import Session
from app.services import chat_service, widget_settings_service, workflow_service, agent_execution_service, messaging_service, integration_service, company_service, agent_service, contact_service, conversation_session_service, user_service, workflow_trigger_service, credential_service
from app.services.workflow_execution_service import WorkflowExecutionService
from app.schemas import chat_message as schemas_chat_message, websocket as schemas_websocket
from app.schemas.websockets import WebSocketMessage
import json
import base64
import uuid
from typing import List, Dict, Any, Optional
from app.core.dependencies import get_current_user_from_ws, get_db
from app.core.database import SessionLocal
from app.models import user as models_user, conversation_session as models_conversation_session
from app.models.workflow_trigger import TriggerChannel
from app.services.connection_manager import manager
from app.services.stt_service import STTService, GroqSTTService
from app.services.tts_service import TTSService
from fastapi import UploadFile
import io
import asyncio
import datetime
from contextlib import contextmanager
from jose import JWTError, jwt
from app.core.config import settings
from app.core.object_storage import s3_client, BUCKET_NAME

router = APIRouter()

def upload_attachment_to_s3(file_data_base64: str, file_name: str, file_type: str) -> Optional[str]:
    """Upload base64-encoded file to S3, return URL or None if failed."""
    try:
        file_bytes = base64.b64decode(file_data_base64)
        # Generate unique key with UUID prefix
        safe_filename = file_name.replace(' ', '_')
        key = f"attachments/{uuid.uuid4()}_{safe_filename}"

        s3_client.put_object(
            Bucket=BUCKET_NAME,
            Key=key,
            Body=file_bytes,
            ContentType=file_type
        )

        # Build URL based on MinIO endpoint
        scheme = 'https' if settings.minio_secure else 'http'
        file_url = f"{scheme}://{settings.minio_endpoint}/{BUCKET_NAME}/{key}"
        print(f"[S3 Upload] Successfully uploaded {file_name} to {file_url}")
        return file_url
    except Exception as e:
        print(f"[S3 Upload] Failed to upload {file_name}: {e}")
        return None

def process_attachments_for_storage(attachments: List[Dict[str, Any]]) -> str:
    """
    Process attachments: upload to S3 if available, build message text.
    Returns message text like "📎 filename.jpg" or "📍 Location (lat, lng)"
    """
    parts = []

    for att in attachments:
        if att.get('file_data'):
            # Try to upload to S3
            file_url = upload_attachment_to_s3(
                att['file_data'],
                att.get('file_name', 'file'),
                att.get('file_type', 'application/octet-stream')
            )
            if file_url:
                att['file_url'] = file_url
            # Build display text
            parts.append(f"📎 {att.get('file_name', 'file')}")
        elif att.get('location'):
            loc = att['location']
            lat = loc.get('latitude', 0)
            lng = loc.get('longitude', 0)
            parts.append(f"📍 Location ({lat:.4f}, {lng:.4f})")

    return '\n'.join(parts) if parts else ""

@contextmanager
def get_db_session():
    """Context manager for database sessions - use this instead of holding connections"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def authenticate_ws_user(websocket: WebSocket, token: Optional[str]) -> Optional[models_user.User]:
    """Authenticate user for WebSocket without holding DB session. Returns None if auth fails."""
    if not token:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Authentication token missing")
        return None

    try:
        with get_db_session() as db:
            payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
            email: str = payload.get("sub")
            if not email:
                await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Invalid token")
                return None
            user = user_service.get_user_by_email(db, email=email)
            if not user:
                await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="User not found")
                return None
            return user
    except JWTError:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Could not validate credentials")
        return None


async def heartbeat_handler(websocket: WebSocket, session_id: str):
    """
    Background task that sends periodic ping messages to keep connection alive
    and detect inactive clients.

    Args:
        websocket: The WebSocket connection
        session_id: Session identifier for logging

    This task runs concurrently with the main message loop and sends pings
    at intervals defined by WS_PING_INTERVAL configuration.
    """
    try:
        while True:
            await asyncio.sleep(settings.WS_PING_INTERVAL)
            try:
                await websocket.send_text(json.dumps({"type": "ping"}))
            except Exception as e:
                print(f"[heartbeat] Error sending ping to session {session_id}: {e}")
                break
    except asyncio.CancelledError:
        print(f"[heartbeat] Heartbeat task cancelled for session {session_id}")
        raise
    except Exception as e:
        print(f"[heartbeat] Unexpected error in heartbeat for session {session_id}: {e}")


@router.websocket("/wschat/{channel_id}")
async def internal_chat_websocket_endpoint(
    websocket: WebSocket,
    channel_id: int,
    token: Optional[str] = Query(None)
):
    print(f"Attempting to connect to WebSocket for channel {channel_id}")
    channel_id_str = str(channel_id)

    # Accept connection first
    await manager.connect(websocket, channel_id_str,"user")

    # Then authenticate (will close connection if auth fails)
    current_user = await authenticate_ws_user(websocket, token)
    if not current_user:
        return

    # Note: User-specific channels are available but incoming calls use company channel
    # This provides better reliability as agents are always connected to company channel

    # Cancel any pending offline task (user reconnected)
    user_service.cancel_pending_offline(current_user.id)

    # Update presence with temporary DB session
    with get_db_session() as db:
        user_service.update_user_presence(db, user_id=current_user.id, status="online")

    presence_message = WebSocketMessage(type="presence_update", payload={"user_id": current_user.id, "status": "online"})
    await manager.broadcast(presence_message.model_dump_json(), channel_id_str)

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket, channel_id_str)

        # Schedule delayed offline update (allows reconnection within grace period)
        await user_service.schedule_offline_update(SessionLocal, current_user.id)

        presence_message = WebSocketMessage(type="presence_update", payload={"user_id": current_user.id, "status": "offline"})
        await manager.broadcast(presence_message.model_dump_json(), channel_id_str)

@router.websocket("/voice/{company_id}/{agent_id}/{session_id}")
async def voice_websocket_endpoint(
    websocket: WebSocket,
    company_id: int,
    agent_id: int,
    session_id: str,
    user_type: str = Query(...),
    voice_id: str = Query("21m00Tcm4TlvDq8ikWAM") # Default voice
):
    # Fetch agent details to get the configured voice (use temporary DB session)
    with get_db_session() as db:
        agent = agent_service.get_agent(db, agent_id, company_id)
    final_voice_id = voice_id
    tts_provider = 'voice_engine' # Default provider
    stt_provider = 'deepgram' # Default provider
    if agent:
        if agent.voice_id:
            final_voice_id = agent.voice_id
        if agent.tts_provider:
            tts_provider = agent.tts_provider
        if agent.stt_provider:
            stt_provider = agent.stt_provider

    await manager.connect(websocket, session_id, user_type)
    
    stt_service = None
    if stt_provider == "deepgram":
        stt_service = STTService(websocket)
    else:
        stt_service = GroqSTTService()

    tts_service = TTSService()
    
    transcript_queue = asyncio.Queue()
    audio_buffer = bytearray()
    last_audio_time = None

    async def handle_transcription():
        if stt_provider == "deepgram":
            if await stt_service.connect():
                while True:
                    try:
                        message = await stt_service.deepgram_ws.receive_json()
                        if message.get("type") == "Results":
                            transcript = message["channel"]["alternatives"][0]["transcript"]
                            if transcript and message.get("is_final", False):
                                await transcript_queue.put(transcript)
                    except Exception as e:
                        print(f"Error receiving from STT service: {e}")
                        break
            await stt_service.close()
        # Groq does not use a persistent connection for transcription

    async def handle_audio_from_client():
        nonlocal audio_buffer, last_audio_time
        try:
            while True:
                audio_chunk = await websocket.receive_bytes()
                last_audio_time = asyncio.get_event_loop().time()
                if stt_provider == "deepgram":
                    if stt_service.deepgram_ws and not stt_service.deepgram_ws.closed:
                        await stt_service.deepgram_ws.send_bytes(audio_chunk)
                    else:
                        break
                elif stt_provider == "groq":
                    audio_buffer.extend(audio_chunk)

        except WebSocketDisconnect:
            pass
        except Exception as e:
            print(f"Error receiving audio from client: {e}")

    async def process_groq_buffer():
        nonlocal audio_buffer, last_audio_time
        while True:
            await asyncio.sleep(0.5) # Check every 500ms
            if audio_buffer and last_audio_time and (asyncio.get_event_loop().time() - last_audio_time > 1.0): # 1 second pause
                try:
                    # Create an UploadFile-like object from the buffer
                    audio_file = UploadFile(filename="audio.wav", file=io.BytesIO(audio_buffer), content_type="audio/wav")
                    transcript_result = await stt_service.transcribe(audio_file)
                    transcript = transcript_result.get("text")
                    if transcript:
                        await transcript_queue.put(transcript)
                except Exception as e:
                    print(f"Error during Groq transcription: {e}")
                finally:
                    audio_buffer.clear()
                    last_audio_time = None


    transcription_task = asyncio.create_task(handle_transcription())
    client_audio_task = asyncio.create_task(handle_audio_from_client())
    
    groq_buffer_task = None
    if stt_provider == "groq":
        groq_buffer_task = asyncio.create_task(process_groq_buffer())


    try:
        while True:
            transcript = await transcript_queue.get()
            
            if transcript:
                # Use temporary DB sessions for each database operation
                with get_db_session() as db:
                    # 1. Save and broadcast the user's transcribed message
                    user_message = schemas_chat_message.ChatMessageCreate(message=transcript, message_type='message')
                    db_user_message = chat_service.create_chat_message(db, user_message, agent_id, session_id, company_id, "user", assignee_id=None)
                    await manager.broadcast_to_session(session_id, schemas_chat_message.ChatMessage.model_validate(db_user_message).model_dump_json(), "user")

                    # Check if AI is enabled for this session
                    session_obj = db.query(models_conversation_session.ConversationSession).filter(
                        models_conversation_session.ConversationSession.conversation_id == session_id,
                        models_conversation_session.ConversationSession.company_id == company_id
                    ).first()

                    if session_obj and not session_obj.is_ai_enabled:
                        print(f"AI is disabled for session {session_id}. No voice response will be generated.")
                        transcript_queue.task_done()
                        continue

                    # 2. Generate the agent's response
                    agent_response_text = await agent_execution_service.generate_agent_response(
                        db, agent_id, session_id, company_id, transcript
                    )

                    # 3. Save and broadcast the agent's text message
                    agent_message = schemas_chat_message.ChatMessageCreate(message=str(agent_response_text), message_type='message')
                    db_agent_message = chat_service.create_chat_message(db, agent_message, agent_id, session_id, company_id, "agent", assignee_id=None)
                    await manager.broadcast_to_session(session_id, schemas_chat_message.ChatMessage.model_validate(db_agent_message).model_dump_json(), "agent")

                # 4. Convert the agent's response to speech and stream it
                audio_stream = tts_service.text_to_speech_stream(agent_response_text, final_voice_id, tts_provider)
                async for audio_chunk in audio_stream:
                    await websocket.send_bytes(audio_chunk)

                transcript_queue.task_done()

    except WebSocketDisconnect:
        print(f"Client in voice session #{session_id} disconnected")
    except Exception as e:
        print(f"Error in main voice processing loop: {e}")
    finally:
        # Cancel tasks and wait for them to finish
        transcription_task.cancel()
        client_audio_task.cancel()
        if groq_buffer_task:
            groq_buffer_task.cancel()

        # Wait for cancellation to complete
        try:
            await transcription_task
        except asyncio.CancelledError:
            pass
        try:
            await client_audio_task
        except asyncio.CancelledError:
            pass
        if groq_buffer_task:
            try:
                await groq_buffer_task
            except asyncio.CancelledError:
                pass

        await tts_service.close()
        manager.disconnect(websocket, session_id)
        print(f"Cleaned up resources for voice session #{session_id}")



@router.websocket("/internal/voice/{agent_id}/{session_id}")
async def internal_voice_websocket_endpoint(
    websocket: WebSocket,
    agent_id: int,
    session_id: str,
    user_type: str = Query(...),
    voice_id: str = Query("default_voice_id"),
    token: Optional[str] = Query(None)
):
    # Accept connection first
    await manager.connect(websocket, session_id, user_type)

    # Then authenticate (will close connection if auth fails)
    current_user = await authenticate_ws_user(websocket, token)
    if not current_user:
        return

    company_id = current_user.company_id

    # Fetch agent with temporary DB session
    with get_db_session() as db:
        agent = agent_service.get_agent(db, agent_id, company_id)
    stt_provider = 'deepgram' # Default provider
    if agent and agent.stt_provider:
        stt_provider = agent.stt_provider
    
    stt_service = None
    if stt_provider == "deepgram":
        stt_service = STTService(websocket)
    else:
        stt_service = GroqSTTService()

    tts_service = TTSService()
    
    transcript_queue = asyncio.Queue()
    audio_buffer = bytearray()
    last_audio_time = None

    async def handle_transcription():
        if stt_provider == "deepgram":
            if await stt_service.connect():
                while True:
                    try:
                        message = await stt_service.deepgram_ws.receive_json()
                        if message.get("type") == "Results":
                            transcript = message["channel"]["alternatives"][0]["transcript"]
                            if transcript and message.get("is_final", False):
                                await transcript_queue.put(transcript)
                    except Exception as e:
                        print(f"Error receiving from STT service: {e}")
                        break
            await stt_service.close()
        # Groq does not use a persistent connection for transcription

    async def handle_audio_from_client():
        nonlocal audio_buffer, last_audio_time
        try:
            while True:
                audio_chunk = await websocket.receive_bytes()
                last_audio_time = asyncio.get_event_loop().time()
                if stt_provider == "deepgram":
                    if stt_service.deepgram_ws and not stt_service.deepgram_ws.closed:
                        await stt_service.deepgram_ws.send_bytes(audio_chunk)
                    else:
                        break
                elif stt_provider == "groq":
                    audio_buffer.extend(audio_chunk)

        except WebSocketDisconnect:
            pass
        except Exception as e:
            print(f"Error receiving audio from client: {e}")

    async def process_groq_buffer():
        nonlocal audio_buffer, last_audio_time
        while True:
            await asyncio.sleep(0.5) # Check every 500ms
            if audio_buffer and last_audio_time and (asyncio.get_event_loop().time() - last_audio_time > 1.0): # 1 second pause
                try:
                    # Create an UploadFile-like object from the buffer
                    audio_file = UploadFile(filename="audio.wav", file=io.BytesIO(audio_buffer), content_type="audio/wav")
                    transcript_result = await stt_service.transcribe(audio_file)
                    transcript = transcript_result.get("text")
                    if transcript:
                        await transcript_queue.put(transcript)
                except Exception as e:
                    print(f"Error during Groq transcription: {e}")
                finally:
                    audio_buffer.clear()
                    last_audio_time = None

    transcription_task = asyncio.create_task(handle_transcription())
    client_audio_task = asyncio.create_task(handle_audio_from_client())

    groq_buffer_task = None
    if stt_provider == "groq":
        groq_buffer_task = asyncio.create_task(process_groq_buffer())

    try:
        while True:
            transcript = await transcript_queue.get()
            
            if transcript:
                # When an agent speaks, we treat it as a message from the agent
                # and send it to the user.
                # Use temporary DB session
                with get_db_session() as db:
                    chat_message = schemas_chat_message.ChatMessageCreate(message=transcript, message_type='message')
                    db_message = chat_service.create_chat_message(db, chat_message, agent_id, session_id, company_id, "agent", assignee_id=current_user.id)

                    # Enrich message with assignee name for broadcast
                    message_dict = schemas_chat_message.ChatMessage.model_validate(db_message).model_dump(mode='json')
                    if db_message.assignee_id:
                        assignee_user = db.query(models_user.User).filter(models_user.User.id == db_message.assignee_id).first()
                        if assignee_user:
                            # Build full name from first_name and last_name
                            name_parts = []
                            if assignee_user.first_name:
                                name_parts.append(assignee_user.first_name)
                            if assignee_user.last_name:
                                name_parts.append(assignee_user.last_name)
                            message_dict['assignee_name'] = ' '.join(name_parts) if name_parts else assignee_user.email

                    await manager.broadcast_to_session(session_id, json.dumps(message_dict), "agent")

                transcript_queue.task_done()

    except WebSocketDisconnect:
        print(f"Agent in voice session #{session_id} disconnected")
    except Exception as e:
        print(f"Error in internal voice processing loop: {e}")
    finally:
        # Cancel tasks and wait for them to finish
        transcription_task.cancel()
        client_audio_task.cancel()
        if groq_buffer_task:
            groq_buffer_task.cancel()

        # Wait for cancellation to complete
        try:
            await transcription_task
        except asyncio.CancelledError:
            pass
        try:
            await client_audio_task
        except asyncio.CancelledError:
            pass
        if groq_buffer_task:
            try:
                await groq_buffer_task
            except asyncio.CancelledError:
                pass

        await tts_service.close()
        manager.disconnect(websocket, session_id)
        print(f"Cleaned up resources for internal voice session #{session_id}")




@router.websocket("/{agent_id}/{session_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    agent_id: int,
    session_id: str,
    user_type: str = Query(...), # 'user' or 'agent'
    token: Optional[str] = Query(None)
):
    print(f"[websocket_conversations] Attempting WebSocket connection for session: {session_id}")

    # Accept connection first
    await manager.connect(websocket, session_id, user_type)
    print(f"[websocket_conversations] WebSocket connection established for session: {session_id}")

    # Then authenticate (will close connection if auth fails)
    current_user = await authenticate_ws_user(websocket, token)
    if not current_user:
        return

    company_id = current_user.company_id # Get company_id from authenticated user
    print(f"[websocket_conversations] Authenticated user: {current_user.email}, company_id: {company_id}")

    # Update session status to active when user connects (use temporary DB session)
    if user_type == "user":
        with get_db_session() as db:
            await conversation_session_service.update_session_connection_status(db, session_id, is_connected=True)

    # Start heartbeat task if enabled
    heartbeat_task = None
    if settings.WS_ENABLE_HEARTBEAT:
        heartbeat_task = asyncio.create_task(heartbeat_handler(websocket, session_id))

    try:
        while True:
            data = await websocket.receive_text()

            # Update activity timestamp
            manager.update_activity(session_id, websocket)

            print(f"[websocket_conversations] Received data from frontend: {data}")
            if not data:
                continue

            try:
                message_data = json.loads(data)

                # Log raw incoming message (excluding large file_data)
                log_data = {k: v for k, v in message_data.items() if k != 'attachments'}
                if 'attachments' in message_data and message_data['attachments']:
                    log_data['attachments'] = [
                        {k: v for k, v in att.items() if k != 'file_data'}
                        for att in message_data['attachments']
                    ]
                    log_data['attachments_count'] = len(message_data['attachments'])
                    log_data['has_file_data'] = any('file_data' in att for att in message_data['attachments'])
                print(f"[websocket_conversations] 📥 RAW MESSAGE RECEIVED: {log_data}")

                # Handle ping/pong messages
                if message_data.get('type') == 'pong':
                    continue  # Just an acknowledgment, no further processing
                if message_data.get('type') == 'ping':
                    await websocket.send_text(json.dumps({"type": "pong"}))
                    continue

                # Handle agent typing events from dashboard
                if message_data.get('type') == 'agent_typing':
                    is_typing = message_data.get('is_typing', False)
                    typing_session_id = message_data.get('session_id')

                    if typing_session_id:
                        # Broadcast typing indicator to the widget user
                        await manager.broadcast_to_session(
                            typing_session_id,
                            json.dumps({
                                "message_type": "typing",
                                "is_typing": is_typing,
                                "sender": "agent"
                            }),
                            "agent"
                        )
                        print(f"[websocket_conversations] Agent typing event: is_typing={is_typing}, session={typing_session_id}")
                    continue

                user_message = message_data.get('message')
                sender = message_data.get('sender')
                option_key = message_data.get('option_key')  # Key for workflow variable (when user selects an option)
                attachments = message_data.get('attachments', [])  # Get attachments from message

                # Log attachment info
                if attachments:
                    print(f"[websocket_conversations] 📎 Received {len(attachments)} attachment(s) from session #{session_id}")
                    for i, att in enumerate(attachments):
                        if att.get('location'):
                            loc = att['location']
                            print(f"[websocket_conversations]   - Attachment {i+1}: 📍 Location ({loc.get('latitude')}, {loc.get('longitude')})")
                        else:
                            print(f"[websocket_conversations]   - Attachment {i+1}: {att.get('file_name')} ({att.get('file_type')}, {att.get('file_size')} bytes)")
                else:
                    print(f"[websocket_conversations] No attachments in message from session #{session_id}")

            except (json.JSONDecodeError, AttributeError):
                print(f"[websocket_conversations] Received invalid data from session #{session_id}: {data}")
                continue

            # Allow messages with attachments even if text is empty
            if (not user_message and not attachments) or not sender:
                print(f"[websocket_conversations] Missing user_message/attachments or sender: user_message={user_message}, attachments={len(attachments)}, sender={sender}")
                continue

            # Process attachments: upload to S3 and build display text
            attachment_text = ""
            if attachments:
                print(f"[websocket_conversations] 📎 Processing {len(attachments)} attachment(s)")
                attachment_text = process_attachments_for_storage(attachments)

            # Build message for storage
            message_to_store = user_message
            if not user_message and attachment_text:
                # If no text but attachments, use attachment description
                message_to_store = attachment_text

            # Create a temporary DB session for this message handling
            with get_db_session() as db:
                # Instantiate the execution service with this DB session
                workflow_exec_service = WorkflowExecutionService(db)

                # 1. Log user message
                chat_message = schemas_chat_message.ChatMessageCreate(
                    message=message_to_store,
                    message_type=message_data.get('message_type', 'message'),
                    token=message_data.get('token')
                )

                # Determine assignee_id: if agent is sending, use current_user.id
                assignee_id = current_user.id if sender == 'agent' else None
                db_message = chat_service.create_chat_message(db, chat_message, agent_id, session_id, company_id, sender, assignee_id, attachments=attachments if attachments else None)
                print(f"[websocket_conversations] Created chat message: {db_message.id}")

                # Enrich message with assignee name for broadcast
                message_dict = schemas_chat_message.ChatMessage.model_validate(db_message).model_dump(mode='json')
                if sender == 'agent' and db_message.assignee_id:
                    assignee_user = db.query(models_user.User).filter(models_user.User.id == db_message.assignee_id).first()
                    if assignee_user:
                        # Build full name from first_name and last_name
                        name_parts = []
                        if assignee_user.first_name:
                            name_parts.append(assignee_user.first_name)
                        if assignee_user.last_name:
                            name_parts.append(assignee_user.last_name)
                        message_dict['assignee_name'] = ' '.join(name_parts) if name_parts else assignee_user.email

                # Include attachments in broadcast for preview display
                if attachments:
                    message_dict['attachments'] = attachments

                await manager.broadcast_to_session(session_id, json.dumps(message_dict), sender)
                print(f"[websocket_conversations] Broadcasted message to session: {session_id}")

                # OPTIMIZATION: Check and send typing indicator IMMEDIATELY for user messages
                # This happens before workflow/AI processing to show immediate feedback
                typing_indicator_sent = False
                if sender == 'user':
                    # Quick check for AI enabled (single DB query, cached by SQLAlchemy)
                    agent = agent_service.get_agent(db, agent_id, company_id)
                    ai_enabled = agent and agent.credential is not None

                    if ai_enabled:
                        # Quick check for typing indicator setting (single DB query, cached)
                        widget_settings = widget_settings_service.get_widget_settings(db, agent_id)
                        if widget_settings and widget_settings.typing_indicator_enabled:
                            # Send typing indicator IMMEDIATELY before any workflow/AI processing
                            await manager.broadcast_to_session(
                                session_id,
                                json.dumps({"message_type": "typing", "is_typing": True, "sender": "agent"}),
                                "agent"
                            )
                            typing_indicator_sent = True
                            print(f"[websocket_conversations] ⚡ Typing indicator ON (immediate) for session: {session_id}")

                # If agent sends message, check session channel and send to external platform
                if sender == 'agent':
                    session_obj = db.query(models_conversation_session.ConversationSession).filter(
                        models_conversation_session.ConversationSession.conversation_id == session_id,
                        models_conversation_session.ConversationSession.company_id == company_id
                    ).first()

                    if session_obj and session_obj.channel == 'whatsapp':
                        # Retrieve WhatsApp credentials for the company
                        # This part needs to be implemented based on your credential management
                        # For now, let's assume you have a way to get api_token and phone_number_id
                        # from your settings or integration service.
                        # Placeholder for actual credential retrieval
                        whatsapp_integration = integration_service.get_integration_by_type_and_company(db, "whatsapp", company_id)

                        if whatsapp_integration:
                            try:
                                # Decrypt credentials to get api_token and phone_number_id
                                whatsapp_credentials = integration_service.get_decrypted_credentials(whatsapp_integration)
                                api_token = whatsapp_credentials.get("api_token") or whatsapp_credentials.get("access_token")
                                phone_number_id = whatsapp_credentials.get("phone_number_id")

                                if not api_token or not phone_number_id:
                                    print(f"[websocket_conversations] WhatsApp credentials missing for company {company_id}. api_token: {bool(api_token)}, phone_number_id: {bool(phone_number_id)}")
                                    continue  # Changed from return to continue

                                await messaging_service.send_whatsapp_message(
                                    recipient_phone_number=session_obj.contact.phone_number,
                                    message_text=user_message,
                                    integration=whatsapp_integration,
                                    db=db
                                )
                                print(f"[websocket_conversations] Sent message to WhatsApp for session {session_id}")
                            except Exception as e:
                                print(f"[websocket_conversations] Error sending to WhatsApp: {e}")

                # 2. If the message is from the user, execute the workflow
                if sender == 'user':
                    # Check if AI is enabled for this session
                    session_obj = db.query(models_conversation_session.ConversationSession).filter(
                        models_conversation_session.ConversationSession.conversation_id == session_id,
                        models_conversation_session.ConversationSession.company_id == company_id
                    ).first()

                    if session_obj and not session_obj.is_ai_enabled:
                        print(f"AI is disabled for session {session_id}. No response will be generated.")
                        continue

                    execution_result = None
                    try:
                        # Priority: 1) Triggers, 2) LLM decision, 3) Similarity search
                        workflow = None
                        execution_result = None

                        # 1. Try trigger-based workflow finding first
                        print(f"[websocket_conversations] Calling trigger service for company_id={company_id}, channel=WEBSOCKET")
                        try:
                            workflow = await workflow_trigger_service.find_workflow_for_channel_message(
                                db=db,
                                channel=TriggerChannel.WEBSOCKET,
                                company_id=company_id,
                                message=user_message,
                                session_data={"session_id": session_id, "agent_id": agent_id}
                            )
                            print(f"[websocket_conversations] Trigger service returned: {workflow.name if workflow else None}")
                        except Exception as trigger_error:
                            print(f"[websocket_conversations] ERROR in trigger service: {trigger_error}")
                            import traceback
                            traceback.print_exc()
                            workflow = None

                        # If trigger found workflow, execute it
                        if workflow:
                            print(f"[websocket_conversations] 🚀 Executing trigger-matched workflow with {len(attachments)} attachment(s)")
                            execution_result = await workflow_exec_service.execute_workflow(
                                user_message=user_message,
                                conversation_id=session_id,
                                company_id=company_id,
                                workflow=workflow,
                                attachments=attachments,
                                option_key=option_key
                            )
                        else:
                            # 2. No trigger match - try LLM-based routing (2nd priority)
                            print(f"[websocket_conversations] No trigger match, trying LLM-based routing")

                            # Check if streaming is enabled
                            should_stream = settings.LLM_STREAMING_ENABLED
                            if widget_settings and hasattr(widget_settings, 'streaming_enabled'):
                                should_stream = widget_settings.streaming_enabled

                            if should_stream:
                                # STREAMING MODE: Stream tokens as they arrive
                                # Note: In streaming mode, we can't easily check for workflow triggers mid-stream
                                # So we stream first, then check similarity as fallback
                                print(f"[websocket_conversations] Using streaming mode for session: {session_id}")
                                full_response = ""
                                async for token_json in agent_execution_service.generate_agent_response_stream(
                                    db, agent_id, session_id, session_id, company_id, message_data['message']
                                ):
                                    # Forward streaming tokens to client
                                    await manager.broadcast_to_session(session_id, token_json, "agent")

                                    # Parse to accumulate full response
                                    try:
                                        token_data = json.loads(token_json)
                                        if token_data.get('type') == 'stream':
                                            full_response += token_data.get('content', '')
                                        elif token_data.get('type') in ['stream_end', 'complete']:
                                            full_response = token_data.get('full_content', full_response) or token_data.get('content', full_response)
                                    except:
                                        pass

                                # Save the complete message to database
                                if full_response:
                                    agent_message = schemas_chat_message.ChatMessageCreate(message=full_response, message_type="message")
                                    db_agent_message = chat_service.create_chat_message(db, agent_message, agent_id, session_id, company_id, "agent", assignee_id=None)
                                    print(f"[websocket_conversations] Saved streamed response to database for session: {session_id}")
                                continue
                            else:
                                # NON-STREAMING MODE: Can check LLM response for workflow trigger
                                print(f"[websocket_conversations] Using non-streaming mode for session: {session_id}")
                                agent_response = await agent_execution_service.generate_agent_response(
                                    db, agent_id, session_id, session_id, company_id, message_data['message']
                                )

                                # Check if LLM decided to trigger a workflow (context-aware routing)
                                if isinstance(agent_response, dict) and agent_response.get("type") == "workflow_trigger":
                                    workflow_id = agent_response.get("workflow_id")
                                    print(f"[websocket_conversations] LLM triggered workflow {workflow_id}")
                                    workflow = workflow_service.get_workflow(db, workflow_id, company_id)
                                    if workflow:
                                        execution_result = await workflow_exec_service.execute_workflow(
                                            user_message=user_message,
                                            conversation_id=session_id,
                                            company_id=company_id,
                                            workflow=workflow,
                                            attachments=attachments,
                                            option_key=option_key
                                        )
                                        # execution_result will be handled below
                                    else:
                                        print(f"[websocket_conversations] Workflow {workflow_id} not found")
                                        continue
                                elif isinstance(agent_response, dict) and agent_response.get("type") == "handoff":
                                    # LLM routing failed - initiate human handoff
                                    reason = agent_response.get("reason", "AI routing unavailable")
                                    print(f"[websocket_conversations] LLM failed, initiating handoff: {reason}")
                                    error_msg = "I'm experiencing some technical difficulties. Let me connect you with a human agent who can help."
                                    agent_message = schemas_chat_message.ChatMessageCreate(message=error_msg, message_type="message")
                                    db_agent_message = chat_service.create_chat_message(db, agent_message, agent_id, session_id, company_id, "agent", assignee_id=None)
                                    await manager.broadcast_to_session(session_id, schemas_chat_message.ChatMessage.model_validate(db_agent_message).model_dump_json(), "agent")
                                    continue
                                else:
                                    # LLM returned normal text - try similarity search as 3rd fallback
                                    print(f"[websocket_conversations] LLM returned text, trying similarity search as fallback")
                                    similar_workflow = workflow_service.find_similar_workflow(
                                        db, company_id=company_id, query=user_message, agent_id=agent_id
                                    )

                                    if similar_workflow:
                                        print(f"[websocket_conversations] 🚀 Found similar workflow '{similar_workflow.name}'")
                                        execution_result = await workflow_exec_service.execute_workflow(
                                            user_message=user_message,
                                            conversation_id=session_id,
                                            company_id=company_id,
                                            workflow=similar_workflow,
                                            attachments=attachments,
                                            option_key=option_key
                                        )
                                        # execution_result will be handled below
                                    else:
                                        # No workflow found anywhere - use LLM's text response
                                        agent_response_text = agent_response if isinstance(agent_response, str) else str(agent_response)
                                        agent_message = schemas_chat_message.ChatMessageCreate(message=agent_response_text, message_type="message")
                                        db_agent_message = chat_service.create_chat_message(db, agent_message, agent_id, session_id, company_id, "agent", assignee_id=None)
                                        await manager.broadcast_to_session(session_id, schemas_chat_message.ChatMessage.model_validate(db_agent_message).model_dump_json(), "agent")
                                        print(f"[websocket_conversations] Broadcasted agent response to session: {session_id}")
                                        continue
                    finally:
                        # Turn off typing indicator after workflow/AI processing completes
                        if typing_indicator_sent:
                            await manager.broadcast_to_session(
                                session_id,
                                json.dumps({"message_type": "typing", "is_typing": False, "sender": "agent"}),
                                "agent"
                            )
                            print(f"[websocket_conversations] Typing indicator OFF for session: {session_id}")

                    if not execution_result:
                        continue

                    # 4. Handle the execution result
                    if execution_result.get("status") == "completed":
                        agent_response_text = execution_result.get("response", "Workflow finished.")
                        agent_message = schemas_chat_message.ChatMessageCreate(message=str(agent_response_text), message_type="message")
                        db_agent_message = chat_service.create_chat_message(db, agent_message, agent_id, session_id, company_id, "agent", assignee_id=None)
                        await manager.broadcast_to_session(session_id, schemas_chat_message.ChatMessage.model_validate(db_agent_message).model_dump_json(), "agent")
                        print(f"[websocket_conversations] Broadcasted workflow completion to session: {session_id}")

                    elif execution_result.get("status") == "paused_for_prompt":
                        # The workflow is paused and wants to prompt the user.
                        prompt_data = execution_result.get("prompt", {})
                        options = prompt_data.get("options", [])
                        allow_text_input = prompt_data.get("allow_text_input", False)

                        # Save prompt message to database (including options for history)
                        prompt_chat_message = schemas_chat_message.ChatMessageCreate(
                            message=prompt_data.get("text", ""),
                            message_type="prompt"
                        )
                        db_prompt_message = chat_service.create_chat_message(
                            db, prompt_chat_message, agent_id, session_id, company_id, "agent", assignee_id=None, options=options
                        )

                        # Broadcast message with options and allow_text_input for frontend display
                        message_dict = schemas_chat_message.ChatMessage.model_validate(db_prompt_message).model_dump(mode='json')
                        message_dict['options'] = options
                        message_dict['allow_text_input'] = allow_text_input
                        await manager.broadcast_to_session(session_id, json.dumps(message_dict), "agent")
                        print(f"[websocket_conversations] Saved and broadcasted prompt to session: {session_id}")

                    elif execution_result.get("status") == "paused_for_form":
                        form_data = execution_result.get("form", {})
                        form_message = {
                            "message": form_data.get("title"),
                            "message_type": "form",
                            "fields": form_data.get("fields", []),
                            "sender": "agent",
                            "session_id": session_id,
                            "agent_id": agent_id,
                            "company_id": company_id,
                        }
                        await manager.broadcast_to_session(session_id, json.dumps(form_message), "agent")
                        print(f"[websocket_conversations] Broadcasted form to session: {session_id}")

                    elif execution_result.get("status") == "paused_for_input":
                        # Broadcast input constraint to widget so it knows what input type is expected
                        expected_input_type = execution_result.get("expected_input_type", "any")
                        input_constraint_message = {
                            "message_type": "input_constraint",
                            "expected_input_type": expected_input_type,
                            "sender": "agent"
                        }
                        await manager.broadcast_to_session(session_id, json.dumps(input_constraint_message), "agent")
                        print(f"[websocket_conversations] Broadcasted input constraint ({expected_input_type}) to session: {session_id}")

    except WebSocketDisconnect:
        manager.disconnect(websocket, session_id)
        print(f"Client in session #{session_id} disconnected")

        # Update session status to inactive when user disconnects
        # Only if there are no more user connections
        if user_type == "user" and not manager.has_user_connection(session_id):
            with get_db_session() as db:
                await conversation_session_service.update_session_connection_status(db, session_id, is_connected=False)

    finally:
        # Cancel heartbeat task
        if heartbeat_task:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass


@router.websocket("/public/{company_id}/{agent_id}/{session_id}")
async def public_websocket_endpoint(
    websocket: WebSocket,
    company_id: int,
    agent_id: int,
    session_id: str,
    user_type: str = Query(...),   # 'user' or 'agent'
    channel: Optional[str] = Query(default="web"),  # preview/channel type from widget
):
    # Verify agent exists with temporary DB session
    with get_db_session() as db:
        agent = agent_service.get_agent(db, agent_id, company_id)
        if not agent:
            await websocket.close(code=1008)
            return

    await manager.connect(websocket, session_id, user_type)

    # Send message history for existing sessions (user reconnection)
    if user_type == "user":
        with get_db_session() as db:
            # Check if session exists
            existing_session = db.query(models_conversation_session.ConversationSession).filter(
                models_conversation_session.ConversationSession.conversation_id == session_id,
                models_conversation_session.ConversationSession.company_id == company_id
            ).first()

            if existing_session:
                # Fetch last 20 messages for this session
                history_messages = chat_service.get_chat_messages(
                    db, agent_id, session_id, company_id, limit=20
                )
                if history_messages:
                    history_payload = {
                        "type": "history",
                        "messages": [
                            schemas_chat_message.ChatMessage.model_validate(msg).model_dump(mode='json')
                            for msg in history_messages
                        ]
                    }
                    await websocket.send_text(json.dumps(history_payload))
                    print(f"[websocket] Sent {len(history_messages)} history messages to session {session_id}")

    # Update session status to active when user connects
    if user_type == "user":
        with get_db_session() as db:
            await conversation_session_service.update_session_connection_status(db, session_id, is_connected=True)

    # Start heartbeat task if enabled
    heartbeat_task = None
    if settings.WS_ENABLE_HEARTBEAT:
        heartbeat_task = asyncio.create_task(heartbeat_handler(websocket, session_id))

    try:
        while True:
            data = await websocket.receive_text()

            # Update activity timestamp
            manager.update_activity(session_id, websocket)

            message_data = json.loads(data)

            # Handle ping/pong messages
            if message_data.get('type') == 'pong':
                continue  # Just an acknowledgment, no further processing
            if message_data.get('type') == 'ping':
                await websocket.send_text(json.dumps({"type": "pong"}))
                continue
            user_message = message_data.get('message')
            sender = message_data.get('sender')
            option_key = message_data.get('option_key')  # Key for workflow variable (when user selects an option)
            attachments = message_data.get('attachments', [])  # Get attachments from message

            # Log attachment info
            if attachments:
                print(f"[websocket_conversations] 📎 PUBLIC: Received {len(attachments)} attachment(s) from session #{session_id}")
                for i, att in enumerate(attachments):
                    print(f"[websocket_conversations]   - Attachment {i+1}: {att.get('file_name')} ({att.get('file_type')}, {att.get('file_size')} bytes)")
            else:
                print(f"[websocket_conversations] PUBLIC: No attachments in message from session #{session_id}")

            # Allow messages with attachments even if text is empty
            if (not user_message and not attachments) or not sender:
                print(f"[websocket_conversations] PUBLIC: Missing user_message/attachments or sender")
                continue

            # Process attachments: upload to S3 and build display text
            attachment_text = ""
            if attachments:
                print(f"[websocket_conversations] 📎 PUBLIC: Processing {len(attachments)} attachment(s)")
                attachment_text = process_attachments_for_storage(attachments)

            # Use temporary DB session for each message
            with get_db_session() as db:
                workflow_exec_service = WorkflowExecutionService(db)

                # Check if this is a new session
                existing_session = db.query(models_conversation_session.ConversationSession).filter(
                    models_conversation_session.ConversationSession.conversation_id == session_id,
                    models_conversation_session.ConversationSession.company_id == company_id
                ).first()
                is_new_session = existing_session is None

                # Resolve workflow_id from the agent's linked workflows (first one is the default).
                # Only needed for new sessions; get_or_create_session won't overwrite an existing workflow_id.
                agent_for_session = agent_service.get_agent(db, agent_id, company_id)
                default_workflow_id = (
                    agent_for_session.workflows[0].id
                    if agent_for_session and agent_for_session.workflows
                    else None
                )

                # Normalise the channel value coming from the widget's previewType.
                channel_map = {
                    "web": "web",
                    "whatsapp": "whatsapp",
                    "messenger": "messenger",
                    "instagram": "instagram",
                    "telegram": "telegram",
                    "voice": "voice",
                }
                session_channel = channel_map.get(channel, "web")

                # Create or fetch session — contact is set later when user provides info
                session = conversation_session_service.get_or_create_session(
                    db,
                    conversation_id=session_id,
                    workflow_id=default_workflow_id,
                    contact_id=None,
                    channel=session_channel,
                    company_id=company_id,
                    agent_id=agent_id,
                )

                # Broadcast new session creation to all company users
                if is_new_session:
                    print(f"[websocket_conversations] 🆕 New session created: {session_id}. Broadcasting to company {company_id}")
                    session_update_schema = schemas_websocket.WebSocketSessionUpdate.model_validate(session)
                    await manager.broadcast_to_company(
                        company_id,
                        json.dumps({"type": "new_session", "session": session_update_schema.model_dump(by_alias=True)})
                    )
                    print(f"[websocket_conversations] ✅ Broadcasted new session to company {company_id}")

                # Check if session was resolved and reopen it when client sends message
                if session.status == 'resolved' and sender == 'user':
                    import datetime

                    old_status = session.status

                    # Track reopening metadata
                    session.reopen_count = (session.reopen_count or 0) + 1
                    session.last_reopened_at = datetime.datetime.utcnow()

                    # Calculate time since resolution (for analytics)
                    time_since_resolution = None
                    if session.resolved_at:
                        time_since_resolution = (datetime.datetime.utcnow() - session.resolved_at).total_seconds()

                    # Set status based on assignee:
                    # - If assignee exists: set to 'assigned' (goes to assigned user's "mine" tab)
                    # - If no assignee: set to 'active' (goes to "open" tab for everyone)
                    if session.assignee_id:
                        session.status = 'assigned'
                    else:
                        session.status = 'active'

                    db.commit()
                    db.refresh(session)

                    # Add system message about reopening
                    reopen_text = f"Conversation reopened by customer"
                    if session.reopen_count > 1:
                        reopen_text += f" (Reopened {session.reopen_count}x)"

                    system_message = schemas_chat_message.ChatMessageCreate(
                        message=reopen_text,
                        message_type="system"
                    )
                    db_system_message = chat_service.create_chat_message(
                        db, system_message, agent_id, session_id, company_id, 'system', assignee_id=None
                    )

                    # Broadcast system message to session
                    await manager.broadcast_to_session(
                        str(session_id),
                        schemas_chat_message.ChatMessage.model_validate(db_system_message).model_dump_json(),
                        'system'
                    )

                    # Get contact information for the broadcast
                    contact_name = session.contact.name if session.contact else None

                    # Enhanced broadcast with metadata
                    await manager.broadcast_to_company(
                        company_id,
                        json.dumps({
                            "type": "session_reopened",
                            "session_id": session_id,
                            "conversation_id": session.conversation_id,
                            "status": session.status,  # Will be 'assigned' or 'active' based on assignee
                            "previous_status": old_status,
                            "assignee_id": session.assignee_id,
                            "contact_name": contact_name,
                            "reopen_count": session.reopen_count,
                            "last_reopened_at": session.last_reopened_at.isoformat(),
                            "resolved_at": session.resolved_at.isoformat() if session.resolved_at else None,
                            "time_since_resolution": time_since_resolution,
                            "updated_at": session.updated_at.isoformat()
                        })
                    )

                    print(f"[websocket_conversations] 🔄 Session {session_id} reopened from resolved → {session.status} (Reopen #{session.reopen_count}, Assignee: {session.assignee_id or 'None'})")

                # Handle form data: convert dict to JSON string for storage
                message_for_storage = user_message
                if isinstance(user_message, dict):
                    # If it's form data, convert to JSON string for chat message storage
                    message_for_storage = json.dumps(user_message)
                elif not user_message and attachment_text:
                    # If no text but attachments, use attachment description (📎 filename, 📍 location)
                    message_for_storage = attachment_text

                # OPTIMIZATION: Check and send typing indicator IMMEDIATELY for user messages
                # This happens before any database operations to minimize delay
                typing_indicator_sent = False
                print(f"[DEBUG] Checking typing indicator - sender: {sender}")
                if sender == 'user':
                    # Quick check for typing indicator setting (single DB query, cached)
                    widget_settings = widget_settings_service.get_widget_settings(db, agent_id)
                    print(f"[DEBUG] Widget settings: {widget_settings}")
                    if widget_settings:
                        print(f"[DEBUG] Widget settings exists, typing_indicator_enabled: {widget_settings.typing_indicator_enabled}")
                    if widget_settings and widget_settings.typing_indicator_enabled:
                        # Send typing indicator IMMEDIATELY before any other processing
                        await manager.broadcast_to_session(
                            str(session_id),
                            json.dumps({"message_type": "typing", "is_typing": True, "sender": "agent"}),
                            "agent"
                        )
                        typing_indicator_sent = True
                        print(f"[websocket_conversations] ⚡ Typing indicator ON (immediate) for session: {session_id}")
                    else:
                        print(f"[DEBUG] Typing indicator NOT sent - widget_settings: {widget_settings}, enabled: {widget_settings.typing_indicator_enabled if widget_settings else 'N/A'}")

                # Now, create and broadcast the chat message
                chat_message = schemas_chat_message.ChatMessageCreate(message=message_for_storage, message_type=message_data.get('message_type', 'message'))
                db_message = chat_service.create_chat_message(db, chat_message, agent_id, session_id, company_id, sender, assignee_id=None, attachments=attachments if attachments else None)

                # Enrich message with assignee name for broadcast (if message has assignee from database)
                message_dict = schemas_chat_message.ChatMessage.model_validate(db_message).model_dump(mode='json')
                if db_message.assignee_id:
                    assignee_user = db.query(models_user.User).filter(models_user.User.id == db_message.assignee_id).first()
                    if assignee_user:
                        # Build full name from first_name and last_name
                        name_parts = []
                        if assignee_user.first_name:
                            name_parts.append(assignee_user.first_name)
                        if assignee_user.last_name:
                            name_parts.append(assignee_user.last_name)
                        message_dict['assignee_name'] = ' '.join(name_parts) if name_parts else assignee_user.email

                # Include attachments in broadcast for preview display
                if attachments:
                    message_dict['attachments'] = attachments

                await manager.broadcast_to_session(str(session_id), json.dumps(message_dict), sender)

                if sender == 'user':
                    # The session is already guaranteed to exist, so we can proceed
                    print(f"DEBUG [Websocket Loop]: Checking session status. ID: {session.id}, Status: '{session.status}', Workflow ID: {session.workflow_id}")

                    # Check if AI is enabled for this session
                    if not session.is_ai_enabled:
                        print(f"AI is disabled for session {session.conversation_id}. No response will be generated.")
                        continue

                    execution_result = None
                    try:
                        # Check if a workflow is paused (indicated by next_step_id being set)
                        if session.next_step_id and session.workflow_id:
                            # A workflow is already in progress, so we resume it.
                            workflow = workflow_service.get_workflow(db, session.workflow_id, company_id)
                            if workflow:
                                 print(f"[websocket_conversations] 🚀 PUBLIC: Resuming workflow with {len(attachments)} attachment(s), option_key={option_key}")
                                 execution_result = await workflow_exec_service.execute_workflow(
                                    user_message=message_for_storage, company_id=company_id, workflow=workflow, conversation_id=session_id, attachments=attachments, option_key=option_key
                                )
                        else:
                            # No workflow is in progress, so we find a new one.
                            # Priority: 1) Triggers, 2) LLM decision, 3) Similarity search
                            print(f"[websocket_conversations] Looking for workflow - company_id={company_id}")

                            # 1. Try trigger-based workflow finding first
                            try:
                                workflow = await workflow_trigger_service.find_workflow_for_channel_message(
                                    db=db,
                                    channel=TriggerChannel.WEBSOCKET,
                                    company_id=company_id,
                                    message=message_for_storage,
                                    session_data={"session_id": session_id, "agent_id": agent_id}
                                )
                                print(f"[websocket_conversations] Trigger service returned: {workflow.name if workflow else None}")
                            except Exception as trigger_error:
                                print(f"[websocket_conversations] ERROR in trigger service: {trigger_error}")
                                import traceback
                                traceback.print_exc()
                                workflow = None

                            if workflow:
                                print(f"[websocket_conversations] 🚀 PUBLIC: Starting workflow from trigger with {len(attachments)} attachment(s)")
                                execution_result = await workflow_exec_service.execute_workflow(
                                    user_message=message_for_storage, company_id=company_id, workflow=workflow, conversation_id=session_id, attachments=attachments, option_key=option_key
                                )

                        # If no trigger-based workflow, try LLM-based routing (2nd priority)
                        if not execution_result:
                            agent_response = await agent_execution_service.generate_agent_response(db, agent_id, session.id, session_id, company_id, message_for_storage)
                            if agent_response:
                                # Check if LLM decided to trigger a workflow (context-aware routing)
                                if isinstance(agent_response, dict) and agent_response.get("type") == "workflow_trigger":
                                    workflow_id = agent_response.get("workflow_id")
                                    print(f"[websocket_conversations] PUBLIC: LLM triggered workflow {workflow_id}")
                                    workflow = workflow_service.get_workflow(db, workflow_id, company_id)
                                    if workflow:
                                        execution_result = await workflow_exec_service.execute_workflow(
                                            user_message=message_for_storage,
                                            conversation_id=session_id,
                                            company_id=company_id,
                                            workflow=workflow,
                                            attachments=attachments,
                                            option_key=option_key
                                        )
                                    else:
                                        print(f"[websocket_conversations] Workflow {workflow_id} not found")

                                # Handle handoff response (LLM failed)
                                elif isinstance(agent_response, dict) and agent_response.get("type") == "handoff":
                                    reason = agent_response.get("reason", "AI routing unavailable")
                                    print(f"[websocket_conversations] PUBLIC: LLM failed, initiating handoff: {reason}")
                                    error_msg = "I'm experiencing some technical difficulties. Let me connect you with a human agent who can help."
                                    agent_response_text = error_msg
                                    call_initiated = False
                                    # TODO: Trigger actual handoff to human agent here
                                # Handle dict response (with call info) or string response
                                elif isinstance(agent_response, dict):
                                    agent_response_text = agent_response.get('text', '')
                                    call_initiated = agent_response.get('call_initiated', False)
                                else:
                                    agent_response_text = agent_response
                                    call_initiated = False

                                # 3. Last fallback: similarity search (only if LLM didn't trigger workflow)
                                if not execution_result and not (isinstance(agent_response, dict) and agent_response.get("type") in ["workflow_trigger", "handoff"]):
                                    print(f"[websocket_conversations] PUBLIC: Trying similarity search as last fallback")
                                    similar_workflow = workflow_service.find_similar_workflow(db, company_id=company_id, query=message_for_storage, agent_id=agent_id)
                                    if similar_workflow:
                                        print(f"[websocket_conversations] 🚀 PUBLIC: Found similar workflow '{similar_workflow.name}'")
                                        execution_result = await workflow_exec_service.execute_workflow(
                                            user_message=message_for_storage, company_id=company_id, workflow=similar_workflow, conversation_id=session_id, attachments=attachments, option_key=option_key
                                        )

                                # Only save/broadcast LLM response if no workflow was executed
                                if not execution_result and not (isinstance(agent_response, dict) and agent_response.get("type") == "workflow_trigger"):
                                    agent_message = schemas_chat_message.ChatMessageCreate(message=str(agent_response_text), message_type="message")
                                    db_agent_message = chat_service.create_chat_message(db, agent_message, agent_id, session_id, company_id, "agent", assignee_id=None)

                                    # Build message dict for broadcast
                                    message_dict = schemas_chat_message.ChatMessage.model_validate(db_agent_message).model_dump(mode='json')

                                    # Add call info to broadcast if call was initiated
                                    if call_initiated and isinstance(agent_response, dict):
                                        message_dict.update({
                                            'call_initiated': True,
                                            'agent_name': agent_response.get('agent_name'),
                                            'room_name': agent_response.get('room_name'),
                                            'livekit_url': agent_response.get('livekit_url'),
                                            'user_token': agent_response.get('user_token')
                                        })

                                    await manager.broadcast_to_session(str(session_id), json.dumps(message_dict), "agent")

                                    # Generate TTS for chat_and_voice mode
                                    # Track if TTS was already sent to avoid duplicate audio
                                    tts_already_sent = False
                                    if widget_settings and widget_settings.communication_mode == 'chat_and_voice':
                                        try:
                                            # Get agent's voice settings
                                            tts_provider = agent.tts_provider or 'voice_engine'
                                            voice_id = agent.voice_id or 'default'

                                            # Get OpenAI API key if needed
                                            openai_api_key = None
                                            openai_credential = credential_service.get_credential_by_service_name(db, 'openai', company_id)
                                            if openai_credential:
                                                try:
                                                    openai_api_key = credential_service.get_decrypted_credential(db, openai_credential.id, company_id)
                                                except Exception as e:
                                                    print(f"[TTS] Failed to get OpenAI key: {e}")

                                            tts_service = TTSService(openai_api_key=openai_api_key)
                                            audio_stream = tts_service.text_to_speech_stream(agent_response_text, voice_id, tts_provider)
                                            async for audio_chunk in audio_stream:
                                                await manager.broadcast_bytes_to_session(str(session_id), audio_chunk)
                                            await tts_service.close()
                                            # Send audio_end marker
                                            await manager.broadcast_to_session(str(session_id), json.dumps({"type": "audio_end"}), "agent")
                                            tts_already_sent = True
                                            print(f"[websocket_conversations] TTS audio sent for chat_and_voice mode in session: {session_id}")
                                        except Exception as tts_error:
                                            print(f"[websocket_conversations] TTS error in chat_and_voice mode: {tts_error}")
                    finally:
                        # Turn off typing indicator after AI processing completes
                        if typing_indicator_sent:
                            await manager.broadcast_to_session(
                                str(session_id),
                                json.dumps({"message_type": "typing", "is_typing": False, "sender": "agent"}),
                                "agent"
                            )
                            print(f"[websocket_conversations] Typing indicator OFF for session: {session_id}")

                    if not execution_result:
                        continue

                    # Track if TTS was already sent (from LLM response above)
                    # This prevents duplicate TTS when workflow also generates a response
                    tts_already_sent = locals().get('tts_already_sent', False)

                    # Handle execution result
                    if execution_result.get("status") == "completed":
                        agent_response_text = execution_result.get("response", "Workflow finished.")
                        agent_message = schemas_chat_message.ChatMessageCreate(message=str(agent_response_text), message_type="message")
                        db_agent_message = chat_service.create_chat_message(db, agent_message, agent_id, session_id, company_id, "agent", assignee_id=None)
                        await manager.broadcast_to_session(str(session_id), schemas_chat_message.ChatMessage.model_validate(db_agent_message).model_dump_json(), "agent")

                        # Generate TTS for chat_and_voice mode (workflow completed)
                        # Skip if TTS was already sent for the LLM response
                        if widget_settings and widget_settings.communication_mode == 'chat_and_voice' and not tts_already_sent:
                            try:
                                tts_provider = agent.tts_provider or 'voice_engine'
                                voice_id = agent.voice_id or 'default'
                                openai_api_key = None
                                openai_credential = credential_service.get_credential_by_service_name(db, 'openai', company_id)
                                if openai_credential:
                                    try:
                                        openai_api_key = credential_service.get_decrypted_credential(db, openai_credential.id, company_id)
                                    except Exception:
                                        pass
                                tts_service = TTSService(openai_api_key=openai_api_key)
                                audio_stream = tts_service.text_to_speech_stream(agent_response_text, voice_id, tts_provider)
                                async for audio_chunk in audio_stream:
                                    await manager.broadcast_bytes_to_session(str(session_id), audio_chunk)
                                await tts_service.close()
                                # Send audio_end marker
                                await manager.broadcast_to_session(str(session_id), json.dumps({"type": "audio_end"}), "agent")
                                tts_already_sent = True
                                print(f"[websocket_conversations] TTS audio sent for workflow completion in session: {session_id}")
                            except Exception as tts_error:
                                print(f"[websocket_conversations] TTS error: {tts_error}")

                    elif execution_result.get("status") == "paused_for_prompt":
                        prompt_data = execution_result.get("prompt", {})
                        prompt_text = prompt_data.get("text", "")
                        options = prompt_data.get("options", [])
                        allow_text_input = prompt_data.get("allow_text_input", False)

                        # Save prompt message to database (including options for history)
                        prompt_chat_message = schemas_chat_message.ChatMessageCreate(
                            message=prompt_text,
                            message_type="prompt"
                        )
                        db_prompt_message = chat_service.create_chat_message(
                            db, prompt_chat_message, agent_id, session_id, company_id, "agent", assignee_id=None, options=options
                        )

                        # Broadcast message with options and allow_text_input for frontend display
                        message_dict = schemas_chat_message.ChatMessage.model_validate(db_prompt_message).model_dump(mode='json')
                        message_dict['options'] = options
                        message_dict['allow_text_input'] = allow_text_input
                        await manager.broadcast_to_session(str(session_id), json.dumps(message_dict), "agent")
                        print(f"[websocket_conversations] PUBLIC: Saved and broadcasted prompt to session: {session_id}")

                        # Generate TTS for chat_and_voice mode (prompt)
                        if widget_settings and widget_settings.communication_mode == 'chat_and_voice' and prompt_text:
                            try:
                                tts_provider = agent.tts_provider or 'voice_engine'
                                voice_id = agent.voice_id or 'default'
                                # Format prompt with options for voice
                                tts_text = prompt_text
                                if options:
                                    # Handle options as list of dicts, list of strings, or comma-separated string
                                    if isinstance(options, str):
                                        option_names = [o.strip() for o in options.split(',')]
                                    elif isinstance(options, list) and options:
                                        if isinstance(options[0], dict):
                                            option_names = [o.get('label', o.get('value', str(o))) for o in options]
                                        else:
                                            option_names = [str(o) for o in options]
                                    else:
                                        option_names = []
                                    if option_names:
                                        tts_text += " Your options are: " + ", ".join(option_names)
                                openai_api_key = None
                                openai_credential = credential_service.get_credential_by_service_name(db, 'openai', company_id)
                                if openai_credential:
                                    try:
                                        openai_api_key = credential_service.get_decrypted_credential(db, openai_credential.id, company_id)
                                    except Exception:
                                        pass
                                tts_service = TTSService(openai_api_key=openai_api_key)
                                audio_stream = tts_service.text_to_speech_stream(tts_text, voice_id, tts_provider)
                                async for audio_chunk in audio_stream:
                                    await manager.broadcast_bytes_to_session(str(session_id), audio_chunk)
                                await tts_service.close()
                                # Send audio_end marker
                                await manager.broadcast_to_session(str(session_id), json.dumps({"type": "audio_end"}), "agent")
                                print(f"[websocket_conversations] TTS audio sent for prompt in session: {session_id}")
                            except Exception as tts_error:
                                print(f"[websocket_conversations] TTS error: {tts_error}")

                    elif execution_result.get("status") == "paused_for_form":
                        form_data = execution_result.get("form", {})
                        form_title = form_data.get("title", "")
                        form_fields = form_data.get("fields", [])
                        form_message = {
                            "message": form_title,
                            "message_type": "form",
                            "fields": form_fields,
                            "sender": "agent",
                        }
                        await manager.broadcast_to_session(str(session_id), json.dumps(form_message), "agent")

                        # Generate TTS for chat_and_voice mode (form)
                        if widget_settings and widget_settings.communication_mode == 'chat_and_voice' and form_title:
                            try:
                                tts_provider = agent.tts_provider or 'voice_engine'
                                voice_id = agent.voice_id or 'default'
                                # Format form info for voice
                                tts_text = form_title
                                if form_fields:
                                    field_names = [f.get("label", f.get("name", "")) for f in form_fields if f.get("label") or f.get("name")]
                                    if field_names:
                                        tts_text += " Please provide: " + ", ".join(field_names)
                                openai_api_key = None
                                openai_credential = credential_service.get_credential_by_service_name(db, 'openai', company_id)
                                if openai_credential:
                                    try:
                                        openai_api_key = credential_service.get_decrypted_credential(db, openai_credential.id, company_id)
                                    except Exception:
                                        pass
                                tts_service = TTSService(openai_api_key=openai_api_key)
                                audio_stream = tts_service.text_to_speech_stream(tts_text, voice_id, tts_provider)
                                async for audio_chunk in audio_stream:
                                    await manager.broadcast_bytes_to_session(str(session_id), audio_chunk)
                                await tts_service.close()
                                # Send audio_end marker
                                await manager.broadcast_to_session(str(session_id), json.dumps({"type": "audio_end"}), "agent")
                                print(f"[websocket_conversations] TTS audio sent for form in session: {session_id}")
                            except Exception as tts_error:
                                print(f"[websocket_conversations] TTS error: {tts_error}")

                    elif execution_result.get("status") == "paused_for_input":
                        # Broadcast input constraint to widget so it knows what input type is expected
                        expected_input_type = execution_result.get("expected_input_type", "any")
                        input_constraint_message = {
                            "message_type": "input_constraint",
                            "expected_input_type": expected_input_type,
                            "sender": "agent"
                        }
                        await manager.broadcast_to_session(str(session_id), json.dumps(input_constraint_message), "agent")
                        print(f"[websocket_conversations] Broadcasted input constraint ({expected_input_type}) to session: {session_id}")

    except WebSocketDisconnect:
        manager.disconnect(websocket, session_id)
        print(f"Client in session #{session_id} disconnected")

        # Update session status to inactive when user disconnects
        # Only if there are no more user connections
        if user_type == "user" and not manager.has_user_connection(session_id):
            with get_db_session() as db:
                await conversation_session_service.update_session_connection_status(db, session_id, is_connected=False)

    finally:
        # Cancel heartbeat task
        if heartbeat_task:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass

