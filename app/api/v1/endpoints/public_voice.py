
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query
from sqlalchemy.orm import Session
from app.services import chat_service, workflow_service, workflow_trigger_service, agent_execution_service, messaging_service, integration_service, company_service, agent_service, contact_service, conversation_session_service, credential_service, widget_settings_service, agent_assignment_service
from app.services.workflow_execution_service import WorkflowExecutionService
from app.models.workflow_trigger import TriggerChannel
from app.schemas import chat_message as schemas_chat_message
import json
from typing import List, Dict, Any
from app.core.dependencies import get_current_user_from_ws, get_db
from app.models import user as models_user, conversation_session as models_conversation_session
from app.services.connection_manager import manager
from app.services.stt_service import STTService, GroqSTTService, OpenAISTTService
from app.services.tts_service import TTSService
from fastapi import UploadFile
import io
import logging
from types import SimpleNamespace
import aiohttp

router = APIRouter()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import asyncio

@router.websocket("/public/voice/{company_id}/{agent_id}/{session_id}")
async def public_voice_websocket_endpoint(
    websocket: WebSocket,
    company_id: int,
    agent_id: int,
    session_id: str,
    user_type: str = Query(...),
    voice_id: str = Query("21m00Tcm4TlvDq8ikWAM"), # Default voice
    stt_provider_param: str = Query(None, alias="stt_provider"), # Optional override from frontend
    db: Session = Depends(get_db)
):
    logger.info(f"New public voice connection for session {session_id}")
    # Fetch agent details to get the configured voice
    agent = agent_service.get_agent(db, agent_id, company_id)
    if not agent:
        logger.error(f"Agent not found for agent_id: {agent_id}, company_id: {company_id}")
        await websocket.close(code=1008)
        return
        
    logger.info(f"Agent found: {agent.name}")
    logger.info(f"Agent credential object: {agent.credential}")

    final_voice_id = voice_id
    tts_provider = 'openai'  # Default to OpenAI (vault credentials available)
    stt_provider = 'openai'  # Default provider
    if agent:
        if agent.voice_id:
            final_voice_id = agent.voice_id
        if agent.tts_provider:
            tts_provider = agent.tts_provider
        if agent.stt_provider:
            stt_provider = agent.stt_provider

    # Allow frontend to override STT provider via query parameter
    if stt_provider_param:
        stt_provider = stt_provider_param
        logger.info(f"STT provider overridden by frontend to: {stt_provider}")

    # voice_engine is not deployed — fall back to openai
    if tts_provider == 'voice_engine':
        logger.info("TTS provider 'voice_engine' is not available, falling back to 'openai'")
        tts_provider = 'openai'

    logger.info(f"STT Provider: {stt_provider}, TTS Provider: {tts_provider}, Voice ID: {final_voice_id}")

    await manager.connect(websocket, session_id, user_type, connection_type="voice")
    logger.info(f"WebSocket connected to manager for session {session_id} (voice connection)")

    # Get OpenAI API key from vault if available (used for both STT and TTS)
    openai_api_key = None
    # Look up OpenAI credential by service name for the company
    openai_credential = credential_service.get_credential_by_service_name(db, 'openai', company_id)
    if openai_credential:
        logger.info("Found OpenAI credential in vault for company.")
        try:
            decrypted_key = credential_service.get_decrypted_credential(db, openai_credential.id, company_id)
            if decrypted_key and isinstance(decrypted_key, str):
                openai_api_key = decrypted_key
                logger.info("Using OpenAI API key from Vault for STT/TTS.")
            else:
                logger.warning("OpenAI credential found but the decrypted key is not a valid string.")
        except Exception as e:
            logger.error(f"Failed to decrypt OpenAI credential: {e}")

    stt_service = None
    if stt_provider == "groq":
        groq_api_key = None
        # Look up Groq credential by service name for the company
        groq_credential = credential_service.get_credential_by_service_name(db, 'groq', company_id)
        if groq_credential:
            logger.info("Found Groq credential in vault for company.")
            try:
                # The credential service returns the raw decrypted key for simple services like Groq
                decrypted_key = credential_service.get_decrypted_credential(db, groq_credential.id, company_id)
                if decrypted_key and isinstance(decrypted_key, str):
                    groq_api_key = decrypted_key
                    logger.info("Using Groq API key from Vault.")
                else:
                    logger.warning("Groq credential found but the decrypted key is not a valid string.")
            except Exception as e:
                logger.error(f"Failed to decrypt Groq credential: {e}")

        if not groq_api_key:
            logger.warning("Groq API key not found in vault. Falling back to environment variable.")

        try:
            stt_service = GroqSTTService(api_key=groq_api_key) if groq_api_key else GroqSTTService()
        except ValueError as e:
            # Groq API key not available - try to fall back to OpenAI if available
            if openai_api_key:
                logger.warning(f"Groq STT failed ({e}). Falling back to OpenAI STT.")
                stt_service = OpenAISTTService(api_key=openai_api_key)
                stt_provider = "openai"  # Update provider for buffer processing
            else:
                logger.error(f"Groq STT failed and no OpenAI fallback available: {e}")
                await websocket.send_text('{"error": "STT service not configured. Please set up Groq or OpenAI credentials."}')
                await websocket.close(code=1008)
                return

    elif stt_provider == "openai":
        # Use the OpenAI key retrieved earlier (from vault or env)
        if openai_api_key:
            stt_service = OpenAISTTService(api_key=openai_api_key)
        else:
            logger.warning("OpenAI API key not found in vault. Falling back to environment variable.")
            stt_service = OpenAISTTService()

    elif stt_provider == "deepgram":
        deepgram_api_key = None

        # Look up Deepgram credential by service name for the company
        deepgram_credential = credential_service.get_credential_by_service_name(db, 'deepgram', company_id)
        if deepgram_credential:
            logger.info("Found Deepgram credential in vault for company.")
            try:
                # Get decrypted key from vault
                decrypted_key = credential_service.get_decrypted_credential(db, deepgram_credential.id, company_id)
                if decrypted_key and isinstance(decrypted_key, str):
                    deepgram_api_key = decrypted_key
                    logger.info("Using Deepgram API key from Vault.")
                else:
                    logger.warning("Deepgram credential found but the decrypted key is not a valid string.")
            except Exception as e:
                logger.error(f"Failed to decrypt Deepgram credential: {e}")

        if not deepgram_api_key:
            logger.warning("Deepgram API key not found in vault. Falling back to environment variable.")

        # Pass api_key to STTService (will use env var if None)
        stt_service = STTService(websocket, api_key=deepgram_api_key)
    else:
        logger.error(f"Unsupported STT provider: {stt_provider}")
        await websocket.close(code=1008)
        return

    tts_service = TTSService(openai_api_key=openai_api_key)
    logger.info(f"STT and TTS services initialized (OpenAI key from vault: {bool(openai_api_key)})")
    
    transcript_queue = asyncio.Queue()
    audio_buffer = bytearray()
    last_audio_time = None

    # Handoff state management
    handoff_requested = False
    waiting_for_agent = False
    handoff_data = None

    async def handle_transcription():
        logger.info("Starting transcription handler")
        if stt_provider == "deepgram":
            if await stt_service.connect():
                while True:
                    try:
                        # Use receive() to handle all message types (text, binary, close, etc.)
                        msg = await stt_service.deepgram_ws.receive()

                        # Handle close frames gracefully
                        if msg.type == aiohttp.WSMsgType.CLOSED:
                            logger.info("Deepgram WebSocket closed normally")
                            break
                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            logger.error(f"Deepgram WebSocket error: {msg}")
                            break
                        elif msg.type == aiohttp.WSMsgType.TEXT:
                            # Parse JSON message
                            message = json.loads(msg.data)
                            if message.get("type") == "Results":
                                transcript = message["channel"]["alternatives"][0]["transcript"]
                                if transcript and message.get("is_final", False):
                                    await transcript_queue.put(transcript)
                        # Ignore other message types (binary, ping, pong, etc.)
                    except Exception as e:
                        logger.error(f"Error receiving from STT service: {e}")
                        break
            await stt_service.close()
        logger.info("Transcription handler finished")
        # Groq does not use a persistent connection for transcription

    async def handle_audio_from_client():
        nonlocal audio_buffer, last_audio_time, handoff_requested, handoff_data
        logger.info("Starting audio from client handler")
        try:
            while True:
                # Receive both text and binary messages
                message = await websocket.receive()

                # Handle text messages (handoff requests)
                if "text" in message:
                    try:
                        data = json.loads(message["text"])
                        if data.get("type") == "request_handoff":
                            logger.info(f"Manual handoff requested: {data}")
                            handoff_requested = True
                            handoff_data = {
                                "reason": data.get("reason", "customer_request"),
                                "summary": data.get("summary", "User requested human agent via button"),
                                "priority": data.get("priority", "normal"),
                                "pool": data.get("pool", "support")
                            }
                            # Put a special marker in the queue to trigger handoff processing
                            await transcript_queue.put("__HANDOFF_REQUEST__")
                    except json.JSONDecodeError:
                        logger.error("Invalid JSON in text message")
                    except Exception as e:
                        logger.error(f"Error processing text message: {e}")

                # Handle binary messages (audio chunks)
                elif "bytes" in message:
                    audio_chunk = message["bytes"]
                    logger.info(f"[AUDIO] Received audio chunk: {len(audio_chunk)} bytes, provider: {stt_provider}")
                    last_audio_time = asyncio.get_event_loop().time()
                    if stt_provider == "deepgram":
                        if stt_service.deepgram_ws and not stt_service.deepgram_ws.closed:
                            await stt_service.deepgram_ws.send_bytes(audio_chunk)
                        else:
                            break
                    elif stt_provider in ("groq", "openai"):
                        audio_buffer.extend(audio_chunk)
                        logger.info(f"[AUDIO] Buffer size now: {len(audio_buffer)} bytes")

        except WebSocketDisconnect:
            logger.info("Client disconnected from audio stream")
        except Exception as e:
            logger.error(f"Error receiving from client: {e}")
        logger.info("Audio from client handler finished")

    async def process_batch_stt_buffer():
        """Process buffered audio for batch STT providers (Groq, OpenAI)."""
        nonlocal audio_buffer, last_audio_time
        logger.info(f"Starting batch STT buffer processor for {stt_provider}")
        while True:
            await asyncio.sleep(0.5) # Check every 500ms
            current_time = asyncio.get_event_loop().time()
            if audio_buffer:
                time_since_last = current_time - last_audio_time if last_audio_time else 0
                logger.debug(f"[{stt_provider.upper()}] Buffer check: {len(audio_buffer)} bytes, time since last audio: {time_since_last:.2f}s")
            if audio_buffer and last_audio_time and (current_time - last_audio_time > 1.0): # 1 second pause
                logger.info(f"[{stt_provider.upper()}] Pause detected, processing audio buffer of {len(audio_buffer)} bytes")
                try:
                    # Create a duck-typed object that mimics UploadFile for the STT service
                    async def read_audio_data():
                        return bytes(audio_buffer)

                    # Browser sends audio/webm format from MediaRecorder
                    mock_audio_file = SimpleNamespace(
                        filename="audio.webm",
                        content_type="audio/webm",
                        read=read_audio_data
                    )

                    logger.info(f"[{stt_provider.upper()}] Calling transcription API...")
                    transcript_result = await stt_service.transcribe(mock_audio_file)
                    logger.info(f"[{stt_provider.upper()}] API response: {transcript_result}")
                    transcript = transcript_result.get("text")
                    if transcript:
                        logger.info(f"[{stt_provider.upper()}] Transcription result: {transcript}")
                        await transcript_queue.put(transcript)
                    else:
                        logger.warning(f"[{stt_provider.upper()}] Empty or no transcript in response")
                except Exception as e:
                    logger.error(f"[{stt_provider.upper()}] Error during transcription: {e}")
                    import traceback
                    traceback.print_exc()
                finally:
                    audio_buffer.clear()
                    last_audio_time = None
        logger.info(f"Batch STT buffer processor finished for {stt_provider}")

    async def handle_waiting_for_agent():
        """Sends periodic waiting messages while customer waits for agent"""
        nonlocal waiting_for_agent
        logger.info("Starting waiting for agent handler")
        wait_messages = [
            "Please hold while I connect you to an agent...",
            "Thank you for your patience. An agent will be with you shortly.",
            "Still connecting you to the next available agent..."
        ]
        message_index = 0

        while waiting_for_agent:
            await asyncio.sleep(10)  # Send message every 10 seconds
            if not waiting_for_agent:  # Check again after sleep
                break

            message_text = wait_messages[message_index % len(wait_messages)]
            message_index += 1

            # Send waiting message via WebSocket
            await manager.broadcast_to_session(
                session_id,
                json.dumps({"message_type": "waiting", "message": message_text}),
                "agent"
            )

            # Optionally convert to speech
            if tts_provider:
                try:
                    audio_stream = tts_service.text_to_speech_stream(message_text, final_voice_id, tts_provider)
                    async for audio_chunk in audio_stream:
                        await websocket.send_bytes(audio_chunk)
                except Exception as e:
                    logger.error(f"Error sending wait message audio: {e}")

        logger.info("Waiting for agent handler finished")

    async def transition_to_livekit(agent_user_id: int):
        """Transitions the voice session to LiveKit for human agent connection"""
        nonlocal waiting_for_agent
        logger.info(f"[LIVEKIT TRANSITION] Starting transition for session {session_id} with agent {agent_user_id}")

        try:
            # Stop waiting state
            waiting_for_agent = False

            # Generate LiveKit tokens
            from app.api.v1.endpoints.video_calls import get_livekit_token
            from app.core.config import settings

            room_name = f"handoff_{session_id}_{agent_user_id}"

            # Token for user
            user_token = get_livekit_token(room_name, f"User-{session_id}")

            # Send transition message to user
            transition_message = {
                "type": "transition_to_livekit",
                "room_name": room_name,
                "livekit_token": user_token,
                "livekit_url": settings.LIVEKIT_URL,
                "agent_id": agent_user_id
            }

            await websocket.send_text(json.dumps(transition_message))
            logger.info(f"[LIVEKIT TRANSITION] Sent transition message to user")

            # Give client time to process before closing WebSocket
            await asyncio.sleep(2)

        except Exception as e:
            logger.error(f"[LIVEKIT TRANSITION] Error: {e}")


    transcription_task = asyncio.create_task(handle_transcription())
    client_audio_task = asyncio.create_task(handle_audio_from_client())
    
    batch_stt_task = None
    if stt_provider in ("groq", "openai"):
        batch_stt_task = asyncio.create_task(process_batch_stt_buffer())

    logger.info("All tasks created")

    waiting_task = None

    try:
        while True:
            transcript = await transcript_queue.get()
            logger.info(f"Processing transcript: {transcript}")

            # Check if this is a manual handoff request
            if transcript == "__HANDOFF_REQUEST__":
                logger.info("[HANDOFF] Manual handoff request detected")
                if handoff_data:
                    # Process the handoff request
                    handoff_result = await agent_assignment_service.request_handoff(
                        db=db,
                        session_id=session_id,
                        reason=handoff_data["reason"],
                        pool_name=handoff_data["pool"],
                        priority=handoff_data["priority"]
                    )

                    logger.info(f"[HANDOFF] Result: {handoff_result}")

                    if handoff_result["status"] == "agent_found":
                        # Agent available - start waiting state
                        waiting_for_agent = True
                        waiting_task = asyncio.create_task(handle_waiting_for_agent())

                        # Notify assigned agent via WebSocket
                        await manager.broadcast(
                            json.dumps({
                                "type": "handoff_request",
                                "session_id": session_id,
                                "agent_id": agent_id,
                                "summary": handoff_data["summary"],
                                "reason": handoff_data["reason"],
                                "assigned_agent_id": handoff_result["agent_id"]
                            }),
                            str(company_id)  # Broadcast to company channel
                        )
                    elif handoff_result["status"] == "no_agents_available":
                        # No agents - collect callback info
                        logger.info("[HANDOFF] No agents available, collecting callback info")
                        # TODO: Implement callback collection via LLM

                transcript_queue.task_done()
                continue

            if transcript:
                # Skip AI processing if waiting for agent
                if waiting_for_agent:
                    logger.info("[HANDOFF] Skipping AI processing - waiting for agent")
                    transcript_queue.task_done()
                    continue

                # 1. Save and broadcast the user's transcribed message
                user_message = schemas_chat_message.ChatMessageCreate(message=transcript, message_type='message')
                db_user_message = chat_service.create_chat_message(db, user_message, agent_id, session_id, company_id, "user")
                await manager.broadcast_to_session(session_id, schemas_chat_message.ChatMessage.model_validate(db_user_message).model_dump_json(), "user")

                # Check and send typing indicator before agent processing
                typing_indicator_sent = False
                widget_settings = widget_settings_service.get_widget_settings(db, agent_id)
                if widget_settings and widget_settings.typing_indicator_enabled:
                    await manager.broadcast_to_session(
                        session_id,
                        json.dumps({"message_type": "typing", "is_typing": True, "sender": "agent"}),
                        "agent"
                    )
                    typing_indicator_sent = True
                    logger.info(f"[Voice] Typing indicator ON for session: {session_id}")

                try:
                    # 2. Check for workflow triggers first
                    agent_response_text = None
                    workflow_executed = False

                    try:
                        workflow = await workflow_trigger_service.find_workflow_for_channel_message(
                            db=db,
                            channel=TriggerChannel.WEBSOCKET,
                            company_id=company_id,
                            message=transcript,
                            session_data={"session_id": session_id, "agent_id": agent_id}
                        )
                        logger.info(f"[Voice] Trigger service returned: {workflow.name if workflow else None}")

                        if workflow:
                            # Execute workflow instead of direct agent response
                            workflow_exec_service = WorkflowExecutionService(db)
                            execution_result = await workflow_exec_service.execute_workflow(
                                user_message=transcript,
                                company_id=company_id,
                                workflow=workflow,
                                conversation_id=session_id
                            )
                            if execution_result:
                                status = execution_result.get("status")
                                logger.info(f"[Voice] Workflow execution result status: {status}")

                                if status == "completed":
                                    agent_response_text = execution_result.get("response", "Workflow completed.")
                                    workflow_executed = True
                                    logger.info(f"[Voice] Workflow completed: {agent_response_text}")

                                elif status == "paused_for_prompt":
                                    # Extract prompt text and options from the result
                                    prompt_data = execution_result.get("prompt", {})
                                    prompt_text = prompt_data.get("text", "")
                                    options = prompt_data.get("options", [])
                                    # Format response with options for voice
                                    # Handle options as list of dicts, list of strings, or comma-separated string
                                    option_names = []
                                    if options:
                                        if isinstance(options, str):
                                            option_names = [o.strip() for o in options.split(',')]
                                        elif isinstance(options, list) and options:
                                            if isinstance(options[0], dict):
                                                option_names = [o.get('label', o.get('value', str(o))) for o in options]
                                            else:
                                                option_names = [str(o) for o in options]

                                    if option_names:
                                        agent_response_text = f"{prompt_text} Your options are: {', '.join(option_names)}"
                                    else:
                                        agent_response_text = prompt_text
                                    workflow_executed = True
                                    logger.info(f"[Voice] Workflow paused for prompt: {agent_response_text}")

                                elif status in ("paused_for_input", "paused_for_form"):
                                    agent_response_text = execution_result.get("response", "Please provide your input.")
                                    workflow_executed = True
                                    logger.info(f"[Voice] Workflow {status}: {agent_response_text}")
                    except Exception as trigger_error:
                        logger.error(f"[Voice] Error in trigger service: {trigger_error}")
                        import traceback
                        traceback.print_exc()

                    # 3. Fallback to agent response if no workflow executed
                    if not workflow_executed:
                        agent_response_text = await agent_execution_service.generate_agent_response(
                            db, agent_id, session_id, session_id, company_id, transcript
                        )
                    logger.info(f"LLM response: {agent_response_text}")

                    # 3. Save and broadcast the agent's text message
                    agent_message = schemas_chat_message.ChatMessageCreate(message=agent_response_text, message_type='message')
                    db_agent_message = chat_service.create_chat_message(db, agent_message, agent_id, session_id, company_id, "agent")
                    await manager.broadcast_to_session(session_id, schemas_chat_message.ChatMessage.model_validate(db_agent_message).model_dump_json(), "agent")

                    # 4. Convert the agent's response to speech and stream it
                    audio_stream = tts_service.text_to_speech_stream(agent_response_text, final_voice_id, tts_provider)
                    logger.info("Streaming TTS audio to client...")
                    async for audio_chunk in audio_stream:
                        await websocket.send_bytes(audio_chunk)
                finally:
                    # Turn off typing indicator after processing completes
                    if typing_indicator_sent:
                        await manager.broadcast_to_session(
                            session_id,
                            json.dumps({"message_type": "typing", "is_typing": False, "sender": "agent"}),
                            "agent"
                        )
                        logger.info(f"[Voice] Typing indicator OFF for session: {session_id}")

                transcript_queue.task_done()
            logger.info("Finished processing transcript")

    except WebSocketDisconnect:
        logger.info(f"Client in voice session #{session_id} disconnected")
    except Exception as e:
        logger.error(f"Error in main voice processing loop: {e}")
    finally:
        logger.info("Cleaning up tasks and resources")
        transcription_task.cancel()
        client_audio_task.cancel()
        if batch_stt_task:
            batch_stt_task.cancel()
        if waiting_task:
            waiting_task.cancel()
        await tts_service.close()
        manager.disconnect(websocket, session_id)
        logger.info(f"Cleaned up resources for voice session #{session_id}")
