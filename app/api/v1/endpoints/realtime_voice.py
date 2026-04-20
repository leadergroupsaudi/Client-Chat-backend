"""
OpenAI Realtime Voice WebSocket Endpoint.

Bridges browser WebSocket audio to the OpenAI Realtime API without LiveKit.

Endpoints
---------
POST /api/v1/ws/public/voice/realtime/start-session
    Create a new session, return a UUID session_id.

GET  /api/v1/ws/public/voice/realtime-token
    Create an ephemeral OpenAI Realtime token for WebRTC connections.

WS   /api/v1/ws/public/voice/realtime/{company_id}/{agent_id}/{session_id}
    Handle real-time voice streaming for an existing session.

Flow
----
    1. Browser (MediaRecorder WebM) --[binary]-> this endpoint
    2. This endpoint converts WebM -> PCM16@24kHz via PyAV
    3. PCM16 is streamed to OpenAI Realtime API over WebSocket
    4. OpenAI responds with PCM16 audio deltas + transcripts (+ tool calls)
    5. Tool calls are executed by Python handlers; results sent back to OpenAI
    6. PCM16 is wrapped in WAV and sent back to the browser
    7. Transcripts are saved to DB and broadcast to all session listeners

Routing:
    This router is mounted under the /ws prefix in app/api/v1/main.py, making
    the full HTTP paths:
        POST /api/v1/ws/public/voice/realtime/start-session
        GET  /api/v1/ws/public/voice/realtime-token
        WS   /api/v1/ws/public/voice/realtime/{company_id}/{agent_id}/{session_id}
"""

import asyncio
import json
import logging
import os
from typing import Optional

import httpx
from fastapi import APIRouter, Body, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.agents.realtime_voice_agent import RealtimeVoiceAgent, convert_webm_to_pcm16_24k
from app.core.dependencies import get_db
from app.core.prompt_builder import PromptBuilder
from app.core.session_manager import session_manager
from app.schemas import chat_message as schemas_chat_message
from app.services import agent_service, chat_service, credential_service
from app.services.connection_manager import manager
from app.services.voice_integration_service import (
    build_agent_tools_for_openai,
    build_openai_session_config,
    build_system_prompt_for_voice,
)
from app.tools import (
    CREATE_INCIDENT_SCHEMA,
    GENERATE_TICKET_SCHEMA,
    create_incident,
    generate_ticket,
)

logger = logging.getLogger(__name__)

router = APIRouter()

OPENAI_REALTIME_MODEL = os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime")
_VALID_VOICES = {"alloy", "echo", "shimmer", "ash", "ballad", "coral", "sage", "verse"}


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class StartSessionRequest(BaseModel):
    """Optional body for POST /start-session."""
    agent_id: Optional[int] = None
    company_id: Optional[int] = None
    # Pre-populate user_context from a form, JWT claims, etc.
    initial_context: Optional[dict] = None


class StartSessionResponse(BaseModel):
    """Response returned by POST /start-session."""
    session_id: str
    message: str = "Session created. Connect via WebSocket to begin."


# ---------------------------------------------------------------------------
# POST /start-session
# ---------------------------------------------------------------------------

@router.post("/public/voice/realtime/start-session", response_model=StartSessionResponse)
async def start_realtime_session(
    body: StartSessionRequest = Body(default_factory=StartSessionRequest),
) -> StartSessionResponse:
    """
    Create a new voice session and return its UUID.

    The caller should store the ``session_id`` and use it when opening the
    WebSocket at ``/api/v1/ws/public/voice/realtime/{company_id}/{agent_id}/{session_id}``.

    Request body (all fields optional)::

        {
          "agent_id": 1,
          "company_id": 42,
          "initial_context": {"caller_name": "Jane"}
        }

    Response::

        {"session_id": "d3f1a2b4-...", "message": "Session created."}
    """
    state = await session_manager.create_session(
        agent_id=body.agent_id,
        company_id=body.company_id,
        initial_context=body.initial_context or {},
    )
    logger.info(
        f"[StartSession] Created session={state.session_id} "
        f"agent={body.agent_id} company={body.company_id}"
    )
    return StartSessionResponse(session_id=state.session_id)


@router.get("/public/voice/realtime-token")
async def get_realtime_session_token(
    company_id: int = Query(...),
    agent_id: int = Query(...),
    db: Session = Depends(get_db),
) -> dict:
    """
    Create an OpenAI Realtime ephemeral session token for WebRTC connection.

    The browser uses this token to connect directly to OpenAI Realtime via WebRTC,
    bypassing the server audio relay entirely (lower latency, same experience as LiveKit).
    """
    agent = agent_service.get_agent(db, agent_id, company_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Resolve OpenAI API key
    openai_api_key: str | None = None
    try:
        cred = credential_service.get_credential_by_service_name(db, "openai", company_id)
        if cred:
            decrypted = credential_service.get_decrypted_credential(db, cred.id, company_id)
            if decrypted and isinstance(decrypted, str):
                openai_api_key = decrypted
    except Exception as exc:
        logger.warning(f"[RealtimeToken] Could not load OpenAI credential: {exc}")

    if not openai_api_key:
        openai_api_key = os.getenv("OPENAI_API_KEY", "")
    if not openai_api_key:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")

    greeting: str = agent.welcome_message or "Hello! How can I help you today?"

    # Build the full session config using the integration service.
    # This includes the agent's tools, voice-optimised system prompt, and VAD settings.
    session_payload = build_openai_session_config(
        agent,
        model=OPENAI_REALTIME_MODEL,
        token_ttl_seconds=600,
    )

    # Create ephemeral client secret via the OpenAI Realtime API.
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(
            "https://api.openai.com/v1/realtime/client_secrets",
            headers={
                "Authorization": f"Bearer {openai_api_key}",
                "Content-Type": "application/json",
            },
            json=session_payload,
        )

    if resp.status_code != 200:
        logger.error(f"[RealtimeToken] OpenAI error {resp.status_code}: {resp.text}")
        raise HTTPException(status_code=502, detail="Failed to create OpenAI Realtime session")

    session_data = resp.json()
    ephemeral_token: str = session_data["value"]

    # Expose tool names so the browser can log/display which tools are active.
    tool_names = [
        t["name"]
        for t in session_payload["session"].get("tools", [])
        if isinstance(t, dict) and t.get("name")
    ]

    logger.info(
        "[RealtimeToken] Created ephemeral token for agent=%s company=%s tools=%s",
        agent_id, company_id, tool_names,
    )
    return {
        "token": ephemeral_token,
        "model": OPENAI_REALTIME_MODEL,
        "greeting": greeting,
        "tools": tool_names,
    }


@router.websocket("/public/voice/realtime/{company_id}/{agent_id}/{session_id}")
async def realtime_voice_websocket(
    websocket: WebSocket,
    company_id: int,
    agent_id: int,
    session_id: str,
    user_type: str = Query(...),
    db: Session = Depends(get_db),
) -> None:
    """
    WebSocket endpoint for OpenAI Realtime voice sessions (no LiveKit).

    Query parameters:
        user_type: "user" or "agent" — used by the connection manager.
    """
    logger.info(f"[RealtimeWS] New session: company={company_id} agent={agent_id} session={session_id}")

    # ------------------------------------------------------------------
    # Load agent from DB
    # ------------------------------------------------------------------
    agent = agent_service.get_agent(db, agent_id, company_id)
    if not agent:
        logger.error(f"[RealtimeWS] Agent not found: agent_id={agent_id}, company_id={company_id}")
        await websocket.close(code=1008)
        return

    logger.info(f"[RealtimeWS] Agent loaded: {agent.name}")

    # ------------------------------------------------------------------
    # Resolve OpenAI API key (vault first, env var fallback)
    # ------------------------------------------------------------------
    openai_api_key: str | None = None
    try:
        cred = credential_service.get_credential_by_service_name(db, "openai", company_id)
        if cred:
            decrypted = credential_service.get_decrypted_credential(db, cred.id, company_id)
            if decrypted and isinstance(decrypted, str):
                openai_api_key = decrypted
                logger.info("[RealtimeWS] Using OpenAI API key from vault")
    except Exception as exc:
        logger.warning(f"[RealtimeWS] Could not load OpenAI credential from vault: {exc}")

    # ------------------------------------------------------------------
    # Build agent parameters from the DB agent record
    # ------------------------------------------------------------------
    system_prompt: str = (
        agent.prompt
        or "You are a helpful and friendly voice assistant. "
           "Keep your responses concise and natural for voice conversation."
    )
    greeting: str = agent.welcome_message or "Hello! How can I help you today?"

    # Voice model/voice can be customised via env vars; the DB agent record
    # stores TTS voice (OpenAI voice name) which we reuse here.
    voice: str | None = None
    valid_realtime_voices = {"alloy", "echo", "shimmer", "ash", "ballad", "coral", "sage", "verse"}
    if agent.voice_id and agent.voice_id.lower() in valid_realtime_voices:
        voice = agent.voice_id.lower()

    # ------------------------------------------------------------------
    # Register WebSocket with the connection manager
    # ------------------------------------------------------------------
    await manager.connect(websocket, session_id, user_type, connection_type="voice")
    logger.info(f"[RealtimeWS] Registered with connection manager (session={session_id})")

    # ------------------------------------------------------------------
    # Callbacks — bridge events back to the browser / DB
    # ------------------------------------------------------------------

    async def on_user_transcript(transcript: str) -> None:
        """Persist and broadcast the user's transcribed speech."""
        logger.info(f"[RealtimeWS] User: {transcript[:120]}")
        try:
            user_msg = schemas_chat_message.ChatMessageCreate(
                message=transcript, message_type="message"
            )
            db_msg = chat_service.create_chat_message(
                db, user_msg, agent_id, session_id, company_id, "user"
            )
            await manager.broadcast_to_session(
                session_id,
                schemas_chat_message.ChatMessage.model_validate(db_msg).model_dump_json(),
                "user",
            )
        except Exception as exc:
            logger.error(f"[RealtimeWS] Error saving user transcript: {exc}")

    async def on_agent_transcript(transcript: str) -> None:
        """Persist and broadcast the agent's response text."""
        logger.info(f"[RealtimeWS] Agent: {transcript[:120]}")
        try:
            agent_msg = schemas_chat_message.ChatMessageCreate(
                message=transcript, message_type="message"
            )
            db_msg = chat_service.create_chat_message(
                db, agent_msg, agent_id, session_id, company_id, "agent"
            )
            await manager.broadcast_to_session(
                session_id,
                schemas_chat_message.ChatMessage.model_validate(db_msg).model_dump_json(),
                "agent",
            )
        except Exception as exc:
            logger.error(f"[RealtimeWS] Error saving agent transcript: {exc}")

    async def on_audio_chunk(wav_data: bytes) -> None:
        """Forward WAV audio to the browser."""
        try:
            await websocket.send_bytes(wav_data)
            logger.debug(f"[RealtimeWS] Sent {len(wav_data)} B WAV to browser")
        except Exception as exc:
            logger.error(f"[RealtimeWS] Error sending audio chunk: {exc}")

    async def on_audio_done() -> None:
        """Signal that the current audio response is complete."""
        try:
            await websocket.send_text(json.dumps({"type": "audio_end"}))
        except Exception as exc:
            logger.error(f"[RealtimeWS] Error sending audio_end: {exc}")

    # ------------------------------------------------------------------
    # Enrich system prompt with session context if a session exists
    # ------------------------------------------------------------------
    session_state = await session_manager.get_session(session_id)
    if session_state:
        prompt_builder = PromptBuilder()
        system_prompt = prompt_builder.build_system_prompt(session=session_state)
        logger.info(f"[RealtimeWS] Enriched system prompt from session context")

    # ------------------------------------------------------------------
    # Create and connect the Realtime agent (with tool calling)
    # ------------------------------------------------------------------
    realtime_agent = RealtimeVoiceAgent(
        system_prompt=system_prompt,
        greeting=greeting,
        voice=voice,
        # Register both tools so the model can call them during the conversation
        tools=[CREATE_INCIDENT_SCHEMA, GENERATE_TICKET_SCHEMA],
        tool_handlers={
            "create_incident": create_incident,
            "generate_ticket": generate_ticket,
        },
        on_user_transcript=on_user_transcript,
        on_agent_transcript=on_agent_transcript,
        on_audio_chunk=on_audio_chunk,
        on_audio_done=on_audio_done,
        # Persist tool execution results to the session
        on_tool_result=lambda name, result: session_manager.update_context(
            session_id, **{f"_last_tool_{name}": result[:200]}
        ),
    )

    connected = await realtime_agent.connect(api_key=openai_api_key)
    if not connected:
        logger.error("[RealtimeWS] Failed to connect to OpenAI Realtime API")
        try:
            await websocket.send_text(
                json.dumps({"error": "Failed to connect to AI voice service"})
            )
        except Exception:
            pass
        await websocket.close(code=1011)
        manager.disconnect(websocket, session_id)
        return

    # Send initial greeting
    await realtime_agent.send_greeting()

    # ------------------------------------------------------------------
    # Audio buffer — batch WebM chunks before converting
    # ------------------------------------------------------------------
    audio_buffer: bytearray = bytearray()
    last_audio_time: float | None = None

    async def handle_client_audio() -> None:
        """Receive binary (WebM) audio and text control messages from the browser."""
        nonlocal audio_buffer, last_audio_time

        try:
            while True:
                msg = await websocket.receive()

                if "text" in msg:
                    # Control messages: interrupt signal etc.
                    try:
                        data = json.loads(msg["text"])
                        if data.get("type") == "interrupt":
                            await realtime_agent.cancel_response()
                            audio_buffer.clear()
                            last_audio_time = None
                    except (json.JSONDecodeError, Exception):
                        pass

                elif "bytes" in msg:
                    chunk: bytes = msg["bytes"]
                    logger.debug(f"[RealtimeWS] Received {len(chunk)} B audio")
                    last_audio_time = asyncio.get_event_loop().time()
                    audio_buffer.extend(chunk)

        except WebSocketDisconnect:
            logger.info(f"[RealtimeWS] Client disconnected (session={session_id})")
        except Exception as exc:
            logger.error(f"[RealtimeWS] handle_client_audio() error: {exc}")

    async def process_audio_buffer() -> None:
        """
        Poll the audio buffer and forward converted audio to OpenAI Realtime
        when a 500 ms silence is detected (same heuristic as public_voice.py).
        """
        nonlocal audio_buffer, last_audio_time

        while True:
            await asyncio.sleep(0.3)

            if not audio_buffer or last_audio_time is None:
                continue

            elapsed = asyncio.get_event_loop().time() - last_audio_time
            if elapsed < 0.5:
                continue

            # Silence threshold reached — process the buffer
            chunk = bytes(audio_buffer)
            audio_buffer.clear()
            last_audio_time = None

            logger.debug(f"[RealtimeWS] Converting {len(chunk)} B WebM → PCM16")
            pcm16 = await convert_webm_to_pcm16_24k(chunk)
            if pcm16:
                await realtime_agent.send_audio(pcm16)
                await realtime_agent.commit_audio()
                logger.debug(f"[RealtimeWS] Sent {len(pcm16)} B PCM16 to OpenAI and committed buffer")

    # ------------------------------------------------------------------
    # Run tasks concurrently
    # ------------------------------------------------------------------
    client_task = asyncio.create_task(handle_client_audio())
    buffer_task = asyncio.create_task(process_audio_buffer())
    events_task = asyncio.create_task(realtime_agent.process_events())

    try:
        # Block until the client disconnects (client_task finishes)
        await client_task
    except Exception as exc:
        logger.error(f"[RealtimeWS] Session error: {exc}")
    finally:
        buffer_task.cancel()
        events_task.cancel()

        await realtime_agent.disconnect()
        manager.disconnect(websocket, session_id)

        # Mark the session as closed in the session manager
        await session_manager.close_session(session_id)

        logger.info(f"[RealtimeWS] Session cleaned up (session={session_id})")
