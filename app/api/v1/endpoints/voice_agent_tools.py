"""
Voice Agent Tool Execution Endpoint
=====================================

Handles server-side tool execution triggered by the OpenAI Realtime voice
agent running in the browser (WebRTC data channel).

Flow
----
1. Browser (OpenAI WebRTC data channel) receives a ``response.function_call_arguments.done``
   event from OpenAI.
2. Browser POSTs the tool name + arguments to ``POST /api/v1/ws/public/voice/tools/execute``.
3. This endpoint resolves the tool via Cintrix's ``tool_execution_service`` and returns the
   result as JSON.
4. Browser relays the result back to OpenAI via a ``conversation.item.create``
   (type: ``function_call_output``) data-channel message, then calls ``response.create``
   to continue the conversation.

Endpoint
--------
POST /api/v1/ws/public/voice/tools/execute

Headers:
    X-Company-ID  (int, required) – company scope for tool lookup

Body (JSON):
    {
        "tool_name":  "create_incident",
        "arguments":  { "title": "...", "severity": "high" },
        "session_id": "voice-abc123",
        "agent_id":   42            // optional, informational
    }

Response (JSON):
    {
        "tool_name": "create_incident",
        "result":    { ... },        // whatever the tool returned
        "success":   true,
        "error":     null
    }
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.dependencies import get_db
from app.services import tool_execution_service

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class VoiceToolExecuteRequest(BaseModel):
    tool_name: str = Field(..., description="Name of the tool to execute.")
    arguments: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments provided by the AI for this tool call.",
    )
    session_id: str = Field(..., description="Voice session identifier.")
    agent_id: Optional[int] = Field(
        default=None,
        description="Agent ID (informational; used for logging).",
    )


class VoiceToolExecuteResponse(BaseModel):
    tool_name: str
    result: Any
    success: bool
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post(
    "/public/voice/tools/execute",
    response_model=VoiceToolExecuteResponse,
    summary="Execute a voice-agent tool call",
    tags=["voice"],
)
async def execute_voice_tool(
    request: VoiceToolExecuteRequest,
    db: Session = Depends(get_db),
    x_company_id: int = Header(
        ...,
        description="Company ID used to scope the tool lookup.",
    ),
) -> VoiceToolExecuteResponse:
    """
    Execute a named Cintrix tool on behalf of the OpenAI Realtime voice agent.

    Called from the browser when the AI emits a ``response.function_call_arguments.done``
    event on the WebRTC data channel.  The caller must relay the result back to
    OpenAI via a ``conversation.item.create / function_call_output`` message.

    Special-cased tools
    -------------------
    * ``end_session`` – acknowledged immediately; the actual WebRTC teardown is
      handled by the browser once it receives the success response.
    """
    logger.info(
        "[VoiceTools] tool=%s session=%s agent=%s company=%s",
        request.tool_name,
        request.session_id,
        request.agent_id,
        x_company_id,
    )

    # end_session is a browser-handled control tool – just ack it.
    if request.tool_name == "end_session":
        return VoiceToolExecuteResponse(
            tool_name="end_session",
            result={"message": "Session ended."},
            success=True,
        )

    try:
        result = await tool_execution_service.execute_tool(
            db=db,
            tool_name=request.tool_name,
            parameters=request.arguments,
            session_id=request.session_id,
            company_id=x_company_id,
        )
    except Exception as exc:
        logger.error(
            "[VoiceTools] Unexpected error executing tool '%s': %s",
            request.tool_name,
            exc,
            exc_info=True,
        )
        return VoiceToolExecuteResponse(
            tool_name=request.tool_name,
            result={"error": str(exc)},
            success=False,
            error=str(exc),
        )

    if result is None:
        msg = f"Tool '{request.tool_name}' not found for company {x_company_id}."
        logger.warning("[VoiceTools] %s", msg)
        return VoiceToolExecuteResponse(
            tool_name=request.tool_name,
            result={"error": msg},
            success=False,
            error=msg,
        )

    # Detect error dicts returned by the execution service
    if isinstance(result, dict) and "error" in result:
        return VoiceToolExecuteResponse(
            tool_name=request.tool_name,
            result=result,
            success=False,
            error=result["error"],
        )

    return VoiceToolExecuteResponse(
        tool_name=request.tool_name,
        result=result,
        success=True,
    )
