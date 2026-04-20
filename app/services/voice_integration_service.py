"""
Voice Agent Integration Service
================================

Bridges the Cintrix agent/tool configuration system with the OpenAI Realtime API.

Responsibilities:
  - Convert Cintrix Tool DB records into OpenAI Realtime function schemas
  - Build the complete tool list (agent tools + always-available voice tools)
  - Build a voice-optimised system prompt from an Agent record
  - Provide a reusable session-config builder used by the token endpoint

No external state is mutated here; every function is a pure transformation.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from app.models.agent import Agent
from app.models.tool import Tool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single tool conversion
# ---------------------------------------------------------------------------

def convert_tool_to_openai_schema(tool: Tool) -> Optional[Dict[str, Any]]:
    """
    Convert one Cintrix Tool model into an OpenAI Realtime function descriptor.

    OpenAI format::

        {
            "type": "function",
            "name": "tool_name",
            "description": "...",
            "parameters": {
                "type": "object",
                "properties": { ... },
                "required": [...]
            }
        }

    Returns ``None`` when the tool has no usable name.
    """
    if not tool.name:
        return None

    # Resolve the JSON-schema for function parameters.
    # Cintrix stores `parameters` as a JSON column (already a dict) or as a
    # JSON string – handle both gracefully.
    parameters: Dict[str, Any] = {"type": "object", "properties": {}}

    if tool.parameters:
        if isinstance(tool.parameters, dict):
            parameters = tool.parameters
        elif isinstance(tool.parameters, str):
            try:
                parsed = json.loads(tool.parameters)
                if isinstance(parsed, dict):
                    parameters = parsed
            except (json.JSONDecodeError, ValueError):
                logger.warning(
                    "[VoiceIntegration] Could not parse parameters JSON for tool '%s'",
                    tool.name,
                )

    return {
        "type": "function",
        "name": tool.name,
        "description": tool.description or f"Execute the {tool.name} action.",
        "parameters": parameters,
    }


# ---------------------------------------------------------------------------
# Agent-level tool list
# ---------------------------------------------------------------------------

def build_agent_tools_for_openai(agent: Agent) -> List[Dict[str, Any]]:
    """
    Build the complete OpenAI Realtime tool list for *agent*.

    Includes:
    * All custom / MCP tools attached to the agent.
    * A hard-wired ``end_session`` tool so the AI can gracefully close calls.

    MCP tools expose their full schema via ``tool.parameters``; builtin tools
    use the Cintrix ``parameters`` JSON column too.
    """
    openai_tools: List[Dict[str, Any]] = []

    if agent.tools:
        for tool in agent.tools:
            schema = convert_tool_to_openai_schema(tool)
            if schema:
                openai_tools.append(schema)
                logger.debug("[VoiceIntegration] Registered tool: %s", tool.name)
            else:
                logger.warning(
                    "[VoiceIntegration] Skipped tool with missing name (id=%s)", tool.id
                )

    # Always-available voice control tool
    openai_tools.append(
        {
            "type": "function",
            "name": "end_session",
            "description": (
                "End the voice conversation after you have completed the user's "
                "request and said a proper goodbye. Call this as the very last action."
            ),
            "parameters": {"type": "object", "properties": {}},
        }
    )

    logger.info(
        "[VoiceIntegration] Built %d tool(s) for agent '%s' (id=%s)",
        len(openai_tools),
        agent.name,
        agent.id,
    )
    return openai_tools


# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------

def build_system_prompt_for_voice(agent: Agent) -> str:
    """
    Build a voice-optimised system prompt from an Agent record.

    Starts from ``agent.prompt`` (or a sensible default), then appends:
    * A reminder to keep responses short and natural for voice.
    * A brief tool-awareness hint when the agent has tools.
    """
    base: str = agent.prompt or (
        "You are a helpful and friendly voice assistant. "
        "Keep your responses concise and natural for voice conversation."
    )

    # Voice-mode reminder (appended only when not already explicit)
    voice_hint = (
        "\n\nIMPORTANT – You are in a VOICE call. "
        "Keep every reply SHORT (1-3 sentences). "
        "Never read out JSON, URLs, or raw code. "
        "Always speak naturally as if talking on the phone."
    )
    if "voice" not in base.lower():
        base += voice_hint

    # Tool awareness
    if agent.tools:
        tool_names = [t.name for t in agent.tools if t.name]
        if tool_names:
            base += (
                "\n\nYou have access to these tools: "
                + ", ".join(tool_names)
                + ". Use them whenever they help fulfil the user's request. "
                "After a tool call completes, summarise the result in one sentence."
            )

    return base


# ---------------------------------------------------------------------------
# OpenAI session config builder
# ---------------------------------------------------------------------------

_VALID_VOICES = frozenset(
    {"alloy", "echo", "shimmer", "ash", "ballad", "coral", "sage", "verse"}
)


def build_openai_session_config(
    agent: Agent,
    *,
    model: str = "gpt-realtime",
    token_ttl_seconds: int = 600,
) -> Dict[str, Any]:
    """
    Build the full payload sent to ``POST /v1/realtime/client_secrets``.

    This is a pure config builder – callers are responsible for sending the
    HTTP request with the correct API key.

    Args:
        agent:              Cintrix Agent ORM instance (with ``tools`` loaded).
        model:              OpenAI Realtime model identifier.
        token_ttl_seconds:  How long the ephemeral token stays valid.

    Returns:
        A dict ready to be JSON-serialised as the request body.
    """
    system_prompt = build_system_prompt_for_voice(agent)
    tools = build_agent_tools_for_openai(agent)

    voice = "alloy"
    if agent.voice_id and agent.voice_id.lower() in _VALID_VOICES:
        voice = agent.voice_id.lower()

    return {
        "expires_after": {
            "anchor": "created_at",
            "seconds": token_ttl_seconds,
        },
        "session": {
            "type": "realtime",
            "model": model,
            "instructions": system_prompt,
            "output_modalities": ["audio"],
            "tools": tools,
            "tool_choice": "auto",
            "audio": {
                "input": {
                    "format": {
                        "type": "audio/pcm",
                        "rate": 24000,
                    },
                    "transcription": {
                        "model": "whisper-1",
                    },
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.5,
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": 500,
                        "create_response": True,
                        "interrupt_response": True,
                    },
                },
                "output": {
                    "format": {
                        "type": "audio/pcm",
                        "rate": 24000,
                    },
                    "voice": voice,
                },
            },
        },
    }
