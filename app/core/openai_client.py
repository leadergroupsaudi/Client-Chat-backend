"""
OpenAI Realtime API WebSocket Client.

Low-level async WebSocket wrapper for the OpenAI Realtime API.
Handles connection lifecycle, JSON event sending/receiving, and an
async-generator-based event stream.

All business logic — session configuration, tool calling, audio buffering,
callback dispatch — lives in the agent layer (RealtimeVoiceAgent) above
this class.

Usage::

    client = OpenAIRealtimeClient(api_key="sk-...", model="gpt-4o-mini-realtime-preview")

    if not await client.connect():
        raise RuntimeError("Could not connect to OpenAI Realtime API")

    # Send a session configuration event
    await client.send({
        "type": "session.update",
        "session": {"voice": "alloy", "instructions": "You are a helpful assistant."},
    })

    # Consume events
    async for event in client.listen():
        print(event["type"])

    await client.close()
"""

import asyncio
import json
import logging
import os
from typing import AsyncGenerator, Optional

import websockets
from websockets.exceptions import ConnectionClosed

logger = logging.getLogger("openai-realtime-client")

OPENAI_REALTIME_BASE_URL = "wss://api.openai.com/v1/realtime"
DEFAULT_MODEL = os.getenv("OPENAI_REALTIME_MODEL", "gpt-4o-mini-realtime-preview")


class OpenAIRealtimeClient:
    """
    Thin async WebSocket client for the OpenAI Realtime API.

    Responsibilities:
        - Open / close the WebSocket connection.
        - Send JSON-serialisable event dicts.
        - Receive and parse JSON events (single recv or streaming generator).

    This class intentionally does *not* know about audio formats, tool schemas,
    or session state — those concerns belong to the agent layer.

    Args:
        api_key: OpenAI API key (``sk-...``).
        model:   Realtime model identifier (defaults to ``OPENAI_REALTIME_MODEL`` env var).
    """

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self.is_connected: bool = False

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        """
        Open the WebSocket connection to OpenAI Realtime API.

        Returns:
            True if the connection succeeded, False otherwise.
        """
        url = f"{OPENAI_REALTIME_BASE_URL}?model={self.model}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "realtime=v1",
        }
        try:
            self._ws = await websockets.connect(
                url,
                additional_headers=headers,
                ping_interval=20,
                ping_timeout=10,
            )
            self.is_connected = True
            logger.info(f"[OpenAIRealtimeClient] Connected (model={self.model})")
            return True
        except Exception as exc:
            logger.error(f"[OpenAIRealtimeClient] Connection failed: {exc}", exc_info=True)
            self.is_connected = False
            return False

    async def close(self) -> None:
        """Close the WebSocket connection gracefully."""
        self.is_connected = False
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        logger.info("[OpenAIRealtimeClient] Connection closed")

    # ------------------------------------------------------------------
    # Sending events
    # ------------------------------------------------------------------

    async def send(self, payload: dict) -> None:
        """
        Serialise *payload* as JSON and send it to OpenAI Realtime.

        Args:
            payload: Any JSON-serialisable dict conforming to the Realtime API
                     event schema (e.g. ``{"type": "input_audio_buffer.append", ...}``).
        """
        if not self._ws or not self.is_connected:
            logger.warning("[OpenAIRealtimeClient] send() called while disconnected — skipping")
            return
        try:
            await self._ws.send(json.dumps(payload))
        except Exception as exc:
            logger.error(f"[OpenAIRealtimeClient] Send error: {exc}")
            self.is_connected = False

    # ------------------------------------------------------------------
    # Receiving events
    # ------------------------------------------------------------------

    async def recv(self) -> Optional[dict]:
        """
        Receive a single event from OpenAI Realtime.

        Useful for bootstrapping (e.g. waiting for ``session.created`` before
        starting the event loop).

        Returns:
            Parsed JSON dict, or None on connection close / parse error.
        """
        if not self._ws:
            return None
        try:
            raw = await self._ws.recv()
            return json.loads(raw)
        except ConnectionClosed:
            logger.info("[OpenAIRealtimeClient] Connection closed during recv()")
            self.is_connected = False
            return None
        except json.JSONDecodeError as exc:
            logger.warning(f"[OpenAIRealtimeClient] JSON decode error in recv(): {exc}")
            return None
        except Exception as exc:
            logger.error(f"[OpenAIRealtimeClient] recv() error: {exc}", exc_info=True)
            self.is_connected = False
            return None

    async def recv_with_timeout(self, timeout: float = 10.0) -> Optional[dict]:
        """
        Receive a single event with a maximum wait time.

        Args:
            timeout: Seconds to wait before raising asyncio.TimeoutError.

        Returns:
            Parsed JSON dict, or None on failure.

        Raises:
            asyncio.TimeoutError: If no message arrives within *timeout* seconds.
        """
        raw = await asyncio.wait_for(self._ws.recv(), timeout=timeout)
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning(f"[OpenAIRealtimeClient] JSON decode error: {exc}")
            return None

    async def listen(self) -> AsyncGenerator[dict, None]:
        """
        Async generator that yields parsed events from the WebSocket until
        the connection closes or an error occurs.

        Usage::

            async for event in client.listen():
                await handle_event(event)
        """
        if not self._ws:
            logger.error("[OpenAIRealtimeClient] listen() called before connect()")
            return

        try:
            async for raw in self._ws:
                try:
                    yield json.loads(raw)
                except json.JSONDecodeError as exc:
                    logger.warning(f"[OpenAIRealtimeClient] JSON decode in listen(): {exc}")
        except ConnectionClosed as exc:
            logger.info(f"[OpenAIRealtimeClient] WebSocket closed in listen(): {exc}")
        except Exception as exc:
            logger.error(f"[OpenAIRealtimeClient] listen() error: {exc}", exc_info=True)
        finally:
            self.is_connected = False

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "OpenAIRealtimeClient":
        await self.connect()
        return self

    async def __aexit__(self, *_) -> None:
        await self.close()
