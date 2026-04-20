
import aiohttp
import logging
from typing import AsyncGenerator
import os

logger = logging.getLogger(__name__)

# --- OpenAI TTS Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_TTS_URL = "https://api.openai.com/v1/audio/speech"

# --- Service for our custom Voice Engine ---
VOICE_ENGINE_URL = os.getenv("VOICE_ENGINE_URL", "http://voice-engine-service:8001/api/v1/synthesize")

class VoiceEngineTTSService:
    async def text_to_speech_stream(self, text: str, voice_id: str, session: aiohttp.ClientSession) -> AsyncGenerator[bytes, None]:
        headers = { "Content-Type": "application/json" }
        data = { "text": text, "voice_id": voice_id }
        try:
            async with session.post(VOICE_ENGINE_URL, json=data, headers=headers) as response:
                response.raise_for_status()
                async for chunk in response.content.iter_any():
                    yield chunk
        except Exception as e:
            logger.error(f"Error streaming from Voice Engine ({VOICE_ENGINE_URL}): {e}")

# --- Service for Local AI ---
LOCALAI_TTS_URL = os.getenv("LOCALAI_TTS_URL", "http://localhost:8082/tts")

class LocalAITTSService:
    async def text_to_speech_stream(self, text: str, voice_id: str, session: aiohttp.ClientSession) -> AsyncGenerator[bytes, None]:
        headers = { "Content-Type": "application/json" }
        data = { "model": "voice-en-us-ryan-medium", "input": text, "voice": voice_id }
        try:
            async with session.post(LOCALAI_TTS_URL, json=data, headers=headers) as response:
                response.raise_for_status()
                async for chunk in response.content.iter_any():
                    yield chunk
        except Exception as e:
            print(f"Error streaming from Local AI: {e}")


# --- Service for OpenAI TTS ---
class OpenAITTSService:
    """OpenAI Text-to-Speech service using the audio/speech endpoint."""

    # Available OpenAI voices: alloy, echo, fable, onyx, nova, shimmer
    DEFAULT_VOICE = "alloy"

    def __init__(self, api_key: str = None):
        self.api_key = api_key or OPENAI_API_KEY

    async def text_to_speech_stream(self, text: str, voice_id: str, session: aiohttp.ClientSession) -> AsyncGenerator[bytes, None]:
        """
        Convert text to speech using OpenAI's TTS API.

        Args:
            text: The text to convert to speech
            voice_id: OpenAI voice name (alloy, echo, fable, onyx, nova, shimmer)
            session: aiohttp client session
        """
        if not self.api_key:
            logger.error("OpenAI API key not configured for TTS")
            return

        # Map common voice IDs or use the provided one
        # If voice_id is 'default' or not a valid OpenAI voice, use alloy
        valid_voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
        voice = voice_id if voice_id in valid_voices else self.DEFAULT_VOICE

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "tts-1",  # Use tts-1 for lower latency, tts-1-hd for higher quality
            "input": text,
            "voice": voice,
            "response_format": "mp3"  # mp3 is widely supported
        }

        try:
            async with session.post(OPENAI_TTS_URL, json=data, headers=headers) as response:
                response.raise_for_status()
                async for chunk in response.content.iter_any():
                    yield chunk
        except aiohttp.ClientResponseError as e:
            logger.error(f"OpenAI TTS API error: {e.status} - {e.message}")
        except Exception as e:
            logger.error(f"Error streaming from OpenAI TTS: {e}")

    async def text_to_speech_pcm(
        self,
        text: str,
        voice: str = "alloy"
    ) -> bytes:
        """
        Convert text to raw PCM audio bytes for Twilio streaming.
        OpenAI TTS returns 24kHz 16-bit mono PCM when response_format is "pcm".

        Args:
            text: Text to convert to speech
            voice: OpenAI voice name (alloy, echo, fable, onyx, nova, shimmer)

        Returns:
            Raw PCM 16-bit 24kHz mono audio bytes
        """
        if not self.api_key:
            raise ValueError("OpenAI API key not configured for TTS")

        valid_voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
        voice = voice if voice in valid_voices else self.DEFAULT_VOICE

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "tts-1",
            "input": text,
            "voice": voice,
            "response_format": "pcm"  # Raw PCM output at 24kHz
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(OPENAI_TTS_URL, json=data, headers=headers) as response:
                response.raise_for_status()
                return await response.read()


# --- Main TTS Router Service ---
class TTSService:
    def __init__(self, openai_api_key: str = None):
        self.session = None
        self.voice_engine_service = VoiceEngineTTSService()
        self.localai_service = LocalAITTSService()
        self.openai_service = OpenAITTSService(api_key=openai_api_key)

    async def _get_session(self):
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session

    async def text_to_speech_stream(self, text: str, voice_id: str, provider: str) -> AsyncGenerator[bytes, None]:
        """
        Routes the TTS request to the appropriate provider based on the agent's configuration.

        Supported providers:
        - openai: OpenAI TTS (voices: alloy, echo, fable, onyx, nova, shimmer)
        - localai: Local AI TTS
        - voice_engine: Custom voice engine service
        """
        session = await self._get_session()

        if provider == 'openai':
            async for chunk in self.openai_service.text_to_speech_stream(text, voice_id, session):
                yield chunk
        elif provider == 'localai':
            async for chunk in self.localai_service.text_to_speech_stream(text, voice_id, session):
                yield chunk
        elif provider == 'voice_engine':
            async for chunk in self.voice_engine_service.text_to_speech_stream(text, voice_id, session):
                yield chunk
        else:
            logger.warning(f"Unknown TTS provider: {provider}. Defaulting to openai.")
            async for chunk in self.openai_service.text_to_speech_stream(text, voice_id, session):
                yield chunk

    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()
