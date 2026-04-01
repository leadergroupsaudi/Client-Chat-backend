from fastapi import WebSocket, UploadFile
import aiohttp
import asyncio
from typing import Literal
from app.core.config import settings

# Deepgram configuration
DEEPGRAM_API_KEY = settings.DEEPGRAM_API_KEY
DEEPGRAM_URL = "wss://api.deepgram.com/v1/listen?model=nova-2&interim_results=true&utterance_end_ms=1000&vad_events=true&keepalive=true"

# Groq configuration
GROQ_API_KEY = settings.GROQ_API_KEY
GROQ_API_URL = "https://api.groq.com/openai/v1/audio"

# OpenAI configuration
OPENAI_API_KEY = settings.OPENAI_API_KEY
OPENAI_API_URL = "https://api.openai.com/v1/audio"


class OpenAISTTService:
    """OpenAI Whisper-based Speech-to-Text service."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or OPENAI_API_KEY
        self.base_url = OPENAI_API_URL

        if not self.api_key:
            raise ValueError(
                "OpenAI API key is missing. Please set OPENAI_API_KEY environment variable "
                "or provide api_key parameter."
            )

    async def transcribe(self, file, model: str = "whisper-1") -> dict:
        """
        Convert audio to text using OpenAI's Whisper transcription service.
        """
        url = f"{self.base_url}/transcriptions"
        async with aiohttp.ClientSession() as session:
            form = aiohttp.FormData()
            form.add_field("file", await file.read(), filename=file.filename, content_type=file.content_type)
            form.add_field("model", model)

            headers = {"Authorization": f"Bearer {self.api_key}"}

            async with session.post(url, data=form, headers=headers) as response:
                response.raise_for_status()
                return await response.json()

    async def transcribe_pcm(self, pcm_bytes: bytes, sample_rate: int = 16000, model: str = "whisper-1") -> dict:
        """
        Transcribe raw PCM audio bytes using OpenAI Whisper.
        Creates a temporary WAV buffer for the API.

        Args:
            pcm_bytes: Raw PCM 16-bit audio bytes
            sample_rate: Sample rate of the audio (default 16000 for Whisper)
            model: Whisper model to use

        Returns:
            Transcription result dict with 'text' field
        """
        from app.services.audio_conversion_service import AudioConversionService

        wav_buffer = AudioConversionService.create_wav_buffer(pcm_bytes, sample_rate)

        url = f"{self.base_url}/transcriptions"
        async with aiohttp.ClientSession() as session:
            form = aiohttp.FormData()
            form.add_field(
                "file",
                wav_buffer.read(),
                filename="audio.wav",
                content_type="audio/wav"
            )
            form.add_field("model", model)

            headers = {"Authorization": f"Bearer {self.api_key}"}

            async with session.post(url, data=form, headers=headers) as response:
                response.raise_for_status()
                return await response.json()


class GroqSTTService:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or GROQ_API_KEY
        self.base_url = GROQ_API_URL

        # Validate API key is present
        if not self.api_key:
            raise ValueError(
                "Groq API key is missing. Please set GROQ_API_KEY environment variable "
                "or provide api_key parameter, or configure agent's Groq credential in vault."
            )

    async def transcribe(self, file: UploadFile, model: Literal["whisper-large-v3-turbo", "whisper-large-v3"] = "whisper-large-v3-turbo") -> dict:
        """
        Convert audio to text using Groq's transcription service.
        """
        url = f"{self.base_url}/transcriptions"
        return await self._request(url, file, model)

    async def translate(self, file: UploadFile, model: Literal["whisper-large-v3-turbo", "whisper-large-v3"] = "whisper-large-v3-turbo") -> dict:
        """
        Translate audio to English text using Groq's translation service.
        """
        url = f"{self.base_url}/translations"
        return await self._request(url, file, model)

    async def _request(self, url: str, file: UploadFile, model: str) -> dict:
        async with aiohttp.ClientSession() as session:
            form = aiohttp.FormData()
            form.add_field("file", await file.read(), filename=file.filename, content_type=file.content_type)
            form.add_field("model", model)
            
            headers = {"Authorization": f"Bearer {self.api_key}"}
            
            async with session.post(url, data=form, headers=headers) as response:
                response.raise_for_status()
                return await response.json()

class STTService:
    def __init__(self, websocket: WebSocket, api_key: str = None):
        self.websocket = websocket
        self.session = aiohttp.ClientSession()
        self.deepgram_ws = None
        self.keepalive_task = None
        # Use provided API key, fallback to environment variable
        self.api_key = api_key or DEEPGRAM_API_KEY

    async def connect(self):
        try:
            if not getattr(self, 'session', None) or self.session.closed:
                self.session = aiohttp.ClientSession()

            self.deepgram_ws = await self.session.ws_connect(
                DEEPGRAM_URL,
                headers={"Authorization": f"Token {self.api_key}"}
            )

            # Start a background task to send KeepAlive messages
            self.keepalive_task = asyncio.create_task(self._keepalive_loop())

            return True
        except Exception as e:
            print(f"Error connecting to Deepgram: {e}")
            return False

    async def _keepalive_loop(self):
        while self.deepgram_ws and not self.deepgram_ws.closed:
            try:
                await asyncio.sleep(5)
                if self.deepgram_ws and not self.deepgram_ws.closed:
                    await self.deepgram_ws.send_json({"type": "KeepAlive"})
            except Exception as e:
                print(f"Error sending Deepgram KeepAlive: {e}")
                break

    async def stream(self):
        """
        Streams audio from the client to Deepgram and sends back transcripts.
        """
        if not self.deepgram_ws:
            await self.connect()

        async def deepgram_receiver():
            """Receives transcripts from Deepgram and forwards them to the client."""
            while self.deepgram_ws and not self.deepgram_ws.closed:
                try:
                    message = await self.deepgram_ws.receive_json()
                    if message.get("type") == "Results":
                        transcript = message["channel"]["alternatives"][0]["transcript"]
                        if transcript:
                            # Forward the transcript back to the client
                            await self.websocket.send_json({
                                "type": "transcript",
                                "data": transcript
                            })
                except Exception as e:
                    print(f"Error receiving from Deepgram: {e}")
                    break

        async def client_receiver():
            """Receives audio from the client and forwards it to Deepgram."""
            try:
                while True:
                    audio_chunk = await self.websocket.receive_bytes()
                    if self.deepgram_ws and not self.deepgram_ws.closed:
                        await self.deepgram_ws.send_bytes(audio_chunk)
                    else:
                        print("Deepgram connection not available.")
                        break
            except Exception as e:
                print(f"Error receiving from client: {e}")

        # Run both tasks concurrently
        if self.deepgram_ws:
            await asyncio.gather(deepgram_receiver(), client_receiver())

    async def close(self):
        if self.deepgram_ws and not self.deepgram_ws.closed:
            await self.deepgram_ws.close()
        if self.session and not self.session.closed:
            await self.session.close()