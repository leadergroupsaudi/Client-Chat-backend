import os
from typing import Optional

from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "AgentConnect"
    API_V1_STR: str = "/api/v1"
    DATABASE_URL: str
    GOOGLE_API_KEY: str = ""
    GROQ_API_KEY: str = ""
    NVIDIA_API_KEY: str = ""
    LIVEKIT_API_KEY: str = ""
    LIVEKIT_API_SECRET: str = ""
    LIVEKIT_URL: str = ""
    FRONTEND_URL: str
    WHATSAPP_VERIFY_TOKEN: str = ""
    MESSENGER_VERIFY_TOKEN: str = ""
    INSTAGRAM_VERIFY_TOKEN: str = ""
    GMAIL_CLIENT_ID: str = ""
    GMAIL_CLIENT_SECRET: str = ""
    GMAIL_REDIRECT_URI: str = ""
    GOOGLE_CLIENT_SECRETS_FILE: str = ""
    GOOGLE_CLIENT_ID: str = ""
    GOOGLE_CLIENT_SECRET: str = ""
    TELEGRAM_BOT_TOKEN: str = ""
    LINKEDIN_CLIENT_ID: str = ""
    LINKEDIN_CLIENT_SECRET: str = ""
    LINKEDIN_REDIRECT_URI: str = ""
    LINKEDIN_COMPANY_ID: str = ""
    STRIPE_SECRET_KEY: str = ""
    STRIPE_WEBHOOK_SECRET: str = ""
    LOCALAI_TTS_URL: str = "http://localhost:8082/tts"
    BACKEND_URL: str = "https://livechat.discretal.com/"

    # Twilio SMS settings
    TWILIO_ACCOUNT_SID: str = ""
    TWILIO_AUTH_TOKEN: str = ""
    TWILIO_PHONE_NUMBER: str = ""

    # MinIO/S3 settings
    minio_endpoint: str
    minio_secure: bool = False
    minio_access_key: str
    minio_secret_key: str
    minio_bucket: str = "agentconnect"
    minio_strict: bool = True

    FAISS_INDEX_DIR: str = "./faiss_indexes"

    CHROMA_DB_HOST: Optional[str] = None
    CHROMA_DB_PORT: Optional[int] = None

    SECRET_KEY: str = "S48jcPB4nMH0gVLHb3Py7DBGp91Xv3bUaDzsn5zB3jg="
    ALGORITHM: str = "HS256"

    # Server configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: str = "http://localhost:8080,http://localhost:5173,*"
    PUBLIC_HOST: Optional[str] = None  # Public hostname for WebSocket URLs (e.g., api.example.com)

    # WebSocket session cleanup settings
    WS_PING_INTERVAL: int = 30  # Send ping every 30 seconds
    WS_CLEANUP_INTERVAL: int = 60  # Run cleanup every 60 seconds
    WS_REGULAR_SESSION_TIMEOUT: int = 1800  # 30 minutes (1800 seconds)
    WS_PREVIEW_SESSION_TIMEOUT: int = 300  # 5 minutes (300 seconds)
    WS_ENABLE_HEARTBEAT: bool = True  # Feature flag to enable/disable heartbeat

    # Workflow Configuration
    MAX_SUBWORKFLOW_DEPTH: int = 5  # Maximum depth for nested subworkflows

    # LLM Streaming Configuration
    LLM_STREAMING_ENABLED: bool = False  # Disabled streaming responses by default
    LLM_STREAM_TOKEN_BUFFER: int = 1  # Number of tokens to buffer before sending (1 = real-time)
    LLM_REQUEST_TIMEOUT: int = 120  # Timeout for LLM API calls in seconds
    HTTP_REQUEST_TIMEOUT: int = 30  # Timeout for workflow HTTP requests in seconds

    # OpenAI Realtime API Configuration
    OPENAI_REALTIME_ENABLED: bool = True  # Use OpenAI Realtime API for voice calls
    OPENAI_REALTIME_MODEL: str = "gpt-realtime"
    OPENAI_REALTIME_VOICE: str = "alloy"  # alloy, echo, shimmer, ash, ballad, coral, sage, verse

    # LiveKit AI Agents Configuration
    OPENAI_API_KEY: str = ""
    DEEPGRAM_API_KEY: str = ""
    AGENT_LLM_PROVIDER: str = "openai"
    AGENT_LLM_MODEL: str = "gpt-4o-mini"
    AGENT_STT_PROVIDER: str = "deepgram"
    AGENT_STT_LANGUAGE: str = "en"
    AGENT_TTS_PROVIDER: str = "openai"
    AGENT_TTS_VOICE: str = "alloy"
    AGENT_VAD_ENABLED: str = "true"
    AGENT_ALLOW_INTERRUPTIONS: str = "true"
    AGENT_GREETING: str = "Hello! I'm your AI voice assistant. How can I help you today?"
    AGENT_SYSTEM_PROMPT: str = "You are a helpful and friendly voice assistant. Keep your responses concise and natural for voice conversation."

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        extra = 'ignore'  # Ignore extra env vars not defined in Settings

settings = Settings()
