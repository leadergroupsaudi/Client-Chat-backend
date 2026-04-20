"""
LiveKit Voice Agent - Standalone worker for voice interactions.

This agent can be run as a separate process using the LiveKit agents CLI:
    python app/agents/voice_agent.py dev
    python app/agents/voice_agent.py start
"""
import logging
import os

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    metrics,
    RoomInputOptions,
    llm
)
from livekit.plugins import deepgram, openai, silero, groq, noise_cancellation

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("voice-agent")


# Load environment variables
# These should be set in your .env file or environment
LIVEKIT_URL = os.getenv("LIVEKIT_URL", "wss://intersee-isrh22yh.livekit.cloud")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")


class AgentConfig:
    """Configuration for the voice agent."""

    # LLM Configuration
    LLM_PROVIDER = os.getenv("AGENT_LLM_PROVIDER", "openai")  # openai or groq
    LLM_MODEL = os.getenv("AGENT_LLM_MODEL", "gpt-4o-mini")  # gpt-4o, gpt-4o-mini, or groq models

    # STT Configuration
    STT_PROVIDER = os.getenv("AGENT_STT_PROVIDER", "deepgram")  # deepgram or groq
    STT_LANGUAGE = os.getenv("AGENT_STT_LANGUAGE", "en")

    # TTS Configuration
    TTS_PROVIDER = os.getenv("AGENT_TTS_PROVIDER", "openai")  # openai
    TTS_VOICE = os.getenv("AGENT_TTS_VOICE", "alloy")  # alloy, echo, fable, onyx, nova, shimmer

    # Agent Behavior
    INITIAL_GREETING = os.getenv(
        "AGENT_GREETING",
        "Hello! I'm your AI voice assistant. How can I help you today?"
    )
    SYSTEM_PROMPT = os.getenv(
        "AGENT_SYSTEM_PROMPT",
        """You are a helpful and friendly voice assistant.
        Keep your responses concise and natural for voice conversation.
        Avoid long-winded explanations unless specifically asked.
        If you don't know something, be honest about it."""
    )

    # Performance Settings
    VAD_ENABLED = os.getenv("AGENT_VAD_ENABLED", "true").lower() == "true"
    ALLOW_INTERRUPTIONS = os.getenv("AGENT_ALLOW_INTERRUPTIONS", "true").lower() == "true"


def get_llm() -> llm.LLM:
    """Get the configured LLM instance."""
    if AgentConfig.LLM_PROVIDER == "openai":
        logger.info(f"Using OpenAI LLM: {AgentConfig.LLM_MODEL}")
        return openai.LLM(model=AgentConfig.LLM_MODEL)
    elif AgentConfig.LLM_PROVIDER == "groq":
        logger.info(f"Using Groq LLM: {AgentConfig.LLM_MODEL}")
        return groq.LLM(model=AgentConfig.LLM_MODEL)
    else:
        raise ValueError(f"Unsupported LLM provider: {AgentConfig.LLM_PROVIDER}")


def get_stt():
    """Get the configured STT instance."""
    if AgentConfig.STT_PROVIDER == "deepgram":
        logger.info("Using Deepgram STT")
        return deepgram.STT(language=AgentConfig.STT_LANGUAGE)
    elif AgentConfig.STT_PROVIDER == "openai":
        logger.info("Using OpenAI STT (Whisper)")
        return openai.STT()
    elif AgentConfig.STT_PROVIDER == "groq":
        logger.info("Using Groq STT (Whisper)")
        return groq.STT()
    elif AgentConfig.STT_PROVIDER == "openai_groq":
        logger.info("Using OpenAI STT with Groq backend")
        return openai.STT.with_groq()
    else:
        raise ValueError(f"Unsupported STT provider: {AgentConfig.STT_PROVIDER}")


def get_tts():
    """Get the configured TTS instance."""
    if AgentConfig.TTS_PROVIDER == "openai":
        logger.info(f"Using OpenAI TTS: {AgentConfig.TTS_VOICE}")
        return openai.TTS(voice=AgentConfig.TTS_VOICE)
    else:
        raise ValueError(f"Unsupported TTS provider: {AgentConfig.TTS_PROVIDER}")


class VoiceAssistant(Agent):
    """Voice Assistant Agent using the newer LiveKit Agents API."""

    def __init__(self) -> None:
        """
        Initialize the voice assistant with configured LLM, STT, and TTS.
        """
        super().__init__(
            instructions=AgentConfig.SYSTEM_PROMPT,
            stt=get_stt(),
            llm=get_llm(),
            tts=get_tts(),
        )
        logger.info("Voice assistant initialized")

    async def on_enter(self):
        """
        Called when the agent enters the room.
        Greet the user when they join.
        """
        logger.info("Agent entered room, greeting user...")
        self.session.generate_reply(
            instructions=AgentConfig.INITIAL_GREETING,
            allow_interruptions=AgentConfig.ALLOW_INTERRUPTIONS
        )


async def entrypoint(ctx: JobContext):
    """
    Main entrypoint for the voice agent worker.

    This function is called when the agent joins a LiveKit room.
    """
    logger.info(f"Agent connecting to room: {ctx.room.name}")

    # Connect to the room (audio only for voice agent)
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Get room metadata if available
    room_metadata = ctx.room.metadata
    if room_metadata:
        logger.info(f"Room metadata: {room_metadata}")

    # Wait for the first participant to connect
    participant = await ctx.wait_for_participant()
    logger.info(f"Starting voice assistant for participant: {participant.identity}")

    # Set up usage collector for metrics
    # usage_collector = metrics.UsageCollector()

    # # Log metrics and collect usage data
    # def on_metrics_collected(agent_metrics: metrics.AgentMetrics):
    #     """Callback for when metrics are collected."""
    #     metrics.log_metrics(agent_metrics)
    #     usage_collector.collect(agent_metrics)
    #     logger.debug(f"Metrics collected: {agent_metrics}")

    # Create agent session with VAD
    session = AgentSession(
        vad=ctx.proc.userdata.get("vad") if AgentConfig.VAD_ENABLED else None,
        # minimum delay for endpointing, used when turn detector believes user is done
        min_endpointing_delay=0.5,
        # maximum delay for endpointing, used when turn detector does not believe user is done
        max_endpointing_delay=5.0,
    )

    # Register metrics callback
    # session.on("metrics_collected", on_metrics_collected)

    # Start the agent session
    await session.start(
        room=ctx.room,
        agent=VoiceAssistant(),
        # room_input_options=RoomInputOptions(
        #     # Enable background voice & noise cancellation (powered by Krisp)
        #     # Included at no additional cost with LiveKit Cloud
        #     noise_cancellation=noise_cancellation.BVC(),
        # ),
    )

    logger.info("Voice agent session started successfully")


def prewarm(proc: JobProcess):
    """
    Prewarm function to load models before the agent starts.
    This improves first-response latency.
    """
    logger.info("Prewarming agent models...")

    # Prewarm VAD if enabled
    if AgentConfig.VAD_ENABLED:
        proc.userdata["vad"] = silero.VAD.load()

    logger.info("Prewarm complete")


if __name__ == "__main__":
    """
    Run the agent worker.

    Usage:
        # Development mode (auto-reload on code changes)
        python app/agents/voice_agent.py dev

        # Production mode
        python app/agents/voice_agent.py start

        # With custom worker options
        python app/agents/voice_agent.py start --room "my-room"
    """

    # Validate environment variables
    if not LIVEKIT_API_KEY or not LIVEKIT_API_SECRET:
        raise ValueError("LIVEKIT_API_KEY and LIVEKIT_API_SECRET must be set")

    # Validate provider-specific API keys (warnings only)
    if AgentConfig.LLM_PROVIDER == "openai" and not OPENAI_API_KEY:
        logger.warning("OPENAI_API_KEY not set. Voice agent will not work without it.")

    if AgentConfig.LLM_PROVIDER == "groq" and not GROQ_API_KEY:
        logger.warning("GROQ_API_KEY not set. Voice agent will not work without it.")

    if AgentConfig.STT_PROVIDER == "deepgram" and not DEEPGRAM_API_KEY:
        logger.warning("DEEPGRAM_API_KEY not set. Voice agent will not work without it.")

    if AgentConfig.TTS_PROVIDER == "openai" and not OPENAI_API_KEY:
        logger.warning("OPENAI_API_KEY not set for TTS. Voice agent will not work without it.")

    logger.info("=" * 60)
    logger.info("Starting LiveKit Voice Agent...")
    logger.info("=" * 60)
    logger.info(f"LiveKit URL: {LIVEKIT_URL}")
    logger.info(f"LLM: {AgentConfig.LLM_PROVIDER}/{AgentConfig.LLM_MODEL}")
    logger.info(f"STT: {AgentConfig.STT_PROVIDER}")
    logger.info(f"TTS: {AgentConfig.TTS_PROVIDER}/{AgentConfig.TTS_VOICE}")
    logger.info(f"VAD Enabled: {AgentConfig.VAD_ENABLED}")
    logger.info(f"Allow Interruptions: {AgentConfig.ALLOW_INTERRUPTIONS}")
    logger.info("=" * 60)

    # Run the agent CLI
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )