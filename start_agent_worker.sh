#!/bin/bash
# Quick start script for LiveKit AI Agent Worker

echo "================================================"
echo "LiveKit AI Agent Worker - Quick Start"
echo "================================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Error: Virtual environment not found!"
    echo "Please create one with: python -m venv venv"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if required packages are installed
echo "Checking dependencies..."
python -c "import livekit.agents" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Error: livekit-agents not installed!"
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Check environment variables
echo ""
echo "Checking environment variables..."

if [ -z "$LIVEKIT_URL" ] || [ -z "$LIVEKIT_API_KEY" ] || [ -z "$LIVEKIT_API_SECRET" ]; then
    echo "⚠️  Warning: LiveKit credentials not set!"
    echo "Please set the following in your .env file:"
    echo "  - LIVEKIT_URL"
    echo "  - LIVEKIT_API_KEY"
    echo "  - LIVEKIT_API_SECRET"
    echo ""
fi

if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  Warning: OPENAI_API_KEY not set!"
    echo "Required for LLM and TTS. Set in .env file."
    echo ""
fi

if [ -z "$DEEPGRAM_API_KEY" ]; then
    echo "⚠️  Warning: DEEPGRAM_API_KEY not set!"
    echo "Required for Speech-to-Text. Set in .env file."
    echo ""
fi

# Run mode selection
MODE=${1:-dev}

echo "================================================"
echo "Starting LiveKit Agent Worker in $MODE mode..."
echo "================================================"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Start the agent
if [ "$MODE" == "dev" ]; then
    python app/agents/workflow_voice_agent.py dev
elif [ "$MODE" == "start" ]; then
    python app/agents/workflow_voice_agent.py start
else
    echo "Usage: ./start_agent_worker.sh [dev|start]"
    echo "  dev   - Development mode with auto-reload"
    echo "  start - Production mode"
    exit 1
fi
