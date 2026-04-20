#!/bin/bash
# Start script for LiveKit Workflow Voice Agent (incident creation)

set -a
source .env
set +a

echo "================================================"
echo "LiveKit Workflow Voice Agent - Starting"
echo "================================================"
echo ""
echo "LiveKit URL : $LIVEKIT_URL2"
echo "Backend URL : $BACKEND_URL"
echo "LLM Model   : ${AGENT_LLM_MODEL:-gpt-4o-mini}"
echo "Room filter : voice_workflow_*"
echo ""
echo "Press Ctrl+C to stop"
echo "================================================"
echo ""

python app/agents/workflow_voice_agent.py dev
