"""
realtime_voice_agent.py

OpenAI Realtime API voice agent worker.
Mirrors workflow_voice_agent.py exactly — only the LiveKit transport layer is replaced.

What changed vs workflow_voice_agent.py
----------------------------------------
  LiveKit Worker/cli.run_app  →  standalone websockets server (port REALTIME_WS_PORT)
  LiveKit Room                →  browser WebSocket + OpenAI Realtime WebSocket
  AgentSession + silero VAD   →  OpenAI Realtime server-side VAD (built-in)
  STT / LLM / TTS plugins     →  single OpenAI Realtime session (unified pipeline)
  @llm.function_tool()        →  OpenAI Realtime tool schema + dispatcher
  publish_data()              →  client_ws.send() JSON text frame
  room.disconnect()           →  _terminate_event.set() → WebSocket close

What is IDENTICAL to workflow_voice_agent.py
---------------------------------------------
  Automax3Client class        (verbatim)
  WorkflowContext class       (verbatim)
  generate_system_prompt()    (verbatim)
  All tool function bodies    (same logic)
  Workflow fetch from backend (same HTTP call)
  Session_id extraction       (same pattern)
  Form-event signalling       (same FORM_DONE / FORM_SUBMITTED handling)
  Logging / greeting / prewarm (same)

Run:
    python realtime_voice_agent.py

Browser connects to:
    ws://host:REALTIME_WS_PORT/voice_realtime/{session_id}

Frame protocol:
    binary frames  = audio   (WebM from browser  /  WAV to browser)
    text   frames  = JSON control messages (OPEN_FORM, FORM_DONE, etc.)
"""

import asyncio
import base64
import io
import json
import logging
import os
import struct
from collections import defaultdict, deque
from datetime import datetime
from typing import Any, Awaitable, Callable, Optional

import httpx
import requests
import websockets
from dotenv import load_dotenv
from requests.auth import HTTPBasicAuth

try:
    from websockets.asyncio.client import ClientConnection
    from websockets.asyncio.server import ServerConnection
except ImportError:  # pragma: no cover - compatibility for older websockets
    from websockets.client import WebSocketClientProtocol as ClientConnection
    from websockets.server import WebSocketServerProtocol as ServerConnection

load_dotenv()

# ─────────────────────────────────────────────────────────────
# Logging  (identical to workflow_voice_agent.py)
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("realtime-voice-agent")

EPM_940_REALTIME_SYSTEM_PROMPT = """You are a helpful voice assistant for EPM 940.

YOUR JOB:
- Answer user questions and assist with their requests
- Keep responses SHORT and conversational because this is a voice call
- Be friendly and professional

INCIDENT CLASSIFICATION OPTIONS (hardcoded):
1. Manholes
2. Street Lights
3. Street Furniture
4. Potholes
5. Barriers

LOCATION OPTIONS (hardcoded):
1. Dammam
2. Dammam East
3. Dammam West

INCIDENT REPORTING IS A REQUIRED TOOL-DRIVEN FLOW.
If the user wants to report an incident, you MUST follow this exact order:
1. Ask for caller's name.
2. As soon as the user answers, call store_collected_data with a JSON string for the answer.
3. Call get_available_classifications() before you present the classification options.
4. Ask for the incident classification using the returned hardcoded options.
5. As soon as the user answers, call store_collected_data with a JSON string for the classification.
6. Call get_available_locations() before you present the location options.
7. Ask for the location using the returned hardcoded options.
8. As soon as the user answers, call store_collected_data with a JSON string for the location.
9. Ask for a brief description. This is optional but recommended.
10. If the user gives a description, call store_collected_data with a JSON string for the description.
11. Ask for criticality: LOW, MEDIUM, HIGH, or CRITICAL. If the user does not specify, use LOW.
12. As soon as criticality is known, call store_collected_data with a JSON string for the criticality.
13. After all required details are collected, call trigger_form_popup(caller_name, classification, location, description, criticality).
14. Only after trigger_form_popup succeeds, say this exact template:
"I've opened the report form for you. [Brief Summary]... Goodbye!"
15. Immediately after the final spoken sentence, call terminate_call().

TOOL USAGE RULES:
- store_collected_data(data) MUST be called after every collected answer.
- get_available_classifications() MUST be called before presenting the classification list.
- get_available_locations() MUST be called before presenting the location list.
- trigger_form_popup(...) MUST be called before you say the report form has been opened.
- terminate_call() MUST be the last tool call.
- create_automax_incident(...) is only for direct incident creation without waiting for user form confirmation.

RULES:
- Keep responses SHORT.
- Ask only one missing question at a time.
- Never skip a required tool call.
- Never say the form is open unless you have already called trigger_form_popup(...).
- Do not wait for FORM_DONE before the goodbye unless another workflow step explicitly requires it.
"""

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────

# OpenAI (same key as workflow_voice_agent.py)
OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY")
OPENAI_REALTIME_URL = "wss://api.openai.com/v1/realtime"

# Model / voice — reuse same env vars as the original
REALTIME_MODEL  = os.getenv("AGENT_LLM_MODEL", "gpt-4o-realtime-preview-2024-12-17")
REALTIME_VOICE  = os.getenv("AGENT_TTS_VOICE", "alloy")

# Worker WebSocket server (replaces LiveKit worker URL)
REALTIME_WS_PORT = int(os.getenv("REALTIME_WS_PORT", "8766"))
REALTIME_WS_HOST = os.getenv("REALTIME_WS_HOST", "0.0.0.0")

# Backend URL for workflow fetch (same as workflow_voice_agent.py)
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# Path prefix — equivalent to ROOM_NAME_PREFIX = "voice_workflow_"
WS_PATH_PREFIX = "/voice_realtime/"

# Automax3 API (identical to workflow_voice_agent.py)
AUTOMAX3_BASEURL  = os.getenv("AUTOMAX3_BASEURL", "https://epmstg.automaxsw.com")
AUTOMAX3_USERNAME = os.getenv("AUTOMAX3_USERNAME", "474062237214310401")
AUTOMAX3_PASSWORD = os.getenv("AUTOMAX3_PASSWORD", "FFtq2lE3V3HrLO9DrjQAd6yxb2i4Xehq1zWRK3zELuewQOgOw0p59bx4bkun0jdI")


# ─────────────────────────────────────────────────────────────
# Automax3Client  (VERBATIM from workflow_voice_agent.py)
# ─────────────────────────────────────────────────────────────

class Automax3Client:
    """Client for Automax3 API - handles authentication and incident creation."""

    def __init__(self):
        self.base_url = AUTOMAX3_BASEURL
        self.userid   = AUTOMAX3_USERNAME
        self.password = AUTOMAX3_PASSWORD

        self.token:    Optional[str] = None
        self.id_token: Optional[str] = None

        self.common_headers = {"Content-Type": "application/json"}

        # Cache for classifications and locations
        self._classifications_cache: Optional[list] = None
        self._locations_cache: Optional[list] = None

    def client_login(self) -> Optional[dict]:
        """Authenticate with Automax3 API using client credentials."""
        url  = f"{self.base_url}/auth/oauth2/token"
        data = {"grant_type": "client_credentials", "scope": "profile api"}
        auth = HTTPBasicAuth(self.userid, self.password)
        hdrs = {"Content-Type": "application/x-www-form-urlencoded"}
        try:
            resp = requests.post(url, data=data, headers=hdrs, auth=auth)
            if resp.status_code == 200:
                rd = resp.json()
                self.token    = rd["access_token"]
                self.id_token = rd["id_token"]
                logger.info("Automax3: Access token obtained successfully!")
                return rd
            else:
                logger.error(f"Automax3 login error: {resp.status_code} {resp.text}")
                return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Automax3 login request failed: {e}")
            return None

    def ensure_authenticated(self) -> bool:
        if not self.token:
            result = self.client_login()
            return result is not None
        return True

    def get_classifications(self) -> Optional[list]:
        if self._classifications_cache:
            return self._classifications_cache
        url = f"{self.base_url}/api/classifications/hierarchy"
        if not self.ensure_authenticated():
            return None
        self.common_headers["Authorization"] = f"Bearer {self.token}"
        try:
            resp = requests.get(url, headers=self.common_headers)
            if resp.status_code == 200:
                data = resp.json()
                self._classifications_cache = data.get("data", {}).get("hierarchy", [])
                return self._classifications_cache
            else:
                logger.error(f"Automax3 get_classifications error: {resp.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Automax3 get_classifications failed: {e}")
            return None

    def get_locations(self) -> Optional[list]:
        if self._locations_cache:
            return self._locations_cache
        url = f"{self.base_url}/api/locations/hierarchy"
        if not self.ensure_authenticated():
            return None
        self.common_headers["Authorization"] = f"Bearer {self.token}"
        try:
            resp = requests.get(url, headers=self.common_headers)
            if resp.status_code == 200:
                data = resp.json()
                self._locations_cache = data.get("data", {}).get("hierarchy", [])
                return self._locations_cache
            else:
                logger.error(f"Automax3 get_locations error: {resp.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Automax3 get_locations failed: {e}")
            return None

    def get_classification_id_by_name(self, name: str) -> Optional[str]:
        classifications = self.get_classifications()
        if not classifications:
            return None
        for c in classifications:
            if c.get("en_name", "").lower() == name.lower():
                return c.get("id")
        return None

    def get_location_id_by_name(self, name: str) -> Optional[str]:
        locations = self.get_locations()
        if not locations:
            return None
        for loc in locations:
            if loc.get("en_name", "").lower() == name.lower():
                return loc.get("id")
        return None

    def get_classification_names(self) -> list:
        classifications = self.get_classifications()
        if not classifications:
            return []
        return [c.get("en_name", "") for c in classifications if c.get("en_name")]

    def get_location_names(self) -> list:
        locations = self.get_locations()
        if not locations:
            return []
        return [loc.get("en_name", "") for loc in locations if loc.get("en_name")]

    def create_incident(
        self,
        caller_name: str,
        classification_id: str,
        location_id: str,
        attachment_id: str = "",
        coordinates: dict = None,
        description: str = "",
        criticality: str = "LOW",
    ) -> Optional[dict]:
        """Create an incident in Automax3."""
        url = (
            f"{self.base_url}/api/compose/namespace/431842611944685569"
            f"/module/431842611943440385/record/"
        )
        if not self.ensure_authenticated():
            return None
        self.common_headers["Authorization"] = f"Bearer {self.token}"
        coordinates_str = json.dumps({"coordinates": coordinates}) if coordinates else ""
        body = {
            "meta": {},
            "records": [],
            "values": [
                {"name": "Channel",              "value": "Chatbot"},
                {"name": "Criticality",          "value": criticality},
                {"name": "Caller_name",          "value": caller_name},
                {"name": "Last_call_date",       "value": datetime.now().isoformat() + "Z"},
                {"name": "National_ID",          "value": ""},
                {"name": "Mobile_number",        "value": ""},
                {"name": "Classification",       "value": classification_id},
                {"name": "Incident_reason",      "value": ""},
                {"name": "Incident_Description", "value": description or "Incident created via voice agent"},
                {"name": "Map",                  "value": coordinates_str},
                {"name": "Primary_Location",     "value": location_id},
                {"name": "District",             "value": ""},
                {"name": "Street",               "value": ""},
                {"name": "Status",               "value": ""},
                {"name": "Assigned_To",          "value": "425635139776282625"},
                {"name": "Comments",             "value": json.dumps({
                    "created": datetime.now().isoformat() + "Z",
                    "comment": "Created via voice workflow agent",
                    "author": "voice-agent",
                    "name": "Voice Agent",
                })},
                {"name": "Attachments",          "value": attachment_id},
            ],
        }
        try:
            resp = requests.post(url, json=body, headers=self.common_headers)
            logger.info(f"Automax3 create_incident response: {resp.status_code}")
            if resp.status_code == 200:
                logger.info("Automax3: Incident created successfully!")
                return resp.json()
            else:
                logger.error(f"Automax3 create_incident error: {resp.status_code} {resp.text}")
                return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Automax3 create_incident failed: {e}")
            return None


# Global Automax3 client instance (identical to workflow_voice_agent.py)
_automax_client: Optional[Automax3Client] = None


def get_automax_client() -> Automax3Client:
    global _automax_client
    if _automax_client is None:
        _automax_client = Automax3Client()
        _automax_client.client_login()
    return _automax_client


# ─────────────────────────────────────────────────────────────
# WorkflowContext  (VERBATIM from workflow_voice_agent.py)
# ─────────────────────────────────────────────────────────────

class WorkflowContext:
    """Holds workflow context parsed from session metadata."""

    def __init__(self, metadata: dict):
        self.session_id       = metadata.get("session_id", "unknown")
        self.workflow         = metadata.get("workflow", {})
        self.backend_url      = metadata.get("backend_url", "http://localhost:8000")
        self.greeting_message = metadata.get("greeting_message")
        self.agent_id         = metadata.get("agent_id")
        self.company_id       = metadata.get("company_id")

        logger.info(f"[WORKFLOW CONTEXT] workflow type: {type(self.workflow)}")
        logger.info(f"[WORKFLOW CONTEXT] workflow keys: "
                    f"{list(self.workflow.keys()) if isinstance(self.workflow, dict) else 'NOT A DICT'}")

        if isinstance(self.workflow, str):
            try:
                self.workflow = json.loads(self.workflow)
                logger.info(f"[WORKFLOW CONTEXT] Parsed workflow from string, "
                            f"keys: {list(self.workflow.keys())}")
            except Exception:
                logger.error("[WORKFLOW CONTEXT] Failed to parse workflow string")
                self.workflow = {}

        self.workflow_name        = self.workflow.get("name", "Workflow")
        self.workflow_description = self.workflow.get("description", "")
        logger.info(f"[WORKFLOW CONTEXT] Workflow name: {self.workflow_name}")

        self.services    = self._extract_services()
        self.input_steps = self._extract_input_steps()

    def _get_nodes(self) -> list:
        visual_steps = self.workflow.get("visual_steps", {})
        if isinstance(visual_steps, str):
            try:
                visual_steps = json.loads(visual_steps)
                logger.info("[WORKFLOW CONTEXT] Parsed visual_steps from string")
            except Exception:
                visual_steps = {}
        nodes = visual_steps.get("nodes", []) if isinstance(visual_steps, dict) else []
        logger.info(f"[WORKFLOW CONTEXT] visual_steps.nodes count: {len(nodes)}")
        if nodes:
            return nodes
        root_nodes = self.workflow.get("nodes", [])
        logger.info(f"[WORKFLOW CONTEXT] root nodes count: {len(root_nodes)}")
        return root_nodes

    def _extract_services(self) -> list:
        for node in self._get_nodes():
            if node.get("type") == "prompt":
                options = node.get("data", {}).get("params", {}).get("options", "")
                if options:
                    return [opt.strip() for opt in options.split(",")]
        return []

    def _extract_input_steps(self) -> list:
        steps = []
        for node in self._get_nodes():
            node_type = node.get("type", "")
            node_id   = node.get("id", "")
            data      = node.get("data", {})
            params    = data.get("params", {})
            if node_type == "prompt":
                steps.append({
                    "node_id":       node_id,
                    "type":          "prompt",
                    "variable_name": params.get("save_to_variable", ""),
                    "prompt_text":   params.get("prompt_text", ""),
                    "options":       params.get("options", ""),
                    "label":         data.get("label", ""),
                })
            elif node_type == "listen":
                expected_type = params.get("expected_input_type", "text")
                steps.append({
                    "node_id":       node_id,
                    "type":          "listen",
                    "variable_name": params.get("save_to_variable", ""),
                    "input_type":    expected_type,
                    "requires_form": expected_type in ["attachment", "location"],
                    "label":         data.get("label", ""),
                })
            elif node_type == "code":
                steps.append({
                    "node_id":          node_id,
                    "type":             "code",
                    "label":            data.get("label", ""),
                    "return_variables": data.get("return_variables", []),
                    "arguments":        data.get("arguments", []),
                })
        return steps

    def get_variables_to_collect(self) -> list:
        variables = []
        for step in self.input_steps:
            if step.get("type") in ["prompt", "listen"] and step.get("variable_name"):
                variables.append({
                    "name":         step["variable_name"],
                    "type":         step.get("input_type", "text"),
                    "requires_form": step.get("requires_form", False),
                })
        return variables

    def _build_workflow_graph(self) -> str:
        nodes = self._get_nodes()
        edges = self._get_edges()
        if not nodes:
            return "No workflow steps defined."

        node_type_map  = {}
        node_label_map = {}
        for n in nodes:
            nid  = n.get("id")
            data = n.get("data", {})
            params = data.get("params", {})
            node_type_map[nid] = n.get("type", "unknown")
            node_label_map[nid] = (
                data.get("label") or
                data.get("output_value", "")[:50] or
                params.get("prompt_text", "")[:50] or
                n.get("type", "unknown")
            )

        adjacency = defaultdict(list)
        for e in edges:
            adjacency[e.get("source")].append(
                (e.get("sourceHandle", "output"), e.get("target"))
            )

        start_node = next(
            (nid for nid, ntype in node_type_map.items() if ntype == "start"),
            nodes[0].get("id") if nodes else None,
        )

        visited, queue, bfs_order = set(), deque([start_node] if start_node else []), []
        while queue:
            n = queue.popleft()
            if n in visited or n is None:
                continue
            visited.add(n)
            bfs_order.append(n)
            for _, child in adjacency.get(n, []):
                if child not in visited:
                    queue.append(child)

        lines = ["=== EXECUTION ORDER ==="]
        for i, nid in enumerate(bfs_order[:30]):
            ntype = node_type_map.get(nid, "unknown")
            label = node_label_map.get(nid, "")[:40]
            if ntype == "start":
                lines.append(f"{i+1}. START")
            elif ntype == "response":
                lines.append(f'{i+1}. SAY: "{label}"')
            elif ntype == "prompt":
                lines.append(f'{i+1}. ASK: "{label}"')
            elif ntype == "listen":
                lines.append(f"{i+1}. LISTEN for input")
            elif ntype == "condition":
                lines.append(f"{i+1}. CHECK: {label}")
            elif ntype == "code":
                lines.append(f"{i+1}. EXECUTE: {label}")
            elif ntype == "tool":
                lines.append(f"{i+1}. TOOL: {label}")
            else:
                lines.append(f"{i+1}. {ntype.upper()}: {label}")

        lines.append("\n=== TRANSITIONS ===")
        edge_count = 0
        for src, neighbors in adjacency.items():
            if edge_count >= 20:
                lines.append("... (more transitions)")
                break
            for handle, tgt in neighbors:
                lines.append(
                    f"- {node_label_map.get(src, src)[:25]} "
                    f"--[{handle}]--> "
                    f"{node_label_map.get(tgt, tgt)[:25]}"
                )
                edge_count += 1
        return "\n".join(lines)

    def _get_edges(self) -> list:
        edges = self.workflow.get("visual_steps", {})
        if isinstance(edges, dict):
            e = edges.get("edges", [])
            if e:
                return e
        return self.workflow.get("edges", [])

    def generate_system_prompt(self) -> str:
        """Generate a system prompt for EPM 940 assistant.
        VERBATIM from workflow_voice_agent.py."""
        return EPM_940_REALTIME_SYSTEM_PROMPT


# ─────────────────────────────────────────────────────────────
# Global session state
# (same pattern as workflow_voice_agent.py — single active session per process)
# ─────────────────────────────────────────────────────────────

_workflow_ctx: Optional[WorkflowContext] = None

# Replaces _current_room (LiveKit room object)
_current_client_ws = None    # WebSocket to the browser
_current_openai_ws = None    # WebSocket to OpenAI Realtime

# Same as original — set when browser sends FORM_DONE / FORM_SUBMITTED
_form_event: asyncio.Event = asyncio.Event()

# New — set by terminate_call() to cleanly end the session
_terminate_event: asyncio.Event = asyncio.Event()

# HTTP client for backend calls (identical to original)
_http_client: Optional[httpx.AsyncClient] = None


async def get_http_client() -> httpx.AsyncClient:
    """Get or create HTTP client. (identical to workflow_voice_agent.py)"""
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(timeout=30.0)
    return _http_client


def _normalize_json_string_payload(data: Any) -> str:
    """Convert tool payloads into the JSON string format expected by the backend API."""
    if isinstance(data, str):
        return data
    if isinstance(data, dict):
        return json.dumps(data)
    raise TypeError(f"Unsupported payload type: {type(data).__name__}")


# ─────────────────────────────────────────────────────────────
# Tool functions
# Same logic as workflow_voice_agent.py.
# Only change: LiveKit room calls → WebSocket calls.
# ─────────────────────────────────────────────────────────────

async def request_server_logic(action: str, data: str = "{}") -> str:
    """
    Call the backend server to execute workflow actions.

    Args:
        action: Action to execute - 'request_files' for file upload,
                'update_data' to save data, 'create_incident' to create record,
                or 'request_handoff' to transfer to human agent
        data: JSON string of data collected from the conversation

    Returns:
        Response from the server
    """
    global _workflow_ctx
    if _workflow_ctx is None:
        return "SERVER_ERROR: Workflow context not initialized"

    try:
        normalized_data = _normalize_json_string_payload(data)
    except TypeError as e:
        return f"SERVER_ERROR: {str(e)}"

    logger.info(f"Calling backend: action={action}, data={normalized_data}")
    try:
        client = await get_http_client()
        response = await client.post(
            f"{_workflow_ctx.backend_url}/api/v1/voice-workflow/step",
            json={"session_id": _workflow_ctx.session_id, "action": action, "data": normalized_data},
        )
        result = response.json()
        logger.info(f"Backend response: {result}")

        if result.get("status") == "waiting_for_input":
            url = result.get("url", "")
            return (f"SENT_FORM: {url}. Tell the user: I've sent you a link to upload the "
                    "required files. Please check it and let me know when you're done.")
        elif result.get("status") == "success":
            return f"SERVER_SUCCESS: {result.get('message', 'Action completed successfully')}"
        elif result.get("status") == "error":
            return f"SERVER_ERROR: {result.get('message', 'An error occurred')}"
        else:
            return f"SERVER_RESPONSE: {result.get('message', 'Action processed')}"
    except Exception as e:
        logger.error(f"Backend call failed: {e}")
        return f"SERVER_ERROR: Failed to communicate with server - {str(e)}"


async def store_collected_data(data: str) -> str:
    """
    Store data collected from the user during conversation.
    Call this after the user answers each question with the correct variable name.

    Args:
        data: JSON string mapping variable name to value, e.g. '{"user_name": "John"}'

    Returns:
        Confirmation that data was stored
    """
    global _workflow_ctx
    if _workflow_ctx is None:
        return "ERROR: Session not initialized"

    try:
        normalized_data = _normalize_json_string_payload(data)
    except TypeError as e:
        return f"ERROR: {str(e)}"

    logger.info(f"Storing collected data: {normalized_data}")
    try:
        client = await get_http_client()
        response = await client.post(
            f"{_workflow_ctx.backend_url}/api/v1/voice-workflow/step",
            json={"session_id": _workflow_ctx.session_id, "action": "store_data", "data": normalized_data},
        )
        result = response.json()
        logger.info(f"Store data response: {result}")
        if result.get("status") == "success":
            return "DATA_STORED: Successfully saved. Continue with the next question."
        else:
            return f"DATA_ERROR: {result.get('message', 'Failed to store data')}"
    except Exception as e:
        logger.error(f"Store data failed: {e}")
        return f"ERROR: {str(e)}"


async def request_form_input(field_type: str, variable_name: str) -> str:
    """
    Request form input for data that cannot be collected via voice (files, images, location).
    After calling this, tell the user to check the chat for a form link.

    Args:
        field_type: Type of input needed - 'attachment' for files/images, 'location' for GPS
        variable_name: The variable name to store the result

    Returns:
        Form URL if successful, or status message
    """
    global _workflow_ctx
    if _workflow_ctx is None:
        return "ERROR: Session not initialized"

    logger.info(f"Requesting form input: type={field_type}, variable={variable_name}")
    try:
        client = await get_http_client()
        response = await client.post(
            f"{_workflow_ctx.backend_url}/api/v1/voice-workflow/step",
            json={
                "session_id": _workflow_ctx.session_id,
                "action": "request_form",
                "data": json.dumps({"field_type": field_type, "variable_name": variable_name}),
            },
        )
        result = response.json()
        logger.info(f"Request form response: {result}")
        if result.get("status") == "waiting_for_input":
            return (f"FORM_SENT: A form has been sent to the chat. Tell the user: "
                    f"'I've sent a form to your chat for uploading {field_type}. "
                    "Please fill it out and let me know when done.'")
        else:
            return f"FORM_ERROR: {result.get('message', 'Failed to send form')}"
    except Exception as e:
        logger.error(f"Request form failed: {e}")
        return f"ERROR: {str(e)}"


async def execute_workflow_step(step_id: str, step_type: str = "code") -> str:
    """
    Execute a workflow processing step on the server (code execution, HTTP requests, etc).
    Say "Let me process that for you" before calling this.

    Args:
        step_id:   The node ID of the step to execute
        step_type: Type of step - "code", "http_request", or "tool"

    Returns:
        Result of the execution
    """
    global _workflow_ctx
    if _workflow_ctx is None:
        return "ERROR: Session not initialized"

    logger.info(f"Executing workflow step: id={step_id}, type={step_type}")
    try:
        client = await get_http_client()
        response = await client.post(
            f"{_workflow_ctx.backend_url}/api/v1/voice-workflow/step",
            json={
                "session_id": _workflow_ctx.session_id,
                "action": "execute_step",
                "data": json.dumps({"step_id": step_id, "step_type": step_type}),
            },
        )
        result = response.json()
        logger.info(f"Execute step response: {result}")
        if result.get("status") == "success":
            output = result.get("output", {})
            return (f"STEP_COMPLETED: Processing done. "
                    f"Result: {json.dumps(output) if output else 'Success'}. "
                    "Continue to the next step.")
        elif result.get("status") == "processing":
            return "STEP_PROCESSING: Still processing. Wait a moment and check again."
        else:
            return f"STEP_ERROR: {result.get('message', 'Step execution failed')}"
    except Exception as e:
        logger.error(f"Execute step failed: {e}")
        return f"ERROR: {str(e)}"


# ── Automax3 function tools (identical to workflow_voice_agent.py) ────────────

async def get_available_classifications() -> str:
    """
    Get list of available incident classifications.
    These are the hardcoded classification options for incident reporting.
    """
    classifications = ["Manholes", "Street Lights", "Street Furniture", "Potholes", "Barriers"]
    logger.info(f"Returning hardcoded classifications: {classifications}")
    return f"CLASSIFICATIONS_AVAILABLE: {', '.join(classifications)}"


async def get_available_locations() -> str:
    """
    Get list of available locations.
    These are the hardcoded location options for incident reporting.
    """
    locations = ["Dammam", "Dammam East", "Dammam West"]
    logger.info(f"Returning hardcoded locations: {locations}")
    return f"LOCATIONS_AVAILABLE: {', '.join(locations)}"


async def trigger_form_popup(
    caller_name: str,
    classification: str,
    location: str,
    description: str = "",
    criticality: str = "LOW",
) -> str:
    """
    Trigger the popup form on the frontend to confirm incident details.
    MUST call this after collecting all incident information from the user.

    Sends an OPEN_FORM JSON message to the browser via WebSocket.
    (Replaces LiveKit room.local_participant.publish_data)
    """
    global _current_client_ws, _form_event

    logger.info(f"Triggering form popup: caller={caller_name}, "
                f"classification={classification}, location={location}")

    try:
        _form_event.clear()

        data_payload = {
            "type":           "OPEN_FORM",
            "caller_name":    caller_name   or "N/A",
            "classification": classification or "N/A",
            "location":       location       or "N/A",
            "description":    description    or "N/A",
            "criticality":    criticality    or "LOW",
        }
        json_msg = json.dumps(data_payload)

        if _current_client_ws:
            # Send the signal 3 times with a short delay (same burst pattern as original)
            for i in range(3):
                # 1. Plain string (legacy/simple — same as original's plain string publish)
                await _current_client_ws.send("OPEN_FORM")
                # 2. Full JSON payload (same as original's JSON publish_data)
                await _current_client_ws.send(json_msg)
                logger.info(f"Burst {i+1} sent (Plain + JSON)")
                await asyncio.sleep(0.4)

            return "FORM_SIGNAL_SENT: Signal sent with AI data payload. Proceeding to goodbye."
        else:
            logger.warning("_current_client_ws is None — cannot send OPEN_FORM signal")
            return "FORM_TRIGGERED: Incident details saved, but could not open the popup."

    except Exception as e:
        logger.error(f"Failed to trigger form popup: {e}")
        return f"FORM_ERROR: Could not trigger the form. Error: {str(e)}."


async def terminate_call() -> str:
    """
    Disconnect the agent and terminate the voice call session.
    Call this ONLY after saying goodbye to the user.

    Replaces LiveKit's room.disconnect() — sets _terminate_event which
    causes handle_session() to close both WebSockets cleanly.
    """
    logger.info("Agent requested call termination. Waiting 3s for audio to clear...")

    async def delayed_disconnect():
        # Final OPEN_FORM burst as a safeguard (same as original)
        if _current_client_ws:
            try:
                await _current_client_ws.send("OPEN_FORM")
                await _current_client_ws.send(json.dumps({"type": "OPEN_FORM"}))
                logger.info("FINAL signal burst sent before disconnect")
            except Exception as e:
                logger.warning(f"Final signal fail: {e}")

        await asyncio.sleep(3.0)
        logger.info("Setting terminate event — closing session...")
        _terminate_event.set()

    asyncio.create_task(delayed_disconnect())
    return "CALL_TERMINATED: I'm ending the call now. You can finish the report on your screen. Goodbye!"


async def create_automax_incident(
    caller_name: str,
    classification_name: str,
    location_name: str,
    description: str = "",
    attachment_id: str = "",
    coordinates: str = "",
    criticality: str = "LOW",
) -> str:
    """
    Create an incident in Automax3 system.
    VERBATIM logic from workflow_voice_agent.py.
    """
    logger.info(f"Creating Automax incident: caller={caller_name}, "
                f"classification={classification_name}, location={location_name}")
    try:
        client = get_automax_client()

        classification_id = client.get_classification_id_by_name(classification_name)
        if not classification_id:
            available = client.get_classification_names()
            return (f"INCIDENT_ERROR: Classification '{classification_name}' not found. "
                    f"Available: {', '.join(available[:10])}")

        location_id = client.get_location_id_by_name(location_name)
        if not location_id:
            available = client.get_location_names()
            return (f"INCIDENT_ERROR: Location '{location_name}' not found. "
                    f"Available: {', '.join(available[:10])}")

        coords_dict = None
        if coordinates:
            try:
                coords_dict = json.loads(coordinates)
            except json.JSONDecodeError:
                logger.warning(f"Could not parse coordinates: {coordinates}")

        result = client.create_incident(
            caller_name=caller_name,
            classification_id=classification_id,
            location_id=location_id,
            attachment_id=attachment_id,
            coordinates=coords_dict,
            description=description,
            criticality=criticality,
        )

        if result:
            record_id = result.get("response", {}).get("recordID", "unknown")
            return (f"INCIDENT_CREATED: Successfully created incident #{record_id}. "
                    "Tell the user their incident has been registered and they will be contacted soon.")
        else:
            return "INCIDENT_ERROR: Failed to create incident in Automax3. Please try again."

    except Exception as e:
        logger.error(f"Create incident failed: {e}")
        return f"INCIDENT_ERROR: {str(e)}"


# ─────────────────────────────────────────────────────────────
# OpenAI Realtime tool schemas
# (replaces @llm.function_tool() decorator from LiveKit)
# ─────────────────────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "name": "request_server_logic",
        "description": (
            "Call the backend server to execute workflow actions. "
            "Actions: 'request_files' for file upload, 'update_data' to save data, "
            "'create_incident' to create a record, 'request_handoff' to transfer to human agent."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "description": "Action to execute"},
                "data":   {"type": "string", "description": "JSON string of collected data"},
            },
            "required": ["action"],
        },
    },
    {
        "type": "function",
        "name": "store_collected_data",
        "description": (
            "Store data collected from the user. "
            "This is mandatory after each answer in the EPM 940 incident flow. "
            "Pass data as a JSON string like '{\"caller_name\": \"John\"}'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "data": {"type": "string", "description": "JSON string mapping variable name to value"},
            },
            "required": ["data"],
        },
    },
    {
        "type": "function",
        "name": "request_form_input",
        "description": (
            "Request form input for data that cannot be collected via voice "
            "(files, images, location). Tell the user to check the chat for a form link."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "field_type":    {"type": "string", "description": "'attachment' or 'location'"},
                "variable_name": {"type": "string", "description": "Variable name to store the result"},
            },
            "required": ["field_type", "variable_name"],
        },
    },
    {
        "type": "function",
        "name": "execute_workflow_step",
        "description": (
            "Execute a workflow processing step on the server. "
            "Say 'Let me process that for you' before calling this."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "step_id":   {"type": "string", "description": "Node ID of the step to execute"},
                "step_type": {"type": "string", "description": "'code', 'http_request', or 'tool'"},
            },
            "required": ["step_id"],
        },
    },
    {
        "type": "function",
        "name": "get_available_classifications",
        "description": (
            "Get the hardcoded incident classifications. "
            "Call this before presenting classification options to the user."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "type": "function",
        "name": "get_available_locations",
        "description": (
            "Get the hardcoded locations. "
            "Call this before presenting location options to the user."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "type": "function",
        "name": "trigger_form_popup",
        "description": (
            "Trigger the popup form on the frontend to confirm incident details. "
            "MUST call this after collecting ALL incident information and before the final goodbye. "
            "Sends OPEN_FORM signal to the browser."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "caller_name":    {"type": "string"},
                "classification": {"type": "string", "description": "Manholes / Street Lights / etc."},
                "location":       {"type": "string", "description": "Dammam / Dammam East / Dammam West"},
                "description":    {"type": "string"},
                "criticality":    {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH", "CRITICAL"]},
            },
            "required": ["caller_name", "classification", "location"],
        },
    },
    {
        "type": "function",
        "name": "terminate_call",
        "description": (
            "Disconnect the agent and end the voice call. "
            "Call ONLY after saying the exact final goodbye sentence."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "type": "function",
        "name": "create_automax_incident",
        "description": (
            "Create an incident in Automax3. "
            "Collect caller_name, classification_name, and location_name BEFORE calling."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "caller_name":         {"type": "string"},
                "classification_name": {"type": "string", "description": "Must match available classifications"},
                "location_name":       {"type": "string", "description": "Must match available locations"},
                "description":         {"type": "string"},
                "attachment_id":       {"type": "string"},
                "coordinates":         {"type": "string", "description": "GPS coords as JSON string"},
                "criticality":         {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH", "CRITICAL"]},
            },
            "required": ["caller_name", "classification_name", "location_name"],
        },
    },
]


# ─────────────────────────────────────────────────────────────
# Tool dispatcher
# Replaces LiveKit's automatic tool discovery via @llm.function_tool()
# ─────────────────────────────────────────────────────────────

async def _dispatch_tool(name: str, args: dict) -> str:
    """Execute a tool by name and return its string result."""
    logger.info(f"[TOOL] Executing: {name}({json.dumps(args)[:120]})")
    try:
        if name == "request_server_logic":
            return await request_server_logic(**args)
        elif name == "store_collected_data":
            return await store_collected_data(**args)
        elif name == "request_form_input":
            return await request_form_input(**args)
        elif name == "execute_workflow_step":
            return await execute_workflow_step(**args)
        elif name == "get_available_classifications":
            return await get_available_classifications()
        elif name == "get_available_locations":
            return await get_available_locations()
        elif name == "trigger_form_popup":
            return await trigger_form_popup(**args)
        elif name == "terminate_call":
            return await terminate_call()
        elif name == "create_automax_incident":
            return await create_automax_incident(**args)
        else:
            logger.warning(f"[TOOL] Unknown tool: {name}")
            return f"ERROR: Tool '{name}' not found."
    except Exception as e:
        logger.error(f"[TOOL] {name} raised: {e}", exc_info=True)
        return f"ERROR: Tool execution failed — {str(e)}"


# ─────────────────────────────────────────────────────────────
# Audio utilities
# ─────────────────────────────────────────────────────────────

async def _maybe_await(result: Any) -> Any:
    """Await callback results only when the callback is async."""
    if asyncio.iscoroutine(result) or hasattr(result, "__await__"):
        return await result
    return result


class RealtimeVoiceAgent:
    """
    Reusable OpenAI Realtime websocket client for FastAPI voice endpoints.
    """

    def __init__(
        self,
        *,
        system_prompt: str,
        greeting: str,
        voice: Optional[str] = None,
        tools: Optional[list[dict[str, Any]]] = None,
        tool_handlers: Optional[dict[str, Callable[..., Any]]] = None,
        on_user_transcript: Optional[Callable[[str], Any]] = None,
        on_agent_transcript: Optional[Callable[[str], Any]] = None,
        on_audio_chunk: Optional[Callable[[bytes], Any]] = None,
        on_audio_done: Optional[Callable[[], Any]] = None,
        on_tool_result: Optional[Callable[[str, str], Any]] = None,
        model: Optional[str] = None,
    ) -> None:
        self.system_prompt = system_prompt
        self.greeting = greeting
        self.voice = voice or REALTIME_VOICE
        self.tools = tools or []
        self.tool_handlers = tool_handlers or {}
        self.on_user_transcript = on_user_transcript
        self.on_agent_transcript = on_agent_transcript
        self.on_audio_chunk = on_audio_chunk
        self.on_audio_done = on_audio_done
        self.on_tool_result = on_tool_result
        self.model = model or REALTIME_MODEL

        self.ws: Optional[ClientConnection] = None
        self._audio_buffer = bytearray()
        self._pending_tool_calls: list[dict[str, str]] = []

    async def connect(self, api_key: Optional[str] = None) -> bool:
        """Connect to OpenAI Realtime and configure the session."""
        resolved_api_key = api_key or OPENAI_API_KEY
        if not resolved_api_key:
            logger.error("[RealtimeVoiceAgent] No OpenAI API key configured")
            return False

        url = f"{OPENAI_REALTIME_URL}?model={self.model}"
        headers = {
            "Authorization": f"Bearer {resolved_api_key}",
            "OpenAI-Beta": "realtime=v1",
        }

        try:
            self.ws = await websockets.connect(
                url,
                additional_headers=headers,
                ping_interval=20,
                ping_timeout=10,
            )
            raw = await asyncio.wait_for(self.ws.recv(), timeout=10.0)
            event = json.loads(raw)
            if event.get("type") != "session.created":
                logger.warning(
                    f"[RealtimeVoiceAgent] Unexpected first event: {event.get('type')}"
                )

            await self._configure_session()
            return True
        except asyncio.TimeoutError:
            logger.error("[RealtimeVoiceAgent] Timeout waiting for session.created")
            await self.disconnect()
            return False
        except Exception as exc:
            logger.error(f"[RealtimeVoiceAgent] Connection failed: {exc}", exc_info=True)
            await self.disconnect()
            return False

    async def _configure_session(self) -> None:
        if self.ws is None:
            raise RuntimeError("RealtimeVoiceAgent is not connected")

        payload = {
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "voice": self.voice,
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {"model": "whisper-1"},
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500,
                },
                "instructions": self.system_prompt,
            },
        }
        if self.tools:
            payload["session"]["tools"] = self.tools
            payload["session"]["tool_choice"] = "auto"

        await self.ws.send(json.dumps(payload))
        logger.info(
            f"[RealtimeVoiceAgent] Session configured with voice={self.voice} tools={len(self.tools)}"
        )

    async def send_greeting(self) -> None:
        if self.ws is None:
            return
        await self.ws.send(json.dumps({
            "type": "response.create",
            "response": {
                "modalities": ["text", "audio"],
                "instructions": self.greeting,
            },
        }))

    async def send_audio(self, pcm16: bytes) -> None:
        if self.ws is None or not pcm16:
            return
        await self.ws.send(json.dumps({
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(pcm16).decode("utf-8"),
        }))

    async def commit_audio(self) -> None:
        if self.ws is None:
            return
        await self.ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

    async def cancel_response(self) -> None:
        if self.ws is None:
            return
        self._audio_buffer.clear()
        await self.ws.send(json.dumps({"type": "response.cancel"}))

    async def _run_tool_handler(self, name: str, arguments: dict[str, Any]) -> str:
        handler = self.tool_handlers.get(name)
        if handler is None:
            logger.warning(f"[RealtimeVoiceAgent] No handler registered for tool '{name}'")
            return f"ERROR: Tool '{name}' not found."

        try:
            result = handler(**arguments)
            result = await _maybe_await(result)
            if not isinstance(result, str):
                result = json.dumps(result, default=str)
        except Exception as exc:
            logger.error(f"[RealtimeVoiceAgent] Tool '{name}' failed: {exc}", exc_info=True)
            result = f"ERROR: Tool execution failed - {exc}"

        if self.on_tool_result:
            await _maybe_await(self.on_tool_result(name, result))
        return result

    async def _flush_tool_calls(self) -> None:
        if self.ws is None or not self._pending_tool_calls:
            return

        calls = self._pending_tool_calls.copy()
        self._pending_tool_calls.clear()

        for call in calls:
            raw_args = call.get("arguments", "{}")
            try:
                arguments = json.loads(raw_args) if raw_args else {}
            except json.JSONDecodeError:
                arguments = {}

            result = await self._run_tool_handler(call.get("name", ""), arguments)
            await self.ws.send(json.dumps({
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": call.get("call_id", ""),
                    "output": result,
                },
            }))

        await self.ws.send(json.dumps({"type": "response.create"}))

    async def process_events(self) -> None:
        """Process OpenAI Realtime events until the websocket closes."""
        if self.ws is None:
            return

        try:
            async for raw in self.ws:
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                event_type = event.get("type", "")

                if event_type == "response.audio.delta":
                    delta = event.get("delta", "")
                    if delta:
                        self._audio_buffer.extend(base64.b64decode(delta))

                elif event_type == "response.audio.done":
                    if self._audio_buffer and self.on_audio_chunk:
                        wav = pcm16_to_wav(bytes(self._audio_buffer))
                        self._audio_buffer.clear()
                        await _maybe_await(self.on_audio_chunk(wav))
                    if self.on_audio_done:
                        await _maybe_await(self.on_audio_done())

                elif event_type == "response.audio_transcript.done":
                    transcript = event.get("transcript", "").strip()
                    if transcript and self.on_agent_transcript:
                        await _maybe_await(self.on_agent_transcript(transcript))

                elif event_type == "conversation.item.input_audio_transcription.completed":
                    transcript = event.get("transcript", "").strip()
                    if transcript and self.on_user_transcript:
                        await _maybe_await(self.on_user_transcript(transcript))

                elif event_type == "input_audio_buffer.speech_started":
                    self._audio_buffer.clear()

                elif event_type == "response.output_item.done":
                    item = event.get("item", {})
                    if item.get("type") == "function_call":
                        self._pending_tool_calls.append({
                            "call_id": item.get("call_id", ""),
                            "name": item.get("name", ""),
                            "arguments": item.get("arguments", "{}"),
                        })

                elif event_type == "response.done":
                    await self._flush_tool_calls()

                elif event_type == "error":
                    err = event.get("error", {})
                    logger.error(
                        f"[RealtimeVoiceAgent] API error: {err.get('type')} - {err.get('message')}"
                    )
        except websockets.exceptions.ConnectionClosed as exc:
            logger.info(f"[RealtimeVoiceAgent] WebSocket closed: {exc}")
        finally:
            self._audio_buffer.clear()

    async def disconnect(self) -> None:
        """Close the Realtime websocket connection."""
        if self.ws is None:
            return
        try:
            await self.ws.close()
        except Exception as exc:
            logger.error(f"[RealtimeVoiceAgent] Error closing websocket: {exc}")
        finally:
            self.ws = None


def pcm16_to_wav(
    pcm_data: bytes,
    sample_rate: int = 24000,
    num_channels: int = 1,
    bits_per_sample: int = 16,
) -> bytes:
    """Wrap raw PCM16 bytes in a WAV container for browser playback."""
    data_size  = len(pcm_data)
    byte_rate  = sample_rate * num_channels * (bits_per_sample // 8)
    block_align = num_channels * (bits_per_sample // 8)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", data_size + 36, b"WAVE",
        b"fmt ", 16, 1,
        num_channels, sample_rate, byte_rate, block_align, bits_per_sample,
        b"data", data_size,
    )
    return header + pcm_data


async def convert_webm_to_pcm16_24k(webm_data: bytes) -> bytes:
    """Convert browser WebM/Opus audio to PCM16 24 kHz mono using PyAV."""
    if not webm_data:
        return b""
    try:
        import av
        input_buf  = io.BytesIO(webm_data)
        output_buf = io.BytesIO()
        with av.open(input_buf) as container:
            streams = [s for s in container.streams if s.type == "audio"]
            if not streams:
                logger.warning("[Audio] No audio stream in WebM data")
                return b""
            resampler = av.AudioResampler(format="s16", layout="mono", rate=24000)
            for frame in container.decode(streams[0]):
                for resampled in resampler.resample(frame):
                    output_buf.write(bytes(resampled.planes[0]))
            for resampled in resampler.resample(None):
                output_buf.write(bytes(resampled.planes[0]))
        result = output_buf.getvalue()
        logger.debug(f"[Audio] {len(webm_data)} B WebM → {len(result)} B PCM16@24kHz")
        return result
    except Exception as e:
        logger.error(f"[Audio] WebM→PCM16 conversion failed: {e}", exc_info=True)
        return b""


# ─────────────────────────────────────────────────────────────
# Data-message handler (replaces ctx.room.on("data_received"))
# ─────────────────────────────────────────────────────────────

def _handle_client_text_message(raw: str) -> None:
    """
    Handle text frames from the browser WebSocket.
    Equivalent to on_data_received() handler in workflow_voice_agent.py.
    """
    try:
        payload = json.loads(raw)
        msg_type = payload.get("type", "")
        if msg_type in ["FORM_SUBMITTED", "FORM_DONE"]:
            logger.info(f"Form submission signal (JSON): {msg_type}")
            _form_event.set()
            return
    except Exception:
        pass

    stripped = raw.strip()
    if stripped in ["FORM_DONE", "OPEN_FORM_DONE", "FORM_SUBMITTED"]:
        logger.info(f"Form submission signal (String): {stripped}")
        _form_event.set()


# ─────────────────────────────────────────────────────────────
# OpenAI Realtime WebSocket helpers
# ─────────────────────────────────────────────────────────────

async def _openai_connect() -> Optional[ClientConnection]:
    """Open WebSocket to OpenAI Realtime API. Returns ws or None on failure."""
    url = f"{OPENAI_REALTIME_URL}?model={REALTIME_MODEL}"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "OpenAI-Beta": "realtime=v1",
    }
    try:
        ws = await websockets.connect(url, additional_headers=headers,
                                      ping_interval=20, ping_timeout=10)
        # Wait for session.created
        raw   = await asyncio.wait_for(ws.recv(), timeout=10.0)
        event = json.loads(raw)
        if event.get("type") == "session.created":
            session_id = event.get("session", {}).get("id")
            logger.info(f"[OpenAI] Session created: {session_id}")
        else:
            logger.warning(f"[OpenAI] Unexpected first event: {event.get('type')}")
        return ws
    except asyncio.TimeoutError:
        logger.error("[OpenAI] Timeout waiting for session.created")
        return None
    except Exception as e:
        logger.error(f"[OpenAI] Connection failed: {e}", exc_info=True)
        return None


async def _openai_configure_session(
    ws: ClientConnection,
    system_prompt: str,
) -> None:
    """Send session.update with VAD, voice, modalities, tools, and system prompt."""
    config = {
        "type": "session.update",
        "session": {
            "modalities":   ["text", "audio"],
            "voice":        REALTIME_VOICE,
            "input_audio_format":  "pcm16",
            "output_audio_format": "pcm16",
            "input_audio_transcription": {"model": "whisper-1"},
            "turn_detection": {
                "type":                "server_vad",
                "threshold":           0.5,
                "prefix_padding_ms":   300,
                "silence_duration_ms": 500,
            },
            "instructions": system_prompt,
            "tools":        TOOLS,
            "tool_choice":  "auto",
        },
    }
    await ws.send(json.dumps(config))
    logger.info("[OpenAI] Session configured (VAD + tools + system prompt)")


async def _openai_send_greeting(
    ws: ClientConnection,
    greeting: str,
) -> None:
    """Ask the model to speak the greeting (replaces session.generate_reply)."""
    await ws.send(json.dumps({
        "type": "response.create",
        "response": {
            "modalities":   ["text", "audio"],
            "instructions": greeting,
        },
    }))
    logger.info("[OpenAI] Greeting response requested")


async def _execute_pending_tool_calls(
    ws: ClientConnection,
    calls: list,
) -> None:
    """
    Execute all queued tool calls, send results back to OpenAI, then
    request a continuation response. Equivalent to LiveKit's automatic
    tool dispatch triggered by the @llm.function_tool() decorator.
    """
    for call in calls:
        call_id = call.get("call_id", "")
        name    = call.get("name", "")
        raw_args = call.get("arguments", "{}")

        try:
            args = json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError:
            args = {}

        result = await _dispatch_tool(name, args)

        # Send function_call_output back to OpenAI
        await ws.send(json.dumps({
            "type": "conversation.item.create",
            "item": {
                "type":    "function_call_output",
                "call_id": call_id,
                "output":  result,
            },
        }))
        logger.info(f"[TOOL] Result sent for call_id={call_id}: {result[:100]}")

    # Ask the model to continue
    await ws.send(json.dumps({"type": "response.create"}))


# ─────────────────────────────────────────────────────────────
# Main session handler
# Equivalent to entrypoint(ctx: JobContext) in workflow_voice_agent.py
# ─────────────────────────────────────────────────────────────

async def handle_session(
    client_ws: ServerConnection,
    session_id: str,
) -> None:
    """
    Handle a single voice session from start to finish.

    Mirrors entrypoint() in workflow_voice_agent.py step-by-step:
      1. Set global state (replaces ctx.room, ctx.connect())
      2. Fetch workflow from backend (identical)
      3. Build WorkflowContext (identical)
      4. Connect to OpenAI Realtime (replaces AgentSession + Agent)
      5. Set up data-message listener (replaces ctx.room.on("data_received"))
      6. Send greeting (replaces session.generate_reply())
      7. Run bidirectional audio bridge + tool loop
      8. Cleanup on disconnect or terminate_call()
    """
    global _workflow_ctx, _current_client_ws, _current_openai_ws, _form_event, _terminate_event

    logger.info(f"[SESSION] Starting session: {session_id}")

    # ── 1. Register global state ──────────────────────────────
    _current_client_ws = client_ws
    _form_event.clear()
    _terminate_event.clear()

    # ── 2. Fetch workflow from backend (identical to entrypoint) ──
    metadata         = {"session_id": session_id}
    workflow_fetched = False

    if session_id:
        backend_url = os.getenv("BACKEND_URL", BACKEND_URL)
        logger.info(f"[WORKFLOW FETCH] Fetching from: "
                    f"{backend_url}/api/v1/voice-workflow/session/{session_id}")
        try:
            async with httpx.AsyncClient(timeout=30.0) as hclient:
                resp = await hclient.get(
                    f"{backend_url}/api/v1/voice-workflow/session/{session_id}"
                )
                logger.info(f"[WORKFLOW FETCH] Response status: {resp.status_code}")
                if resp.status_code == 200:
                    session_data = resp.json()
                    if session_data.get("workflow_json"):
                        metadata["workflow"]     = session_data["workflow_json"]
                        metadata["backend_url"]  = backend_url
                        workflow_fetched         = True
                        wf_name  = session_data["workflow_json"].get("name", "Unknown")
                        vs_nodes = len(
                            session_data["workflow_json"]
                            .get("visual_steps", {}).get("nodes", [])
                        )
                        logger.info(f"[WORKFLOW FETCH] SUCCESS — {wf_name}, "
                                    f"visual_steps nodes: {vs_nodes}")
                else:
                    logger.error(f"[WORKFLOW FETCH] Failed: {resp.status_code}")
        except Exception as e:
            logger.error(f"[WORKFLOW FETCH] Exception: {e}")

    if not workflow_fetched:
        logger.error("[WORKFLOW FETCH] FAILED — No workflow data available!")

    # ── 3. Build WorkflowContext (identical) ──────────────────
    workflow_ctx = WorkflowContext(metadata)
    _workflow_ctx = workflow_ctx  # set global for tool functions

    # ── 4. Connect to OpenAI Realtime (replaces AgentSession + Agent) ──
    logger.info("[SESSION] Connecting to OpenAI Realtime...")
    openai_ws = await _openai_connect()
    if openai_ws is None:
        logger.error("[SESSION] Cannot connect to OpenAI Realtime — aborting session")
        try:
            await client_ws.send(json.dumps({"error": "Failed to connect to AI voice service"}))
            await client_ws.close(1011)
        except Exception:
            pass
        return

    _current_openai_ws = openai_ws

    system_prompt = workflow_ctx.generate_system_prompt()
    await _openai_configure_session(openai_ws, system_prompt)

    # ── 5. Log transcript events (replaces session.on("*_speech_committed")) ──
    # (handled inside the event loop below)

    # ── 6. Send greeting (replaces session.generate_reply) ───
    greeting = "Hello, welcome to EPM 940. How can I assist you today?"
    logger.info("[SESSION] Using fixed EPM 940 greeting")
    await _openai_send_greeting(openai_ws, greeting)
    logger.info("[SESSION] Greeting sent")

    # ── 7. Bidirectional audio bridge + event / tool loop ────
    audio_buffer:    bytearray    = bytearray()
    last_audio_time: Optional[float] = None
    openai_audio_buf: bytearray  = bytearray()
    pending_tool_calls: list     = []

    # ── Task A: browser → OpenAI (audio + control messages) ──
    async def forward_browser_to_openai() -> None:
        nonlocal audio_buffer, last_audio_time
        try:
            async for message in client_ws:
                if isinstance(message, bytes):
                    # Binary frame = WebM audio chunk from browser
                    last_audio_time = asyncio.get_event_loop().time()
                    audio_buffer.extend(message)
                else:
                    # Text frame = JSON control message from browser
                    _handle_client_text_message(message)
        except websockets.exceptions.ConnectionClosed:
            logger.info("[SESSION] Browser disconnected")
        except Exception as e:
            logger.error(f"[SESSION] forward_browser_to_openai error: {e}")

    # ── Task B: silence-detection audio flusher ───────────────
    async def flush_audio_buffer() -> None:
        nonlocal audio_buffer, last_audio_time
        while True:
            await asyncio.sleep(0.3)
            if not audio_buffer or last_audio_time is None:
                continue
            elapsed = asyncio.get_event_loop().time() - last_audio_time
            if elapsed < 0.5:
                continue
            # Silence threshold reached — convert and send
            chunk = bytes(audio_buffer)
            audio_buffer.clear()
            last_audio_time = None
            pcm16 = await convert_webm_to_pcm16_24k(chunk)
            if pcm16:
                audio_b64 = base64.b64encode(pcm16).decode()
                try:
                    await openai_ws.send(json.dumps({
                        "type": "input_audio_buffer.append",
                        "audio": audio_b64,
                    }))
                    await openai_ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
                    logger.debug(f"[Audio] Sent {len(pcm16)} B PCM16 to OpenAI")
                except Exception as e:
                    logger.error(f"[Audio] Failed to send audio to OpenAI: {e}")

    # ── Task C: OpenAI → browser (audio + events + tool calls) ──
    async def process_openai_events() -> None:
        nonlocal openai_audio_buf, pending_tool_calls
        try:
            async for raw in openai_ws:
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                event_type: str = event.get("type", "")

                # Suppress verbose audio-delta log spam
                if event_type != "response.audio.delta":
                    logger.debug(f"[OpenAI] Event: {event_type}")

                # ─ Audio output ──────────────────────────────
                if event_type == "response.audio.delta":
                    delta = event.get("delta", "")
                    if delta:
                        openai_audio_buf.extend(base64.b64decode(delta))

                elif event_type == "response.audio.done":
                    if openai_audio_buf:
                        wav = pcm16_to_wav(bytes(openai_audio_buf))
                        openai_audio_buf.clear()
                        try:
                            await client_ws.send(wav)
                            logger.debug(f"[Audio] Sent {len(wav)} B WAV to browser")
                        except Exception as e:
                            logger.error(f"[Audio] Failed to send WAV to browser: {e}")
                    # Signal end-of-audio to browser (same as realtime_voice.py)
                    try:
                        await client_ws.send(json.dumps({"type": "audio_end"}))
                    except Exception:
                        pass

                # ─ Transcript logging (replaces agent/user_speech_committed) ──
                elif event_type == "response.audio_transcript.done":
                    transcript = event.get("transcript", "").strip()
                    if transcript:
                        logger.info(f"[TRANSCRIPT] AGENT: {transcript}")

                elif event_type == "conversation.item.input_audio_transcription.completed":
                    transcript = event.get("transcript", "").strip()
                    if transcript:
                        logger.info(f"[TRANSCRIPT] USER: {transcript}")

                # ─ VAD interruption ──────────────────────────
                elif event_type == "input_audio_buffer.speech_started":
                    if openai_audio_buf:
                        logger.debug("[Audio] User interrupted — discarding audio buffer")
                        openai_audio_buf.clear()

                # ─ Tool calling ──────────────────────────────
                elif event_type == "response.output_item.done":
                    item = event.get("item", {})
                    if item.get("type") == "function_call":
                        pending_tool_calls.append({
                            "call_id":   item.get("call_id"),
                            "name":      item.get("name"),
                            "arguments": item.get("arguments", "{}"),
                        })
                        logger.info(f"[TOOL] Queued: {item.get('name')} (call_id={item.get('call_id')})")

                elif event_type == "response.done":
                    # Execute all queued tool calls after the full response is ready
                    if pending_tool_calls:
                        calls = pending_tool_calls.copy()
                        pending_tool_calls.clear()
                        await _execute_pending_tool_calls(openai_ws, calls)

                elif event_type == "error":
                    err = event.get("error", {})
                    logger.error(f"[OpenAI] API error: {err.get('type')} — {err.get('message')}")

        except websockets.exceptions.ConnectionClosed as e:
            logger.info(f"[OpenAI] WebSocket closed: {e}")
        except Exception as e:
            logger.error(f"[OpenAI] process_openai_events error: {e}", exc_info=True)

    # ── Task D: terminate_event watcher (replaces room.disconnect) ──
    async def watch_terminate() -> None:
        await _terminate_event.wait()
        logger.info("[SESSION] Terminate event received — closing connections")
        try:
            await openai_ws.close()
        except Exception:
            pass
        try:
            await client_ws.close()
        except Exception:
            pass

    # ── Run all tasks concurrently ────────────────────────────
    tasks = [
        asyncio.create_task(forward_browser_to_openai(), name="browser→openai"),
        asyncio.create_task(flush_audio_buffer(),         name="audio-flush"),
        asyncio.create_task(process_openai_events(),      name="openai-events"),
        asyncio.create_task(watch_terminate(),             name="terminate-watcher"),
    ]

    # Wait until the browser disconnects OR terminate_call() fires
    _, pending = await asyncio.wait(
        tasks,
        return_when=asyncio.FIRST_COMPLETED,
    )
    for task in pending:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    # ── 8. Cleanup ────────────────────────────────────────────
    try:
        await openai_ws.close()
    except Exception:
        pass
    _current_client_ws  = None
    _current_openai_ws  = None
    logger.info(f"[SESSION] Session {session_id} cleaned up")


# ─────────────────────────────────────────────────────────────
# Connection dispatcher (replaces request_fnc in workflow_voice_agent.py)
# ─────────────────────────────────────────────────────────────

async def handle_connection(
    websocket: ServerConnection,
    path: str,
) -> None:
    """
    Accept or reject incoming WebSocket connections.
    Equivalent to request_fnc() in workflow_voice_agent.py.

    Accepts:   /voice_realtime/{session_id}
    Rejects:   everything else
    """
    if not path.startswith(WS_PATH_PREFIX):
        logger.debug(f"Rejecting connection: {path!r} (not a voice_realtime path)")
        await websocket.close(1008, "Invalid path — expected /voice_realtime/{session_id}")
        return

    session_id = path[len(WS_PATH_PREFIX):]
    if not session_id:
        logger.warning("Rejecting connection: missing session_id in path")
        await websocket.close(1008, "Missing session_id")
        return

    logger.info(f"Accepting voice_realtime session: {session_id}")
    await handle_session(websocket, session_id)


# ─────────────────────────────────────────────────────────────
# Prewarm  (same purpose as prewarm() in workflow_voice_agent.py)
# VAD is built into OpenAI Realtime — no silero pre-load needed.
# ─────────────────────────────────────────────────────────────

def prewarm() -> None:
    """Prewarm: initialise Automax3 client (OpenAI Realtime has built-in VAD)."""
    logger.info("Prewarming realtime voice agent...")
    # (Automax3 init commented out matching original — uncomment if needed)
    # try:
    #     client = get_automax_client()
    #     classifications = client.get_classifications()
    #     locations       = client.get_locations()
    #     logger.info(f"Automax3: {len(classifications or [])} classifications, "
    #                 f"{len(locations or [])} locations")
    # except Exception as e:
    #     logger.warning(f"Automax3 prewarm failed (will retry on first use): {e}")
    logger.info("Prewarm complete")


# ─────────────────────────────────────────────────────────────
# Async main  (replaces cli.run_app(WorkerOptions(...)))
# ─────────────────────────────────────────────────────────────

async def main() -> None:
    """Start the WebSocket server and run forever."""
    async with websockets.serve(
        handle_connection,
        REALTIME_WS_HOST,
        REALTIME_WS_PORT,
        ping_interval=20,
        ping_timeout=10,
    ):
        logger.info(
            f"WebSocket server listening on "
            f"ws://{REALTIME_WS_HOST}:{REALTIME_WS_PORT}{WS_PATH_PREFIX}{{session_id}}"
        )
        await asyncio.Future()  # run forever (same intent as cli.run_app)


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Validate environment (mirrors workflow_voice_agent.py validation)
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY must be set in .env")

    logger.info("=" * 60)
    logger.info("Starting Realtime Voice Agent Worker...")
    logger.info("=" * 60)
    logger.info(f"WebSocket: ws://{REALTIME_WS_HOST}:{REALTIME_WS_PORT}{WS_PATH_PREFIX}{{session_id}}")
    logger.info(f"Model:     {REALTIME_MODEL}")
    logger.info(f"Voice:     {REALTIME_VOICE}")
    logger.info(f"Backend:   {BACKEND_URL}")
    logger.info("=" * 60)

    prewarm()
    asyncio.run(main())
