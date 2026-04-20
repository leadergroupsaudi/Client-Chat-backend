import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Annotated, Optional

import httpx
import requests
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    AutoSubscribe,
    JobContext,
    JobProcess,
    RoomInputOptions,
    RoomOutputOptions,
    WorkerOptions,
    cli,
    llm,
)
from livekit import rtc
from livekit.plugins import openai, silero

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("workflow-voice-agent")

# Environment configuration - use LIVEKIT_*2 to match workflow_livekit_service.py
LIVEKIT_URL = os.getenv("LIVEKIT_URL2", os.getenv("LIVEKIT_URL", "wss://your-livekit-server.livekit.cloud"))
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY2", os.getenv("LIVEKIT_API_KEY"))
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET2", os.getenv("LIVEKIT_API_SECRET"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Room name prefix for this agent
ROOM_NAME_PREFIX = "voice_workflow_"

# Agent configuration
LLM_MODEL = os.getenv("AGENT_LLM_MODEL", "gpt-4o-mini")
TTS_VOICE = os.getenv("AGENT_TTS_VOICE", "alloy")
STT_LANGUAGE = os.getenv("AGENT_STT_LANGUAGE", "en")
VAD_ENABLED = os.getenv("AGENT_VAD_ENABLED", "true").lower() == "true"

# Automax3 API configuration
AUTOMAX3_BASEURL = os.getenv("AUTOMAX3_BASEURL", "https://epmstg.automaxsw.com")
AUTOMAX3_USERNAME = os.getenv("AUTOMAX3_USERNAME", "474062237214310401")
AUTOMAX3_PASSWORD = os.getenv("AUTOMAX3_PASSWORD", "FFtq2lE3V3HrLO9DrjQAd6yxb2i4Xehq1zWRK3zELuewQOgOw0p59bx4bkun0jdI")


class Automax3Client:
    """Client for Automax3 API - handles authentication and incident creation."""

    def __init__(self):
        self.base_url = AUTOMAX3_BASEURL
        self.userid = AUTOMAX3_USERNAME
        self.password = AUTOMAX3_PASSWORD

        self.token: Optional[str] = None
        self.id_token: Optional[str] = None

        self.common_headers = {
            'Content-Type': "application/json",
        }

        # Cache for classifications and locations
        self._classifications_cache: Optional[list] = None
        self._locations_cache: Optional[list] = None

    def client_login(self) -> Optional[dict]:
        """Authenticate with Automax3 API using client credentials."""
        url = f'{self.base_url}/auth/oauth2/token'

        data = {
            'grant_type': 'client_credentials',
            'scope': 'profile api'
        }

        auth = HTTPBasicAuth(self.userid, self.password)

        headers = {
            'Content-Type': 'application/x-www-form-urlencoded'
        }

        try:
            response = requests.post(url, data=data, headers=headers, auth=auth)

            if response.status_code == 200:
                response_data = response.json()

                self.token = response_data['access_token']
                self.id_token = response_data['id_token']

                logger.info("Automax3: Access token obtained successfully!")
                return response_data
            else:
                logger.error(f'Automax3 login error: {response.status_code} {response.text}')
                return None
        except requests.exceptions.RequestException as e:
            logger.error(f'Automax3 login request failed: {e}')
            return None

    def ensure_authenticated(self) -> bool:
        """Ensure we have a valid token, login if necessary."""
        if not self.token:
            result = self.client_login()
            return result is not None
        return True

    def get_classifications(self) -> Optional[list]:
        """Get classification hierarchy from Automax3."""
        if self._classifications_cache:
            return self._classifications_cache

        url = f'{self.base_url}/api/classifications/hierarchy'

        if not self.ensure_authenticated():
            return None

        self.common_headers['Authorization'] = f"Bearer {self.token}"

        try:
            response = requests.get(url, headers=self.common_headers)
            if response.status_code == 200:
                data = response.json()
                self._classifications_cache = data.get('data', {}).get('hierarchy', [])
                return self._classifications_cache
            else:
                logger.error(f'Automax3 get_classifications error: {response.status_code}')
                return None
        except requests.exceptions.RequestException as e:
            logger.error(f'Automax3 get_classifications failed: {e}')
            return None

    def get_locations(self) -> Optional[list]:
        """Get location hierarchy from Automax3."""
        if self._locations_cache:
            return self._locations_cache

        url = f'{self.base_url}/api/locations/hierarchy'

        if not self.ensure_authenticated():
            return None

        self.common_headers['Authorization'] = f"Bearer {self.token}"

        try:
            response = requests.get(url, headers=self.common_headers)
            if response.status_code == 200:
                data = response.json()
                self._locations_cache = data.get('data', {}).get('hierarchy', [])
                return self._locations_cache
            else:
                logger.error(f'Automax3 get_locations error: {response.status_code}')
                return None
        except requests.exceptions.RequestException as e:
            logger.error(f'Automax3 get_locations failed: {e}')
            return None

    def get_classification_id_by_name(self, name: str) -> Optional[str]:
        """Get classification ID by name."""
        classifications = self.get_classifications()
        if not classifications:
            return None

        for classification in classifications:
            if classification.get('en_name', '').lower() == name.lower():
                return classification.get('id')
        return None

    def get_location_id_by_name(self, name: str) -> Optional[str]:
        """Get location ID by name."""
        locations = self.get_locations()
        if not locations:
            return None

        for location in locations:
            if location.get('en_name', '').lower() == name.lower():
                return location.get('id')
        return None

    def get_classification_names(self) -> list:
        """Get list of available classification names."""
        classifications = self.get_classifications()
        if not classifications:
            return []
        return [c.get('en_name', '') for c in classifications if c.get('en_name')]

    def get_location_names(self) -> list:
        """Get list of available location names."""
        locations = self.get_locations()
        if not locations:
            return []
        return [loc.get('en_name', '') for loc in locations if loc.get('en_name')]

    def create_incident(
        self,
        caller_name: str,
        classification_id: str,
        location_id: str,
        attachment_id: str = "",
        coordinates: dict = None,
        description: str = "",
        criticality: str = "LOW"
    ) -> Optional[dict]:
        """Create an incident in Automax3."""
        url = f'{self.base_url}/api/compose/namespace/431842611944685569/module/431842611943440385/record/'

        if not self.ensure_authenticated():
            return None

        self.common_headers['Authorization'] = f"Bearer {self.token}"

        coordinates_str = json.dumps({"coordinates": coordinates}) if coordinates else ""

        body = {
            "meta": {},
            "records": [],
            "values": [
                {"name": "Channel", "value": "Chatbot"},
                {"name": "Criticality", "value": criticality},
                {"name": "Caller_name", "value": caller_name},
                {"name": "Last_call_date", "value": datetime.now().isoformat() + "Z"},
                {"name": "National_ID", "value": ""},
                {"name": "Mobile_number", "value": ""},
                {"name": "Classification", "value": classification_id},
                {"name": "Incident_reason", "value": ""},
                {"name": "Incident_Description", "value": description or "Incident created via voice agent"},
                {"name": "Map", "value": coordinates_str},
                {"name": "Primary_Location", "value": location_id},
                {"name": "District", "value": ""},
                {"name": "Street", "value": ""},
                {"name": "Status", "value": ""},
                {"name": "Assigned_To", "value": "425635139776282625"},
                {"name": "Comments", "value": json.dumps({
                    'created': datetime.now().isoformat() + "Z",
                    'comment': 'Created via voice workflow agent',
                    'author': 'voice-agent',
                    'name': 'Voice Agent'
                })},
                {"name": "Attachments", "value": attachment_id}
            ]
        }

        try:
            response = requests.post(url, json=body, headers=self.common_headers)
            logger.info(f"Automax3 create_incident response: {response.status_code}")

            if response.status_code == 200:
                response_data = response.json()
                logger.info("Automax3: Incident created successfully!")
                return response_data
            else:
                logger.error(f'Automax3 create_incident error: {response.status_code} {response.text}')
                return None
        except requests.exceptions.RequestException as e:
            logger.error(f'Automax3 create_incident failed: {e}')
            return None


# Global Automax3 client instance
_automax_client: Optional[Automax3Client] = None


def get_automax_client() -> Automax3Client:
    """Get or create Automax3 client instance."""
    global _automax_client
    if _automax_client is None:
        _automax_client = Automax3Client()
        _automax_client.client_login()
    return _automax_client


class WorkflowContext:
    """Holds workflow context parsed from room metadata."""
    
    def __init__(self, metadata: dict):
        self.session_id = metadata.get("session_id", "unknown")
        self.workflow = metadata.get("workflow", {})
        self.backend_url = metadata.get("backend_url", "http://localhost:8000")
        self.greeting_message = metadata.get("greeting_message")
        self.agent_id = metadata.get("agent_id")
        self.company_id = metadata.get("company_id")
        
        # DEBUG: Log what workflow data we received
        logger.info(f"[WORKFLOW CONTEXT] workflow type: {type(self.workflow)}")
        logger.info(f"[WORKFLOW CONTEXT] workflow keys: {list(self.workflow.keys()) if isinstance(self.workflow, dict) else 'NOT A DICT'}")
        
        # If workflow is a string, try to parse it
        if isinstance(self.workflow, str):
            try:
                import json
                self.workflow = json.loads(self.workflow)
                logger.info(f"[WORKFLOW CONTEXT] Parsed workflow from string, keys: {list(self.workflow.keys())}")
            except:
                logger.error(f"[WORKFLOW CONTEXT] Failed to parse workflow string")
                self.workflow = {}
        
        # Extract workflow details
        self.workflow_name = self.workflow.get("name", "Workflow")
        self.workflow_description = self.workflow.get("description", "")
        
        logger.info(f"[WORKFLOW CONTEXT] Workflow name: {self.workflow_name}")
        
        # Extract services from first prompt node's options
        self.services = self._extract_services()
        
        # Extract all workflow steps that collect user input
        self.input_steps = self._extract_input_steps()
        
    def _get_nodes(self) -> list:
        """Get nodes from workflow, handling different JSON structures."""
        # Try visual_steps.nodes first (940 workflow format)
        visual_steps = self.workflow.get("visual_steps", {})
        
        # visual_steps might be a string
        if isinstance(visual_steps, str):
            try:
                import json
                visual_steps = json.loads(visual_steps)
                logger.info(f"[WORKFLOW CONTEXT] Parsed visual_steps from string")
            except:
                visual_steps = {}
        
        nodes = visual_steps.get("nodes", []) if isinstance(visual_steps, dict) else []
        logger.info(f"[WORKFLOW CONTEXT] visual_steps.nodes count: {len(nodes)}")
        
        if nodes:
            return nodes
        
        # Fall back to root-level nodes
        root_nodes = self.workflow.get("nodes", [])
        logger.info(f"[WORKFLOW CONTEXT] root nodes count: {len(root_nodes)}")
        return root_nodes
    
    def _extract_services(self) -> list:
        """Extract service options from the first prompt node (main menu)."""
        nodes = self._get_nodes()
        
        for node in nodes:
            if node.get("type") == "prompt":
                params = node.get("data", {}).get("params", {})
                options = params.get("options", "")
                if options:
                    return [opt.strip() for opt in options.split(",")]
        return []
    
    def _extract_input_steps(self) -> list:
        """Extract all steps that require user input with their variable names."""
        nodes = self._get_nodes()
        steps = []
        
        for node in nodes:
            node_type = node.get("type", "")
            node_id = node.get("id", "")
            data = node.get("data", {})
            params = data.get("params", {})
            
            if node_type == "prompt":
                steps.append({
                    "node_id": node_id,
                    "type": "prompt",
                    "variable_name": params.get("save_to_variable", ""),
                    "prompt_text": params.get("prompt_text", ""),
                    "options": params.get("options", ""),
                    "label": data.get("label", "")
                })
            elif node_type == "listen":
                expected_type = params.get("expected_input_type", "text")
                steps.append({
                    "node_id": node_id,
                    "type": "listen",
                    "variable_name": params.get("save_to_variable", ""),
                    "input_type": expected_type,  # text, attachment, location
                    "requires_form": expected_type in ["attachment", "location"],
                    "label": data.get("label", "")
                })
            elif node_type == "code":
                steps.append({
                    "node_id": node_id,
                    "type": "code",
                    "label": data.get("label", ""),
                    "return_variables": data.get("return_variables", []),
                    "arguments": data.get("arguments", [])
                })
        
        return steps
    
    def get_variables_to_collect(self) -> list:
        """Get list of variable names that need to be collected from user."""
        variables = []
        for step in self.input_steps:
            if step.get("type") in ["prompt", "listen"] and step.get("variable_name"):
                variables.append({
                    "name": step["variable_name"],
                    "type": step.get("input_type", "text"),
                    "requires_form": step.get("requires_form", False)
                })
        return variables
    
    def _build_workflow_graph(self) -> str:
        """Build a readable workflow graph using BFS traversal."""
        from collections import defaultdict, deque
        
        nodes = self._get_nodes()
        edges = self._get_edges()
        
        if not nodes:
            return "No workflow steps defined."
        
        # Build node registry
        node_type_map = {}
        node_label_map = {}
        
        for n in nodes:
            nid = n.get("id")
            node_type_map[nid] = n.get("type", "unknown")
            data = n.get("data", {})
            params = data.get("params", {})
            
            # Get best label for node
            label = (
                data.get("label") or 
                data.get("output_value", "")[:50] or
                params.get("prompt_text", "")[:50] or
                n.get("type", "unknown")
            )
            node_label_map[nid] = label
        
        # Build adjacency list
        adjacency = defaultdict(list)
        for e in edges:
            src = e.get("source")
            tgt = e.get("target")
            handle = e.get("sourceHandle", "output")
            adjacency[src].append((handle, tgt))
        
        # Find start node
        start_node = None
        for nid, ntype in node_type_map.items():
            if ntype == "start":
                start_node = nid
                break
        
        if not start_node:
            # Use first node if no start
            start_node = nodes[0].get("id") if nodes else None
        
        # BFS traversal order
        visited = set()
        queue = deque([start_node]) if start_node else deque()
        bfs_order = []
        
        while queue:
            n = queue.popleft()
            if n in visited or n is None:
                continue
            visited.add(n)
            bfs_order.append(n)
            for _, child in adjacency.get(n, []):
                if child not in visited:
                    queue.append(child)
        
        # Build output text
        lines = []
        lines.append("=== EXECUTION ORDER ===")
        for i, nid in enumerate(bfs_order[:30]):  # Limit to 30 nodes
            ntype = node_type_map.get(nid, "unknown")
            label = node_label_map.get(nid, "")[:40]
            
            # Format by type with emoji
            if ntype == "start":
                lines.append(f"{i+1}. � START")
            elif ntype == "response":
                lines.append(f"{i+1}. � SAY: \"{label}\"")
            elif ntype == "prompt":
                lines.append(f"{i+1}. ❓ ASK: \"{label}\"")
            elif ntype == "listen":
                lines.append(f"{i+1}. � LISTEN for input")
            elif ntype == "condition":
                lines.append(f"{i+1}. 🔀 CHECK: {label}")
            elif ntype == "code":
                lines.append(f"{i+1}. ⚙️ EXECUTE: {label}")
            elif ntype == "tool":
                lines.append(f"{i+1}. 🔧 TOOL: {label}")
            else:
                lines.append(f"{i+1}. 📌 {ntype.upper()}: {label}")
        
        # Add edge descriptions (transitions)
        lines.append("\n=== TRANSITIONS ===")
        edge_count = 0
        for src, neighbors in adjacency.items():
            if edge_count >= 20:  # Limit edges
                lines.append("... (more transitions)")
                break
            for handle, tgt in neighbors:
                src_label = node_label_map.get(src, src)[:25]
                tgt_label = node_label_map.get(tgt, tgt)[:25]
                lines.append(f"- {src_label} --[{handle}]--> {tgt_label}")
                edge_count += 1
        
        return "\n".join(lines)
    
    def _get_edges(self) -> list:
        """Get edges from workflow, handling different JSON structures."""
        edges = self.workflow.get("visual_steps", {}).get("edges", [])
        if edges:
            return edges
        return self.workflow.get("edges", [])
        
    def generate_system_prompt(self) -> str:
        """Generate a system prompt for EPM 940 assistant."""
        base_prompt = """You are a helpful voice assistant for EPM 940.

YOUR JOB:
- Answer user questions and assist with their requests
- Keep responses SHORT and conversational (this is voice)
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

When the user wants to report an incident, follow this EXACT flow:
1. Ask for caller's name (who is reporting)
2. Present the INCIDENT CLASSIFICATION OPTIONS above and ask which one describes their issue
3. Present the LOCATION OPTIONS above and ask where the incident is located
4. Ask for a brief description of the incident (optional but recommended)
5. Ask about criticality: LOW, MEDIUM, HIGH, or CRITICAL (default is LOW)
6. IMPORTANT: After collecting all information, follow these steps in order:
   a. Immediately say your final summary and gooodbye:must say these exact words that is written after this inside inveerted comma and do not change a single letter "I've opened the report form for you. [Brief Summary]... Goodbye!"
   b. Call terminate_call() to end the session.

TOOL USAGE:
- store_collected_data({"variable_name": "value"}) - Store user-provided data
- get_available_classifications() - Returns the hardcoded classification list
- get_available_locations() - Returns the hardcoded location list
- terminate_call() - MUST call this at the very end to end the conversation
- create_automax_incident(...) - Only call this if you need to create the incident WITHOUT user form confirmation

RULES:
- Keep responses SHORT (this is voice)
- Be helpful and conversational
- Wait for FORM_DONE signal before finalizing
"""

        return base_prompt


# Global workflow context holder for function tools
_workflow_ctx: WorkflowContext = None
_http_client: httpx.AsyncClient = None


async def get_http_client() -> httpx.AsyncClient:
    """Get or create HTTP client."""
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(timeout=30.0)
    return _http_client


@llm.function_tool()
async def request_server_logic(
    action: str,
    data: str = "{}"
) -> str:
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
    
    logger.info(f"Calling backend: action={action}, data={data}")
    
    try:
        client = await get_http_client()
        response = await client.post(
            f"{_workflow_ctx.backend_url}/api/v1/voice-workflow/step",
            json={
                "session_id": _workflow_ctx.session_id,
                "action": action,
                "data": data
            }
        )
        result = response.json()
        
        logger.info(f"Backend response: {result}")
        
        # Handle different response statuses
        if result.get("status") == "waiting_for_input":
            url = result.get("url", "")
            return f"SENT_FORM: {url}. Tell the user: I've sent you a link to upload the required files. Please check it and let me know when you're done."
        
        elif result.get("status") == "success":
            return f"SERVER_SUCCESS: {result.get('message', 'Action completed successfully')}"
        
        elif result.get("status") == "error":
            return f"SERVER_ERROR: {result.get('message', 'An error occurred')}"
        
        else:
            return f"SERVER_RESPONSE: {result.get('message', 'Action processed')}"
            
    except Exception as e:
        logger.error(f"Backend call failed: {e}")
        return f"SERVER_ERROR: Failed to communicate with server - {str(e)}"


@llm.function_tool()
async def store_collected_data(data: str) -> str:
    """
    Store data collected from the user during conversation.
    Call this after the user answers each question with the correct variable name.
    
    Args:
        data: JSON string mapping variable name to value, e.g. '{"user_name": "John"}'
              The variable name should match the workflow's save_to_variable field.
    
    Returns:
        Confirmation that data was stored
    """
    global _workflow_ctx
    
    if _workflow_ctx is None:
        return "ERROR: Session not initialized"
    
    logger.info(f"Storing collected data: {data}")
    
    try:
        client = await get_http_client()
        response = await client.post(
            f"{_workflow_ctx.backend_url}/api/v1/voice-workflow/step",
            json={
                "session_id": _workflow_ctx.session_id,
                "action": "store_data",
                "data": data
            }
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


@llm.function_tool()
async def request_form_input(field_type: str, variable_name: str) -> str:
    """
    Request form input for data that cannot be collected via voice (files, images, location).
    After calling this, tell the user to check the chat for a form link.
    
    Args:
        field_type: Type of input needed - 'attachment' for files/images, 'location' for GPS
        variable_name: The variable name to store the result (matches workflow save_to_variable)
    
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
                "data": json.dumps({
                    "field_type": field_type,
                    "variable_name": variable_name
                })
            }
        )
        result = response.json()
        logger.info(f"Request form response: {result}")
        
        if result.get("status") == "waiting_for_input":
            url = result.get("url", "")
            return f"FORM_SENT: A form has been sent to the chat. Tell the user: 'I've sent a form to your chat for uploading {field_type}. Please fill it out and let me know when done.'"
        else:
            return f"FORM_ERROR: {result.get('message', 'Failed to send form')}"
            
    except Exception as e:
        logger.error(f"Request form failed: {e}")
        return f"ERROR: {str(e)}"


@llm.function_tool()
async def execute_workflow_step(step_id: str, step_type: str = "code") -> str:
    """
    Execute a workflow processing step on the server (code execution, HTTP requests, etc).
    Say "Let me process that for you" before calling this.
    
    Args:
        step_id: The node ID of the step to execute (e.g. "code-1766053887866")
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
                "data": json.dumps({
                    "step_id": step_id,
                    "step_type": step_type
                })
            }
        )
        result = response.json()
        logger.info(f"Execute step response: {result}")
        
        if result.get("status") == "success":
            output = result.get("output", {})
            return f"STEP_COMPLETED: Processing done. Result: {json.dumps(output) if output else 'Success'}. Continue to the next step."
        elif result.get("status") == "processing":
            return "STEP_PROCESSING: Still processing. Wait a moment and check again."
        else:
            return f"STEP_ERROR: {result.get('message', 'Step execution failed')}"
            
    except Exception as e:
        logger.error(f"Execute step failed: {e}")
        return f"ERROR: {str(e)}"


# ============================================================================
# Automax3 API Function Tools
# ============================================================================

@llm.function_tool()
async def get_available_classifications() -> str:
    """
    Get list of available incident classifications.
    These are the hardcoded classification options for incident reporting.

    Returns:
        List of available classification names
    """
    # Hardcoded classification options
    classifications = ["Manholes", "Street Lights", "Street Furniture", "Potholes", "Barriers"]
    logger.info(f"Returning hardcoded classifications: {classifications}")
    return f"CLASSIFICATIONS_AVAILABLE: {', '.join(classifications)}"


@llm.function_tool()
async def get_available_locations() -> str:
    """
    Get list of available locations.
    These are the hardcoded location options for incident reporting.

    Returns:
        List of available location names
    """
    # Hardcoded location options
    locations = ["Dammam", "Dammam East", "Dammam West"]
    logger.info(f"Returning hardcoded locations: {locations}")
    return f"LOCATIONS_AVAILABLE: {', '.join(locations)}"


# Global reference to room and event for sending data messages and waiting
_current_room = None
_form_event = asyncio.Event()


@llm.function_tool()
async def trigger_form_popup(
    caller_name: str,
    classification: str,
    location: str,
    description: str = "",
    criticality: str = "LOW"
) -> str:
    """
    Trigger the popup form on the frontend to confirm incident details.
    MUST call this after collecting all incident information from the user.
    
    This sends an OPEN_FORM signal to the frontend with all collected data.
    The user will see a popup to review and confirm the incident details.
    Wait for the form submission before creating the incident.
    
    Args:
        caller_name: Name of the person reporting the incident
        classification: Type of incident (Manholes, Street Lights, Street Furniture, Potholes, or Barriers)
        location: Where the incident is (Dammam, Dammam East, or Dammam West)
        description: Description of the incident
        criticality: Severity level (LOW, MEDIUM, HIGH, or CRITICAL)
    
    Returns:
        Status message indicating form was triggered
    """
    global _current_room, _form_event
    
    logger.info(f"Triggering form popup with data: caller={caller_name}, classification={classification}, location={location}")
    
    try:
        # Clear the event before triggering
        _form_event.clear()
        
        # SIMPLE TRIGGER: Send plain string as requested to ensure it opens
        signal = "OPEN_FORM"
        
        if _current_room:
            logger.info(f"Room state: {_current_room.name}, connected={_current_room.isconnected}")
            if hasattr(_current_room, 'local_participant'):
                # Prepare data payload
                data_payload = {
                    "type": "OPEN_FORM",
                    "caller_name": caller_name or "N/A",
                    "classification": classification or "N/A",
                    "location": location or "N/A",
                    "description": description or "N/A",
                    "criticality": criticality or "LOW"
                }
                json_msg = json.dumps(data_payload).encode('utf-8')

                # Send signal multiple times with slight delay to ensure delivery
                remote_count = len(_current_room.remote_participants)
                logger.info(f"Broadcasting OPEN_FORM (Topic: form_trigger) with payload: {data_payload} to {remote_count} participants")
                
                for i in range(3):
                    # 1. Plain string (legacy/simple)
                    await _current_room.local_participant.publish_data(
                        signal.encode('utf-8'),
                        kind=rtc.DataPacketKind.RELIABLE
                    )
                    # 2. Topic based (String)
                    await _current_room.local_participant.publish_data(
                        signal.encode('utf-8'),
                        kind=rtc.DataPacketKind.RELIABLE,
                        topic="form_trigger"
                    )
                    # 3. JSON payload (Topic-based now for better routing)
                    await _current_room.local_participant.publish_data(
                        json_msg,
                        kind=rtc.DataPacketKind.RELIABLE,
                        topic="form_trigger"
                    )
                    
                    logger.info(f"Burst {i+1} sent (Plain, Topic, Full JSON with Topic)")
                    await asyncio.sleep(0.4)
                
                return "FORM_SIGNAL_SENT: Signal sent with AI data payload. Proceeding to goodbye."
            else:
                logger.warning("Room object has no local_participant attribute")
        else:
            logger.warning("Global _current_room is None - cannot send data message")
            return "FORM_TRIGGERED: The incident details have been saved, but I couldn't open the popup."
            
    except Exception as e:
        logger.error(f"Failed to trigger form popup: {e}")
        return f"FORM_ERROR: Could not trigger the form. Error: {str(e)}."


@llm.function_tool()
async def terminate_call() -> str:
    """
    Disconnect the agent and terminate the voice call session.
    Call this ONLY after saying goodbye to the user.
    """
    logger.info("Agent requested call termination. Waiting 3s for audio to clear...")
    
    async def delayed_disconnect():
        # Final "Force" trigger to the frontend as a safeguard
        if _current_room and hasattr(_current_room, 'local_participant'):
            try:
                msg = "OPEN_FORM".encode('utf-8')
                await _current_room.local_participant.publish_data(msg, kind=rtc.DataPacketKind.RELIABLE)
                await _current_room.local_participant.publish_data(msg, kind=rtc.DataPacketKind.RELIABLE, topic="form_trigger")
                # Also JSON for good measure
                json_msg = json.dumps({"type": "OPEN_FORM"}).encode('utf-8')
                await _current_room.local_participant.publish_data(json_msg, kind=rtc.DataPacketKind.RELIABLE)
                logger.info("FINAL signal burst sent before disconnect")
            except Exception as e:
                logger.warning(f"Final signal fail: {e}")
                
        await asyncio.sleep(3.0)
        if _current_room:
            logger.info("Closing room connection...")
            await _current_room.disconnect()
            
    asyncio.create_task(delayed_disconnect())
    return "CALL_TERMINATED: I'm ending the call now. You can finish the report on your screen. Goodbye!"


@llm.function_tool()
async def create_automax_incident(
    caller_name: str,
    classification_name: str,
    location_name: str,
    description: str = "",
    attachment_id: str = "",
    coordinates: str = "",
    criticality: str = "LOW"
) -> str:
    """
    Create an incident in Automax3 system. You MUST collect all required information
    from the user BEFORE calling this function:

    Required information to collect from user:
    1. caller_name - The name of the person reporting the incident
    2. classification_name - Type of incident (call get_available_classifications first)
    3. location_name - Where the incident occurred (call get_available_locations first)

    Optional information:
    4. description - Details about the incident
    5. attachment_id - ID of any attached files (from form submission)
    6. coordinates - GPS coordinates as JSON string, e.g. '{"lat": 12.34, "lng": 56.78}'
    7. criticality - Severity level: "LOW", "MEDIUM", "HIGH", or "CRITICAL" (default: LOW)

    Args:
        caller_name: Name of the person reporting
        classification_name: Classification type name (must match available classifications)
        location_name: Location name (must match available locations)
        description: Incident description
        attachment_id: ID of attached file if any
        coordinates: GPS coordinates as JSON string
        criticality: Severity level

    Returns:
        Success message with incident ID or error message
    """
    logger.info(f"Creating Automax incident: caller={caller_name}, classification={classification_name}, location={location_name}")

    try:
        client = get_automax_client()

        # Get classification ID by name
        classification_id = client.get_classification_id_by_name(classification_name)
        if not classification_id:
            available = client.get_classification_names()
            return f"INCIDENT_ERROR: Classification '{classification_name}' not found. Available: {', '.join(available[:10])}"

        # Get location ID by name
        location_id = client.get_location_id_by_name(location_name)
        if not location_id:
            available = client.get_location_names()
            return f"INCIDENT_ERROR: Location '{location_name}' not found. Available: {', '.join(available[:10])}"

        # Parse coordinates if provided
        coords_dict = None
        if coordinates:
            try:
                coords_dict = json.loads(coordinates)
            except json.JSONDecodeError:
                logger.warning(f"Could not parse coordinates: {coordinates}")

        # Create the incident
        result = client.create_incident(
            caller_name=caller_name,
            classification_id=classification_id,
            location_id=location_id,
            attachment_id=attachment_id,
            coordinates=coords_dict,
            description=description,
            criticality=criticality
        )

        if result:
            record_id = result.get('response', {}).get('recordID', 'unknown')
            return f"INCIDENT_CREATED: Successfully created incident #{record_id}. Tell the user their incident has been registered and they will be contacted soon."
        else:
            return "INCIDENT_ERROR: Failed to create incident in Automax3. Please try again."

    except Exception as e:
        logger.error(f"Create incident failed: {e}")
        return f"INCIDENT_ERROR: {str(e)}"


class WorkflowVoiceAssistant(Agent):
    """Voice assistant that executes workflows."""
    
    def __init__(self, ctx: WorkflowContext):
        self.workflow_ctx = ctx
        
        super().__init__(
            instructions=ctx.generate_system_prompt(),
            stt=openai.STT(),  # Uses OpenAI Whisper
            llm=openai.LLM(model=LLM_MODEL),
            tts=openai.TTS(voice=TTS_VOICE),
        )
        logger.info(f"Workflow voice assistant initialized for: {ctx.workflow_name}")
    
    # Note: on_enter removed - greeting is handled in entrypoint after session.start()


# Global variable to hold the agent session for data message handling
_current_session: AgentSession = None


async def entrypoint(ctx: JobContext):
    """Main entrypoint for the workflow voice agent worker."""
    global _current_session, _workflow_ctx, _current_room
    
    try:
        logger.info(f"Workflow agent connecting to room: {ctx.room.name}")
        
        # Connect to the room (audio only)
        await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
        logger.info("Connected to room successfully")
        
        # Set current room for tool reference
        _current_room = ctx.room
        
        # Extract session_id from room name (format: voice_workflow_{session_id})
        room_name = ctx.room.name
        session_id = room_name.replace("voice_workflow_", "") if room_name.startswith("voice_workflow_") else None
        logger.info(f"[WORKFLOW FETCH] Room: {room_name}, Session ID: {session_id}")
        
        # ALWAYS fetch workflow from backend - don't rely on room metadata
        metadata = {"session_id": session_id}
        workflow_fetched = False
        
        if session_id:
            backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
            logger.info(f"[WORKFLOW FETCH] Fetching from: {backend_url}/api/v1/voice-workflow/session/{session_id}")
            
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.get(f"{backend_url}/api/v1/voice-workflow/session/{session_id}")
                    logger.info(f"[WORKFLOW FETCH] Response status: {response.status_code}")
                    
                    if response.status_code == 200:
                        session_data = response.json()
                        # logger.info(f"[WORKFLOW FETCH] Session data keys: {list(session_data.keys())}")
                        
                        if session_data.get("workflow_json"):
                            metadata["workflow"] = session_data["workflow_json"]
                            metadata["backend_url"] = backend_url
                            workflow_fetched = True
                            wf_name = session_data["workflow_json"].get("name", "Unknown")
                            wf_nodes = len(session_data["workflow_json"].get("nodes", []))
                            vs_nodes = len(session_data["workflow_json"].get("visual_steps", {}).get("nodes", []))
                            # logger.info(f"[WORKFLOW FETCH] SUCCESS! Workflow: {wf_name}, root nodes: {wf_nodes}, visual_steps nodes: {vs_nodes}")
                    else:
                        logger.error(f"[WORKFLOW FETCH] Failed: {response.status_code} - {response.text[:200]}")
            except Exception as e:
                logger.error(f"[WORKFLOW FETCH] Exception: {e}")
        
        if not workflow_fetched:
            logger.error("[WORKFLOW FETCH] FAILED - No workflow data available!")
        
        workflow_ctx = WorkflowContext(metadata)
        _workflow_ctx = workflow_ctx  # Set global context for function tools
        
        # Detailed logging for debugging
        # logger.info(f"[AGENT DEBUG] Session ID: {workflow_ctx.session_id}")
        # logger.info(f"[AGENT DEBUG] Workflow Name: {workflow_ctx.workflow_name}")
        # logger.info(f"[AGENT DEBUG] Services extracted: {workflow_ctx.services}")
        # logger.info(f"[AGENT DEBUG] Input steps count: {len(workflow_ctx.input_steps)}")
        # logger.info(f"[AGENT DEBUG] Variables to collect: {workflow_ctx.get_variables_to_collect()}")
        
        # Log first part of system prompt
        system_prompt = workflow_ctx.generate_system_prompt()
        # logger.info(f"[AGENT DEBUG] Session prompt (first 500 chars):\n{system_prompt[:500]}...")
        
        # Check for already-present participants first, then wait if none
        participants = list(ctx.room.remote_participants.values())
        if participants:
            participant = participants[0]
            logger.info(f"Found existing participant: {participant.identity}")
        else:
            logger.info("Waiting for participant...")
            participant = await ctx.wait_for_participant()
        logger.info(f"Starting workflow voice assistant for participant: {participant.identity}")
        
        # Create agent session with VAD
        session = AgentSession(
            vad=ctx.proc.userdata.get("vad") if VAD_ENABLED else None,
        )
        _current_session = session
        
        # Set up connection state handlers for network resilience
        @ctx.room.on("connection_state_changed")
        def on_connection_state_changed(state):
            """Monitor connection state for reconnection handling."""
            logger.info(f"[CONNECTION] State changed to: {state}")
            if str(state) == "ConnectionState.CONNECTED":
                logger.info("[CONNECTION] Successfully connected/reconnected!")
            elif str(state) == "ConnectionState.RECONNECTING":
                logger.warning("[CONNECTION] Reconnecting due to network issues...")
        
        @ctx.room.on("reconnected")
        def on_reconnected():
            """Handle successful reconnection."""
            logger.info("[CONNECTION] Successfully reconnected to room!")
        
        @ctx.room.on("disconnected")
        def on_disconnected():
            """Log disconnection event."""
            logger.warning("[CONNECTION] Disconnected from room.")
        
        # Set up data message listener for form completion
        @ctx.room.on("data_received")
        def on_data_received(data: bytes, participant, kind):
            """Handle data messages from the frontend (e.g., form submissions)."""
            try:
                decoded = data.decode().strip()
                
                # Try JSON first
                try:
                    payload = json.loads(decoded)
                    message_type = payload.get("type", "")
                    if message_type in ["FORM_SUBMITTED", "FORM_DONE"]:
                        logger.info(f"Form submission signal received (JSON): {message_type}")
                        _form_event.set()
                        return
                except:
                    pass
                
                # Fallback to plain string
                if decoded in ["FORM_DONE", "OPEN_FORM_DONE", "FORM_SUBMITTED"]:
                    logger.info(f"Form submission signal received (String): {decoded}")
                    _form_event.set()
                    
            except Exception as e:
                logger.error(f"Error handling data message: {e}")
        
        # Create the agent
        agent = WorkflowVoiceAssistant(workflow_ctx)
        logger.info("Created WorkflowVoiceAssistant")
        
        # Start the session with RoomIO options
        logger.info("Starting agent session with RoomIO...")
        await session.start(
            room=ctx.room,
            agent=agent,
            room_input_options=RoomInputOptions(
                audio_enabled=True,
                close_on_disconnect=False  # Keep agent alive on temporary network drops
            ),
            room_output_options=RoomOutputOptions(
                audio_enabled=True,
                transcription_enabled=True
            )
        )
        logger.info("Agent session started successfully")

        # Log AGENT speech (what the AI says)
        @session.on("agent_speech_committed")
        def on_agent_speech_committed(msg):
            """Log agent speech to terminal for transcript visibility."""
            content = getattr(msg, 'content', getattr(msg, 'text', str(msg)))
            logger.info(f"[TRANSCRIPT] 🤖 AGENT: {content}")
        
        # Log USER speech (what the user says)
        @session.on("user_speech_committed")
        def on_user_speech_committed(msg):
            """Log user speech to terminal for transcript visibility."""
            content = getattr(msg, 'content', getattr(msg, 'text', str(msg)))
            logger.info(f"[TRANSCRIPT] 👤 USER: {content}")
        
        # Fixed greeting for EPM 940
        greeting = "Hello, welcome to EPM 940. How can I assist you today?"

        logger.info("Using fixed EPM 940 greeting")
        
        await session.generate_reply(
            instructions=greeting,
            allow_interruptions=True
        )
        
        logger.info("Workflow voice agent greeting sent")
        
    except Exception as e:
        logger.error(f"Error in entrypoint: {e}", exc_info=True)


def prewarm(proc: JobProcess):
    """Prewarm function to load models before the agent starts."""
    logger.info("Prewarming workflow agent models...")

    if VAD_ENABLED:
        proc.userdata["vad"] = silero.VAD.load()

    # Initialize Automax3 client and authenticate
    # logger.info("Initializing Automax3 client...")
    # try:
    #     automax_client = get_automax_client()
    #     # Pre-cache classifications and locations
    #     classifications = automax_client.get_classifications()
    #     locations = automax_client.get_locations()
    #     logger.info(f"Automax3: Loaded {len(classifications) if classifications else 0} classifications, "
    #                f"{len(locations) if locations else 0} locations")
    # except Exception as e:
    #     logger.warning(f"Automax3 prewarm failed (will retry on first use): {e}")

    logger.info("Prewarm complete")


async def request_fnc(request):
    """
    Filter which rooms this agent should join.
    Only accept rooms with the voice_workflow_ prefix.
    """
    room_name = request.room.name
    should_accept = room_name.startswith(ROOM_NAME_PREFIX)
    
    if should_accept:
        logger.info(f"Accepting job for room: {room_name}")
        await request.accept()  # MUST call accept() to join the room
    else:
        logger.debug(f"Rejecting job for room: {room_name} (not a voice workflow room)")
        await request.reject()


if __name__ == "__main__":
    """Run the workflow voice agent worker."""
    
    # Validate environment
    if not LIVEKIT_API_KEY or not LIVEKIT_API_SECRET:
        raise ValueError("LIVEKIT_API_KEY2 or LIVEKIT_API_KEY must be set in .env")
    
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY must be set in .env for LLM, STT, and TTS")
    
    logger.info("=" * 60)
    logger.info("Starting Workflow Voice Agent...")
    logger.info("=" * 60)
    logger.info(f"LiveKit URL: {LIVEKIT_URL}")
    logger.info(f"API Key: {LIVEKIT_API_KEY[:8]}..." if LIVEKIT_API_KEY else "API Key: NOT SET")
    logger.info(f"Room prefix filter: {ROOM_NAME_PREFIX}*")
    logger.info(f"LLM Model: {LLM_MODEL}")
    logger.info(f"TTS Voice: {TTS_VOICE}")
    logger.info(f"STT Language: {STT_LANGUAGE}")
    logger.info(f"VAD Enabled: {VAD_ENABLED}")
    logger.info("=" * 60)
    
    # Run the agent with explicit LiveKit credentials
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            request_fnc=request_fnc,  # Only accept voice_workflow_ rooms
            ws_url=LIVEKIT_URL,  # Explicitly set the LiveKit server URL
            api_key=LIVEKIT_API_KEY,
            api_secret=LIVEKIT_API_SECRET,
        ),
    )

