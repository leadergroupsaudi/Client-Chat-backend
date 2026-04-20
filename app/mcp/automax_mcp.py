from fastmcp import FastMCP
from pydantic import BaseModel
from typing import Literal, Optional
from starlette.requests import Request
from starlette.responses import JSONResponse
import os
import sys
import json

# Ensure the mcp folder is in path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from automax import Automax3

mcp = FastMCP(name="Automax Incident MCP Server")

# ── Health check route via FastMCP's official custom_route decorator ──────────
@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok", "service": "automax-mcp"})


# ── Automax client — lazy/non-fatal construction and login ───────────────────
# Bug fix: Automax3() raises RuntimeError at module level if AUTOMAX_BASEURL
# is missing, which crashes the server before it ever binds to port 8002.
# Defer construction so the MCP server always starts.
_auto_instance: Automax3 | None = None

def _get_auto() -> Automax3:
    global _auto_instance
    if _auto_instance is None:
        _auto_instance = Automax3()
    return _auto_instance

def _ensure_logged_in() -> Automax3:
    """Return a logged-in Automax3 instance, constructing/logging in on first use."""
    try:
        client = _get_auto()
    except Exception as e:
        raise RuntimeError(
            f"Automax client failed to initialise — check AUTOMAX_* env vars: {e}"
        ) from e

    if not client.token:
        try:
            client.login()
        except Exception as e:
            raise RuntimeError(
                f"Automax login failed — check AUTOMAX_* env vars: {e}"
            ) from e
    return client


# ── Data models ───────────────────────────────────────────────────────────────

class IncidentCreate(BaseModel):
    title: str
    description: str
    classification: str  # Classification ID from get_classifications
    location: str        # Location ID from get_locations
    workflow: str        # Workflow ID from get_workflows
    priority: Literal["LOW", "MEDIUM", "HIGH"] = "LOW"
    severity: Literal["LOW", "MEDIUM", "HIGH"] = "LOW"
    reporter_name: Optional[str] = None
    reporter_id: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    channel: str = "Chatbot"
    national_id: str = "45"


# ── MCP Tools ─────────────────────────────────────────────────────────────────

@mcp.tool()
def get_classifications():
    """Fetches all valid incident classification categories from AutoMax. Returns IDs and names."""
    return _ensure_logged_in().get_classifications()


@mcp.tool()
def get_locations():
    """Fetches all valid building/location options from AutoMax. Returns IDs and names."""
    return _ensure_logged_in().get_locations()


@mcp.tool()
def get_workflows():
    """Fetches available AutoMax workflows. Required when creating an incident."""
    return _ensure_logged_in().get_workflows()


@mcp.tool()
def get_departments():
    """Fetches department list from AutoMax."""
    return _ensure_logged_in().get_departments()


@mcp.tool()
def create_incident(incident: IncidentCreate):
    """
    Creates a real incident in AutoMax with full details.

    Args:
        title: Incident title/summary
        description: Detailed description of the incident
        classification: Classification ID (from get_classifications)
        location: Location ID (from get_locations)
        workflow: Workflow ID (from get_workflows)
        priority: Priority level - LOW, MEDIUM, or HIGH
        severity: Severity level - LOW, MEDIUM, or HIGH
        reporter_name: Name of the person reporting (optional)
        reporter_id: Reporter national/employee ID (optional)
        latitude: GPS latitude coordinate (optional)
        longitude: GPS longitude coordinate (optional)
    """
    client = _ensure_logged_in()

    coordinates = ""
    if incident.latitude is not None and incident.longitude is not None:
        coordinates = json.dumps({"coordinates": [incident.latitude, incident.longitude]})

    return client.create_incident(
        title=incident.title,
        description=incident.description,
        classification=incident.classification,
        location=incident.location,
        workflow=incident.workflow,
        priority=incident.priority,
        severity=incident.severity,
        reporter_name=incident.reporter_name or "",
        reporter_id=incident.reporter_id or incident.national_id,
        coordinates=coordinates,
        channel=incident.channel,
        national_id=incident.national_id,
    )


@mcp.tool()
def attach_file_to_incident(incident_id: str, file_path: str):
    """
    Attaches a local file to an existing incident in AutoMax.

    Args:
        incident_id: The ID of the existing incident to attach the file to
        file_path: Absolute path to the local file to attach
    """
    return _ensure_logged_in().attach_file_to_incident(
        incident_id=incident_id, file_path=file_path
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Starting Automax Incident MCP Server on http://127.0.0.1:8002/mcp")
    print("Health check available at http://127.0.0.1:8002/health")

    # Attempt early login — non-fatal so the server starts regardless
    try:
        _ensure_logged_in()
        print("✅ Automax login successful at startup")
    except Exception as e:
        print(f"⚠️  Automax login failed at startup: {e}")
        print("   Tools will retry login when first called.")

    mcp.run(
        transport="streamable-http",
        host="127.0.0.1",
        port=8002,
        path="/mcp",
    )
