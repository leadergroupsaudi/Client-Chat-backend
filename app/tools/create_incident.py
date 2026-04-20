"""
Tool: create_incident

Creates an infrastructure incident report in the Automax3 system.
Registered with the OpenAI Realtime agent as a callable function tool.

The function validates input, resolves human-readable names to internal IDs,
authenticates with Automax3, and POSTs the incident record.  When
``AUTOMAX3_USE_MOCK=true`` it returns a fake incident ID without hitting
the external API — useful for development and testing.

Environment variables
---------------------
AUTOMAX3_BASEURL      Base URL for the Automax3 API.
AUTOMAX3_USERNAME     Client credentials username (used for OAuth2).
AUTOMAX3_PASSWORD     Client credentials password.
AUTOMAX3_USE_MOCK     Set to "true" to bypass the real API.
"""

import logging
import os
import time
import uuid
from datetime import datetime
from typing import Optional

import httpx

logger = logging.getLogger("tool.create-incident")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

AUTOMAX3_BASEURL = os.getenv("AUTOMAX3_BASEURL", "https://epmstg.automaxsw.com")
AUTOMAX3_USERNAME = os.getenv("AUTOMAX3_USERNAME", "474062237214310401")
AUTOMAX3_PASSWORD = os.getenv(
    "AUTOMAX3_PASSWORD",
    "FFtq2lE3V3HrLO9DrjQAd6yxb2i4Xehq1zWRK3zELuewQOgOw0p59bx4bkun0jdI",
)
USE_MOCK: bool = os.getenv("AUTOMAX3_USE_MOCK", "false").lower() == "true"

# Hardcoded classification / location maps.
# In production these should be fetched from the Automax3 API and cached,
# but matching workflow_voice_agent.py we keep them as constants for now.
_CLASSIFICATION_MAP: dict[str, str] = {
    "manholes":         "cls_manholes_id",
    "street lights":    "cls_streetlights_id",
    "street furniture": "cls_streetfurniture_id",
    "potholes":         "cls_potholes_id",
    "barriers":         "cls_barriers_id",
}

_LOCATION_MAP: dict[str, str] = {
    "dammam":       "loc_dammam_id",
    "dammam east":  "loc_dammam_east_id",
    "dammam west":  "loc_dammam_west_id",
}

# ---------------------------------------------------------------------------
# OpenAI Realtime tool schema
# ---------------------------------------------------------------------------

CREATE_INCIDENT_SCHEMA: dict = {
    "type": "function",
    "name": "create_incident",
    "description": (
        "Create an infrastructure incident report in the system. "
        "ONLY call this tool after you have collected: "
        "caller name, classification type, and location. "
        "Description and criticality are optional but recommended."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "caller_name": {
                "type": "string",
                "description": "Full name of the person reporting the incident.",
            },
            "classification": {
                "type": "string",
                "description": (
                    "Incident type. Must be one of: "
                    "Manholes, Street Lights, Street Furniture, Potholes, Barriers."
                ),
            },
            "location": {
                "type": "string",
                "description": (
                    "Where the incident occurred. Must be one of: "
                    "Dammam, Dammam East, Dammam West."
                ),
            },
            "description": {
                "type": "string",
                "description": "Optional description of the incident.",
            },
            "criticality": {
                "type": "string",
                "enum": ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
                "description": "Severity level. Defaults to LOW if not specified.",
            },
        },
        "required": ["caller_name", "classification", "location"],
    },
}

# ---------------------------------------------------------------------------
# Public tool function
# ---------------------------------------------------------------------------

async def create_incident(
    caller_name: str,
    classification: str,
    location: str,
    description: str = "",
    criticality: str = "LOW",
) -> dict:
    """
    Create an infrastructure incident report.

    Args:
        caller_name:    Full name of the reporter.
        classification: Incident type (Manholes, Street Lights, etc.).
        location:       Where the incident occurred.
        description:    Additional details (optional).
        criticality:    Severity — LOW | MEDIUM | HIGH | CRITICAL.

    Returns:
        dict with keys:
            status      "success" or "error"
            incident_id Reference number (on success)
            message     Human-readable result sentence
    """
    logger.info(
        f"[create_incident] caller={caller_name!r} "
        f"classification={classification!r} location={location!r} "
        f"criticality={criticality}"
    )

    if USE_MOCK:
        return _mock_response(caller_name, classification, location, description, criticality)

    return await _call_automax3(caller_name, classification, location, description, criticality)


# ---------------------------------------------------------------------------
# Implementation helpers
# ---------------------------------------------------------------------------

async def _call_automax3(
    caller_name: str,
    classification: str,
    location: str,
    description: str,
    criticality: str,
) -> dict:
    """Authenticate with Automax3 and POST the incident record."""

    # Step 1 — authenticate
    token = await _get_token()
    if not token:
        return {"status": "error", "message": "Authentication with Automax3 failed. Please try again."}

    # Step 2 — resolve human-readable names to internal IDs
    classification_id = _lookup(_CLASSIFICATION_MAP, classification)
    if not classification_id:
        available = ", ".join(k.title() for k in _CLASSIFICATION_MAP)
        return {
            "status": "error",
            "message": f"Classification '{classification}' not recognised. Available: {available}.",
        }

    location_id = _lookup(_LOCATION_MAP, location)
    if not location_id:
        available = ", ".join(k.title() for k in _LOCATION_MAP)
        return {
            "status": "error",
            "message": f"Location '{location}' not recognised. Available: {available}.",
        }

    # Step 3 — POST incident
    url = (
        f"{AUTOMAX3_BASEURL}/api/compose/namespace/431842611944685569"
        f"/module/431842611943440385/record/"
    )
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }
    body = {
        "meta": {},
        "records": [],
        "values": [
            {"name": "Channel",              "value": "Voice Agent"},
            {"name": "Criticality",          "value": criticality},
            {"name": "Caller_name",          "value": caller_name},
            {"name": "Last_call_date",       "value": datetime.utcnow().isoformat() + "Z"},
            {"name": "Classification",       "value": classification_id},
            {"name": "Incident_Description", "value": description or "Reported via voice agent"},
            {"name": "Primary_Location",     "value": location_id},
            {"name": "Status",               "value": ""},
            {"name": "Assigned_To",          "value": "425635139776282625"},
            {"name": "Comments",             "value": _comment_json(caller_name)},
        ],
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(url, json=body, headers=headers)

        if resp.status_code == 200:
            data = resp.json()
            # Automax3 wraps the record ID at different depths depending on the version
            incident_id = (
                data.get("response", {}).get("record", {}).get("recordID")
                or data.get("response", {}).get("recordID")
                or f"INC-{int(time.time())}"
            )
            logger.info(f"[create_incident] Incident created: {incident_id}")
            return {
                "status": "success",
                "incident_id": str(incident_id),
                "message": (
                    f"Incident created successfully. "
                    f"Your reference number is {incident_id}."
                ),
            }

        logger.error(f"[create_incident] Automax3 error {resp.status_code}: {resp.text[:200]}")
        return {
            "status": "error",
            "message": f"Could not create incident (server error {resp.status_code}). Please try again.",
        }

    except Exception as exc:
        logger.error(f"[create_incident] Network error: {exc}", exc_info=True)
        return {"status": "error", "message": f"Network error while creating incident: {exc}"}


async def _get_token() -> Optional[str]:
    """Obtain an OAuth2 client-credentials token from Automax3."""
    url = f"{AUTOMAX3_BASEURL}/auth/oauth2/token"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                url,
                data={"grant_type": "client_credentials", "scope": "profile api"},
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                auth=(AUTOMAX3_USERNAME, AUTOMAX3_PASSWORD),
            )
        if resp.status_code == 200:
            return resp.json().get("access_token")
        logger.error(f"[create_incident] Auth failed: HTTP {resp.status_code}")
        return None
    except Exception as exc:
        logger.error(f"[create_incident] Auth error: {exc}", exc_info=True)
        return None


def _lookup(mapping: dict[str, str], name: str) -> Optional[str]:
    """Case-insensitive lookup in a name → ID mapping."""
    return mapping.get(name.lower().strip())


def _comment_json(caller_name: str) -> str:
    """Build the Comments JSON string that Automax3 expects."""
    import json as _json
    return _json.dumps({
        "created": datetime.utcnow().isoformat() + "Z",
        "comment": "Created via realtime voice agent",
        "author": "realtime-voice-agent",
        "name": caller_name,
    })


def _mock_response(
    caller_name: str,
    classification: str,
    location: str,
    description: str,
    criticality: str,
) -> dict:
    """Return a mock incident response for dev/test environments."""
    incident_id = f"MOCK-{uuid.uuid4().hex[:8].upper()}"
    logger.info(f"[create_incident] MOCK mode — returning {incident_id}")
    return {
        "status": "success",
        "incident_id": incident_id,
        "message": (
            f"Incident created successfully. "
            f"Your reference number is {incident_id}. "
            f"Summary: {classification} at {location}, reported by {caller_name}."
        ),
    }
