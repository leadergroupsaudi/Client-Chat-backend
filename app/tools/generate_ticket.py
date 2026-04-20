"""
Tool: generate_ticket

Creates a general support ticket for issues that are not infrastructure incidents.
Registered with the OpenAI Realtime agent as a callable function tool.

When ``TICKET_USE_MOCK=true`` (or no ``TICKET_API_URL`` is configured) a fake
ticket number is returned without hitting an external API.

Environment variables
---------------------
TICKET_API_URL   Base URL of the ticketing system REST endpoint.
TICKET_API_KEY   Bearer token / API key for the ticketing system.
TICKET_USE_MOCK  Set to "true" to bypass the real API (default: "true").
"""

import logging
import os
import uuid
from typing import Optional

import httpx

logger = logging.getLogger("tool.generate-ticket")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TICKET_API_URL: str = os.getenv("TICKET_API_URL", "")
TICKET_API_KEY: str = os.getenv("TICKET_API_KEY", "")

# Default to mock when no API URL is configured
USE_MOCK: bool = os.getenv("TICKET_USE_MOCK", "true").lower() == "true" or not TICKET_API_URL

_VALID_PRIORITIES = frozenset({"low", "medium", "high", "urgent"})
_VALID_CATEGORIES = frozenset({"general", "billing", "technical", "account", "complaint"})

# ---------------------------------------------------------------------------
# OpenAI Realtime tool schema
# ---------------------------------------------------------------------------

GENERATE_TICKET_SCHEMA: dict = {
    "type": "function",
    "name": "generate_ticket",
    "description": (
        "Create a support ticket for a user's issue. "
        "Use this for general support requests that are NOT infrastructure incidents. "
        "Collect title, description, and optionally priority and category before calling."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "One-sentence summary of the issue.",
            },
            "description": {
                "type": "string",
                "description": "Full description of the user's issue.",
            },
            "priority": {
                "type": "string",
                "enum": ["low", "medium", "high", "urgent"],
                "description": "Ticket priority. Defaults to medium.",
            },
            "category": {
                "type": "string",
                "enum": ["general", "billing", "technical", "account", "complaint"],
                "description": "Issue category. Defaults to general.",
            },
            "contact_name": {
                "type": "string",
                "description": "Name of the person submitting the ticket (optional).",
            },
        },
        "required": ["title", "description"],
    },
}

# ---------------------------------------------------------------------------
# Public tool function
# ---------------------------------------------------------------------------

async def generate_ticket(
    title: str,
    description: str,
    priority: str = "medium",
    category: str = "general",
    contact_name: str = "",
) -> dict:
    """
    Create a support ticket.

    Args:
        title:        Short summary of the issue.
        description:  Full issue description.
        priority:     low | medium | high | urgent.
        category:     general | billing | technical | account | complaint.
        contact_name: Name of the requesting user (optional).

    Returns:
        dict with keys:
            status      "success" or "error"
            ticket_id   Reference number (on success)
            message     Human-readable result sentence
    """
    # Sanitise — fall back to defaults if the model passes an invalid value
    priority = priority.lower() if priority.lower() in _VALID_PRIORITIES else "medium"
    category = category.lower() if category.lower() in _VALID_CATEGORIES else "general"

    logger.info(
        f"[generate_ticket] title={title!r} priority={priority} "
        f"category={category} contact={contact_name!r}"
    )

    if USE_MOCK:
        return _mock_response(title, description, priority, category, contact_name)

    return await _call_ticket_api(title, description, priority, category, contact_name)


# ---------------------------------------------------------------------------
# Implementation helpers
# ---------------------------------------------------------------------------

async def _call_ticket_api(
    title: str,
    description: str,
    priority: str,
    category: str,
    contact_name: str,
) -> dict:
    """POST to the configured external ticketing API."""
    payload = {
        "title": title,
        "description": description,
        "priority": priority,
        "category": category,
        "contact_name": contact_name,
        "source": "realtime_voice_agent",
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {TICKET_API_KEY}",
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(TICKET_API_URL, json=payload, headers=headers)

        if resp.status_code in (200, 201):
            data = resp.json()
            ticket_id = (
                data.get("id")
                or data.get("ticket_id")
                or f"TKT-{_short_id()}"
            )
            logger.info(f"[generate_ticket] Created ticket: {ticket_id}")
            return {
                "status": "success",
                "ticket_id": str(ticket_id),
                "message": (
                    f"Support ticket created successfully. "
                    f"Your ticket number is {ticket_id}. "
                    f"We'll follow up with you shortly."
                ),
            }

        logger.error(f"[generate_ticket] API error {resp.status_code}: {resp.text[:200]}")
        return {
            "status": "error",
            "message": f"Ticketing system returned error {resp.status_code}. Please try again.",
        }

    except Exception as exc:
        logger.error(f"[generate_ticket] Network error: {exc}", exc_info=True)
        return {"status": "error", "message": f"Network error while creating ticket: {exc}"}


def _mock_response(
    title: str,
    description: str,
    priority: str,
    category: str,
    contact_name: str,
) -> dict:
    """Return a mock ticket response for dev/test environments."""
    ticket_id = f"TKT-{_short_id()}"
    logger.info(f"[generate_ticket] MOCK mode — returning {ticket_id}")
    return {
        "status": "success",
        "ticket_id": ticket_id,
        "message": (
            f"Support ticket submitted successfully. "
            f"Your ticket number is {ticket_id}. "
            f"We'll be in touch regarding: {title}."
        ),
    }


def _short_id() -> str:
    """Generate an 8-character uppercase hex ID."""
    return uuid.uuid4().hex[:8].upper()
