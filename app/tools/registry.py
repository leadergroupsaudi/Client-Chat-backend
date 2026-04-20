"""
Central tool registry.

Maps tool names to their OpenAI Realtime API schemas and async Python handlers.
Add new tools here — they will be automatically available for selection
in the Voice Agents configuration UI.
"""
from typing import Dict, Any, List, Tuple, Callable

from app.tools.create_incident import CREATE_INCIDENT_SCHEMA, create_incident
from app.tools.generate_ticket import GENERATE_TICKET_SCHEMA, generate_ticket

# Registry structure:
#   name → { schema, handler, label, description }
#
# schema   : OpenAI function-tool schema (same format used in session.update)
# handler  : async callable that executes the tool
# label    : human-readable display name for the UI
# description : short summary shown in the tool selection dropdown
TOOL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "create_incident": {
        "schema": CREATE_INCIDENT_SCHEMA,
        "handler": create_incident,
        "label": "Create Incident",
        "description": "Create an infrastructure incident report in the Automax3 system",
    },
    "generate_ticket": {
        "schema": GENERATE_TICKET_SCHEMA,
        "handler": generate_ticket,
        "label": "Generate Support Ticket",
        "description": "Create a support ticket for a customer issue",
    },
}


def list_available_tools() -> List[Dict[str, str]]:
    """Return UI-friendly list of all registered tools."""
    return [
        {
            "name": name,
            "label": entry["label"],
            "description": entry["description"],
        }
        for name, entry in TOOL_REGISTRY.items()
    ]


def resolve_tools(names: List[str]) -> Tuple[List[dict], Dict[str, Callable]]:
    """
    Given a list of tool names, return:
      schemas  – list of OpenAI function-tool dicts (for session.update)
      handlers – { name: async_callable } (for server-side execution)

    Unknown names are silently skipped.
    """
    schemas: List[dict] = []
    handlers: Dict[str, Callable] = {}
    for name in names:
        entry = TOOL_REGISTRY.get(name)
        if entry:
            schemas.append(entry["schema"])
            handlers[name] = entry["handler"]
    return schemas, handlers


async def execute_tool(tool_name: str, tool_args: dict) -> dict:
    """
    Execute a registered tool by name with the given arguments.
    Returns the tool result dict.
    Raises ValueError for unknown tools.
    """
    entry = TOOL_REGISTRY.get(tool_name)
    if not entry:
        raise ValueError(f"Unknown tool: {tool_name!r}. Available: {list(TOOL_REGISTRY)}")
    handler = entry["handler"]
    result = await handler(**tool_args)
    return result
