"""
Realtime Voice Agent — Tools Package.

Each module in this package exposes:
    - An async callable that the agent invokes when the model triggers a tool.
    - A ``*_SCHEMA`` dict in the OpenAI Realtime function-tool format, used to
      register the tool with the model during session configuration.

Available tools
---------------
create_incident
    Report an infrastructure incident to the Automax3 system (or a mock).
    Schema: CREATE_INCIDENT_SCHEMA

generate_ticket
    Create a general support ticket.
    Schema: GENERATE_TICKET_SCHEMA

Quick-start::

    from app.tools import (
        create_incident, CREATE_INCIDENT_SCHEMA,
        generate_ticket, GENERATE_TICKET_SCHEMA,
    )

    agent = RealtimeVoiceAgent(
        tools=[CREATE_INCIDENT_SCHEMA, GENERATE_TICKET_SCHEMA],
        tool_handlers={
            "create_incident": create_incident,
            "generate_ticket": generate_ticket,
        },
    )
"""

from app.tools.create_incident import CREATE_INCIDENT_SCHEMA, create_incident
from app.tools.generate_ticket import GENERATE_TICKET_SCHEMA, generate_ticket

__all__ = [
    "create_incident",
    "CREATE_INCIDENT_SCHEMA",
    "generate_ticket",
    "GENERATE_TICKET_SCHEMA",
]
