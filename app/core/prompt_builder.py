"""
Prompt Builder for the OpenAI Realtime Voice Agent.

Generates strong, context-aware system prompts from a structured configuration
and optional live session state.  Separating prompt logic from agent logic
keeps both clean and independently testable.

Usage::

    from app.core.prompt_builder import PromptBuilder
    from app.core.session_manager import session_manager

    builder = PromptBuilder(
        agent_name="EPM 940 Support Agent",
        agent_description="Handles infrastructure incident reports for EPM 940.",
    )

    state = await session_manager.get_session(session_id)
    system_prompt = builder.build_system_prompt(session=state)
"""

import logging
from typing import Optional

from app.core.session_manager import SessionState

logger = logging.getLogger("prompt-builder")


class PromptBuilder:
    """
    Constructs the system prompt and optional context-injection strings
    used by the OpenAI Realtime voice agent.

    Args:
        agent_name:        Display name used in the agent's self-introduction.
        agent_description: Core identity/role sentence injected at the top.
        language:          Primary response language (default: English).
    """

    def __init__(
        self,
        *,
        agent_name: str = "AI Voice Support Agent",
        agent_description: str = (
            "You are a professional and friendly AI voice support agent. "
            "Your role is to assist users by understanding their issues, "
            "gathering the necessary information, and taking the appropriate action."
        ),
        language: str = "English",
    ) -> None:
        self.agent_name = agent_name
        self.agent_description = agent_description
        self.language = language

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def build_system_prompt(
        self,
        session: Optional[SessionState] = None,
        extra_instructions: Optional[str] = None,
    ) -> str:
        """
        Build the full system prompt, optionally enriched with session context.

        The prompt is structured in clearly labelled sections so the model can
        follow each rule independently.

        Args:
            session:            Active session state.  When provided, any already-
                                collected user_context is injected so the model
                                does not re-ask for known information.
            extra_instructions: Domain-specific rules appended verbatim at the end
                                (e.g. hardcoded classification lists, escalation paths).

        Returns:
            Complete system prompt string (plain text, no markdown).
        """
        parts: list[str] = []

        # ── 1. Identity ──────────────────────────────────────────────────
        parts.append(f"You are {self.agent_name}.")
        parts.append(self.agent_description)
        parts.append("")

        # ── 2. Voice conversation rules ──────────────────────────────────
        parts.append("VOICE CONVERSATION RULES:")
        parts.append(
            "- This is a real-time VOICE conversation. "
            "Keep every response to 1-3 short sentences."
        )
        parts.append("- Do NOT use bullet points, numbered lists, or markdown formatting.")
        parts.append("- Speak naturally, exactly as a human support agent would.")
        parts.append("- Acknowledge the user's answer before moving to the next question.")
        parts.append("- Be empathetic, patient, and professional at all times.")
        parts.append(
            f"- Respond in {self.language} unless the user clearly speaks another language."
        )
        parts.append("")

        # ── 3. Information gathering ─────────────────────────────────────
        parts.append("INFORMATION GATHERING:")
        parts.append("- Ask ONE question at a time — never bundle multiple questions.")
        parts.append("- If an answer is unclear or incomplete, ask for clarification politely.")
        parts.append("- Confirm critical details (name, location) by reading them back.")
        parts.append("")

        # ── 4. Tool usage ────────────────────────────────────────────────
        parts.append("TOOL USAGE:")
        parts.append(
            "- When you have ALL required information, call the appropriate tool immediately."
        )
        parts.append("- Do NOT call a tool unless every required field is collected.")
        parts.append(
            "- After a tool executes successfully, tell the user the outcome "
            "(e.g. ticket number) in one sentence."
        )
        parts.append(
            "- If a tool returns an error, apologise briefly and offer to try again."
        )
        parts.append("")

        # ── 5. Incident reporting flow ───────────────────────────────────
        parts.append("INCIDENT REPORTING FLOW:")
        parts.append("When a user wants to report an issue, collect in this exact order:")
        parts.append("1. Caller's full name")
        parts.append("2. Type of incident (classification)")
        parts.append("3. Location of the incident")
        parts.append("4. Brief description (optional but helpful)")
        parts.append("5. Severity — LOW, MEDIUM, HIGH, or CRITICAL (default LOW)")
        parts.append(
            "Once all required information is confirmed, call the create_incident tool."
        )
        parts.append("")

        # ── 6. General support ticket flow ───────────────────────────────
        parts.append("SUPPORT TICKET FLOW:")
        parts.append("For general (non-incident) requests, collect:")
        parts.append("1. A short description of the issue")
        parts.append("2. Priority level (low / medium / high / urgent)")
        parts.append("3. Category (general / billing / technical / account / complaint)")
        parts.append("Then call the generate_ticket tool.")
        parts.append("")

        # ── 7. Session context injection ─────────────────────────────────
        if session and session.user_context:
            parts.append("ALREADY COLLECTED INFORMATION (do NOT re-ask for these):")
            for key, value in session.user_context.items():
                parts.append(f"- {key}: {value}")
            parts.append("")

        # ── 8. Conversation history summary ──────────────────────────────
        if session and session.conversation_history:
            turn_count = len(session.conversation_history)
            parts.append(
                f"CONVERSATION PROGRESS: {turn_count} turn(s) have already occurred."
            )
            parts.append("")

        # ── 9. Ending the call ───────────────────────────────────────────
        parts.append("ENDING THE CALL:")
        parts.append(
            "- After completing the user's request, ask if there is anything else."
        )
        parts.append("- If the user is done, thank them and say goodbye warmly.")
        parts.append("")

        # ── 10. Domain-specific extras ───────────────────────────────────
        if extra_instructions:
            parts.append("ADDITIONAL INSTRUCTIONS:")
            parts.append(extra_instructions)
            parts.append("")

        return "\n".join(parts)

    def build_context_injection(self, session: SessionState) -> str:
        """
        Build a short hidden context string to prepend to the user's first turn.

        Unlike the system prompt (which is static per session), this is injected
        once at call-start to give the model awareness of any pre-loaded data
        (e.g. user profile from the DB, previous ticket history).

        Args:
            session: Active session state.

        Returns:
            Plain-text context block, or an empty string if there is nothing to inject.
        """
        if not session.user_context:
            return ""

        lines = ["[CONTEXT — provided before the call started:"]
        for k, v in session.user_context.items():
            lines.append(f"  {k} = {v}")
        lines.append("]")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Default singleton for standard voice agent use
# ---------------------------------------------------------------------------

default_prompt_builder = PromptBuilder()
