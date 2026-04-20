"""
Session Manager for the OpenAI Realtime Voice Agent.

Manages in-memory session state for voice interactions.  Each session is
identified by a UUID and carries:

    - Conversation history  (list of {role, content} dicts)
    - User context          (arbitrary key/value pairs collected during the call)
    - Active status flag
    - Creation / last-activity timestamps
    - Optional agent_id / company_id for DB-backend integration

Not suitable for multi-process deployments without a shared backing store
(Redis, etc.), but correct for single-process FastAPI workers.
"""

import asyncio
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger("session-manager")

# How long an idle session is kept before cleanup_expired() removes it
SESSION_EXPIRY_MINUTES: int = int(os.getenv("SESSION_EXPIRY_MINUTES", "60"))


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

@dataclass
class SessionState:
    """
    Full in-memory state for a single voice interaction session.

    Attributes:
        session_id:             Unique UUID string assigned at creation.
        is_active:              False once the session has been explicitly closed.
        created_at:             UTC timestamp of creation.
        last_activity:          UTC timestamp of the most recent update.
        conversation_history:   Ordered list of {"role": "user"|"assistant", "content": str}.
        user_context:           Arbitrary data collected during the conversation
                                (e.g. caller_name, classification, location).
        agent_id:               Optional reference to the DB agent record.
        company_id:             Optional reference to the owning company.
    """

    session_id: str
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)

    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    user_context: Dict[str, Any] = field(default_factory=dict)

    agent_id: Optional[int] = None
    company_id: Optional[int] = None


# ---------------------------------------------------------------------------
# Session manager
# ---------------------------------------------------------------------------

class SessionManager:
    """
    In-memory registry of all active voice sessions.

    All mutating methods acquire an asyncio.Lock so the manager is safe to use
    from multiple concurrent WebSocket handlers within the same event loop.

    Usage::

        manager = SessionManager()

        # Create a new session
        state = await manager.create_session(agent_id=1, company_id=42)
        print(state.session_id)   # "d3f1a2b4-..."

        # Update collected data
        await manager.update_context(state.session_id, caller_name="John")

        # Append a transcript turn
        await manager.add_message(state.session_id, "user", "I need help.")

        # Close when done
        await manager.close_session(state.session_id)
    """

    def __init__(self) -> None:
        self._sessions: Dict[str, SessionState] = {}
        self._lock: asyncio.Lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    async def create_session(
        self,
        *,
        agent_id: Optional[int] = None,
        company_id: Optional[int] = None,
        initial_context: Optional[Dict[str, Any]] = None,
    ) -> SessionState:
        """
        Create a new session with a fresh UUID.

        Args:
            agent_id:        Optional DB agent identifier.
            company_id:      Optional company identifier.
            initial_context: Pre-populate the user_context dict (e.g. from
                             a pre-call form or JWT claims).

        Returns:
            The newly created SessionState.
        """
        session_id = str(uuid.uuid4())
        state = SessionState(
            session_id=session_id,
            agent_id=agent_id,
            company_id=company_id,
            user_context=dict(initial_context) if initial_context else {},
        )
        async with self._lock:
            self._sessions[session_id] = state
        logger.info(
            f"[SessionManager] Created session={session_id} "
            f"agent={agent_id} company={company_id}"
        )
        return state

    async def get_session(self, session_id: str) -> Optional[SessionState]:
        """
        Retrieve a session by ID.

        Returns:
            The SessionState, or None if it does not exist.
        """
        async with self._lock:
            return self._sessions.get(session_id)

    async def update_context(self, session_id: str, **kwargs: Any) -> bool:
        """
        Merge key/value pairs into the session's user_context and update
        the last_activity timestamp.

        Args:
            session_id: Target session UUID.
            **kwargs:   Key/value data to merge.

        Returns:
            True on success, False if the session was not found.
        """
        async with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                logger.warning(f"[SessionManager] update_context: session {session_id!r} not found")
                return False
            state.user_context.update(kwargs)
            state.last_activity = datetime.utcnow()
        return True

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
    ) -> bool:
        """
        Append a conversation turn to the session history.

        Args:
            session_id: Target session UUID.
            role:       "user" or "assistant".
            content:    Transcript text of the turn.

        Returns:
            True on success, False if the session was not found.
        """
        async with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                logger.warning(f"[SessionManager] add_message: session {session_id!r} not found")
                return False
            state.conversation_history.append({"role": role, "content": content})
            state.last_activity = datetime.utcnow()
        return True

    async def close_session(self, session_id: str) -> bool:
        """
        Mark a session as inactive without removing it from memory.

        The session remains queryable (e.g. for audit) until deleted or cleaned up.

        Returns:
            True on success, False if not found.
        """
        async with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                return False
            state.is_active = False
            state.last_activity = datetime.utcnow()
        logger.info(f"[SessionManager] Closed session={session_id}")
        return True

    async def delete_session(self, session_id: str) -> bool:
        """
        Permanently remove a session from memory.

        Returns:
            True if it existed and was removed, False otherwise.
        """
        async with self._lock:
            if session_id not in self._sessions:
                return False
            del self._sessions[session_id]
        logger.info(f"[SessionManager] Deleted session={session_id}")
        return True

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    async def cleanup_expired(
        self,
        expiry_minutes: int = SESSION_EXPIRY_MINUTES,
    ) -> int:
        """
        Remove sessions that are inactive OR have been idle longer than
        expiry_minutes.

        Args:
            expiry_minutes: Idle threshold in minutes (default: SESSION_EXPIRY_MINUTES).

        Returns:
            Number of sessions removed.
        """
        cutoff = datetime.utcnow() - timedelta(minutes=expiry_minutes)
        async with self._lock:
            expired_ids = [
                sid
                for sid, s in self._sessions.items()
                if not s.is_active or s.last_activity < cutoff
            ]
            for sid in expired_ids:
                del self._sessions[sid]

        if expired_ids:
            logger.info(
                f"[SessionManager] Cleaned up {len(expired_ids)} expired session(s)"
            )
        return len(expired_ids)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def active_count(self) -> int:
        """Number of currently active (not yet closed) sessions."""
        return sum(1 for s in self._sessions.values() if s.is_active)

    @property
    def total_count(self) -> int:
        """Total sessions in memory (active + closed but not yet cleaned up)."""
        return len(self._sessions)


# ---------------------------------------------------------------------------
# Process-wide singleton
# ---------------------------------------------------------------------------

# Import this instance wherever session management is needed.
session_manager = SessionManager()
