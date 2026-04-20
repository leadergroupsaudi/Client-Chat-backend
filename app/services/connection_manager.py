import json
import time
from typing import Any, Dict, List, Tuple
from fastapi import WebSocket


class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[Dict[str, Any]]] = {}
        # Track last activity timestamp for each websocket: {session_id: {websocket_id: timestamp}}
        self.last_activity: Dict[str, Dict[int, float]] = {}

    async def connect(self, websocket: WebSocket, session_id: str, user_type: str, connection_type: str = "chat"):
        await websocket.accept()
        if session_id not in self.active_connections:
            self.active_connections[session_id] = []
            self.last_activity[session_id] = {}

        self.active_connections[session_id].append({"websocket": websocket, "user_type": user_type, "connection_type": connection_type})

        # Initialize activity timestamp
        ws_id = id(websocket)
        self.last_activity[session_id][ws_id] = time.time()

        print(f"Connected: {user_type} ({connection_type}) to session {session_id}. Total connections for session: {len(self.active_connections[session_id])}")

    def disconnect(self, websocket: WebSocket, session_id: str):
        print(f"[disconnect] Attempting to disconnect from channel '{session_id}'")
        if session_id in self.active_connections:
            connection_to_remove = next((c for c in self.active_connections[session_id] if c["websocket"] == websocket), None)
            if connection_to_remove:
                self.active_connections[session_id].remove(connection_to_remove)

                # Remove activity tracking
                ws_id = id(websocket)
                if session_id in self.last_activity and ws_id in self.last_activity[session_id]:
                    del self.last_activity[session_id][ws_id]

                print(f"[disconnect] ✅ Disconnected {connection_to_remove['user_type']} from channel '{session_id}'. Remaining: {len(self.active_connections.get(session_id, []))}")
                if not self.active_connections[session_id]:
                    del self.active_connections[session_id]
                    if session_id in self.last_activity:
                        del self.last_activity[session_id]
                    print(f"[disconnect] 🗑️ Removed empty channel '{session_id}'")
                print(f"[disconnect] 📊 Active channels after disconnect: {list(self.active_connections.keys())}")
        else:
            print(f"[disconnect] ⚠️ Channel '{session_id}' not found in active connections")

    async def broadcast_to_session(self, session_id: str, message: str, sender_type: str):
        print(f"[ConnectionManager] Broadcasting to session {session_id}. Message: {message[:50]}...")
        if session_id in self.active_connections:
            message_data = json.loads(message)
            if message_data.get('message_type') == 'note':
                connections_to_send = [c for c in self.active_connections[session_id] if c["user_type"] == "agent"]
                print(f"[ConnectionManager] Sending note to {len(connections_to_send)} agent connections in session {session_id}")
            else:
                connections_to_send = self.active_connections[session_id]
                print(f"[ConnectionManager] Sending message to {len(connections_to_send)} connections in session {session_id}")
            
            failed_connections = []

            for connection in connections_to_send:
                try:
                    await connection["websocket"].send_text(message)
                    # print(f"[ConnectionManager] Sent message to websocket: {connection["websocket"]}")
                except Exception as e:
                    print(f"[ConnectionManager] Error sending message to websocket: {e}")
                    failed_connections.append(connection)

            # Clean up dead connections after iteration
            for failed_conn in failed_connections:
                try:
                    self.active_connections[session_id].remove(failed_conn)
                    print(f"[ConnectionManager] 🗑️ Removed dead connection from session {session_id}")
                except ValueError:
                    pass  # Already removed

            # Clean up empty session
            if session_id in self.active_connections and not self.active_connections[session_id]:
                del self.active_connections[session_id]
                print(f"[ConnectionManager] 🗑️ Removed empty session '{session_id}'")
        else:
            print(f"[ConnectionManager] No active connections for session {session_id}")
    
    async def broadcast(self, message: str, channel_id: str):
        print(f"[broadcast] Attempting to broadcast to channel '{channel_id}' (type: {type(channel_id).__name__})")
        print(f"[broadcast] Active connection keys: {list(self.active_connections.keys())}")
        print(f"[broadcast] Active connection key types: {[type(k).__name__ for k in self.active_connections.keys()]}")

        if channel_id in self.active_connections:
            print(f"[broadcast] ✅ Found {len(self.active_connections[channel_id])} connections for channel {channel_id}")

            # Track connections that fail so we can remove them after iteration
            failed_connections = []

            for connection in self.active_connections[channel_id]:
                try:
                    await connection["websocket"].send_text(message)
                    print(f"[broadcast] Sent to {connection['user_type']} via {connection['websocket']}")
                except Exception as e:
                    print(f"[broadcast] ⚠️ Failed to send to {connection['user_type']}: {e}")
                    failed_connections.append(connection)

            # Clean up dead connections after iteration
            for failed_conn in failed_connections:
                try:
                    self.active_connections[channel_id].remove(failed_conn)
                    print(f"[broadcast] 🗑️ Removed dead connection for {failed_conn['user_type']}")
                except ValueError:
                    pass  # Already removed

            # Clean up empty channel
            if channel_id in self.active_connections and not self.active_connections[channel_id]:
                del self.active_connections[channel_id]
                print(f"[broadcast] 🗑️ Removed empty channel '{channel_id}'")
        else:
            print(f"[broadcast] ❌ No connections found for channel '{channel_id}'")

    async def broadcast_bytes_to_session(self, session_id: str, data: bytes):
        """
        Broadcast binary data (audio) to voice connections only for a session.
        Used for sending TTS audio to voice WebSocket connections.

        Args:
            session_id: The session ID to broadcast to
            data: Binary data (audio bytes) to send
        """
        if session_id in self.active_connections:
            # Filter for voice connections only
            voice_connections = [c for c in self.active_connections[session_id] if c.get("connection_type") == "voice"]
            print(f"[broadcast_bytes_to_session] Sending audio to {len(voice_connections)} voice connections in session {session_id}")

            failed_connections = []

            for connection in voice_connections:
                try:
                    await connection["websocket"].send_bytes(data)
                except Exception as e:
                    print(f"[broadcast_bytes_to_session] Error sending audio to websocket: {e}")
                    failed_connections.append(connection)

            # Clean up dead connections after iteration
            for failed_conn in failed_connections:
                try:
                    self.active_connections[session_id].remove(failed_conn)
                    print(f"[broadcast_bytes_to_session] Removed dead voice connection from session {session_id}")
                except ValueError:
                    pass
        else:
            print(f"[broadcast_bytes_to_session] No active connections for session {session_id}")

    async def broadcast_to_company(self, company_id: int, message: str):
        """
        Broadcast a message to all users connected to a company channel.
        This is an alias for broadcast() with company_id converted to string.
        """
        channel_id = str(company_id)
        print(f"[broadcast_to_company] Broadcasting to company {company_id} (channel: '{channel_id}')")
        await self.broadcast(message, channel_id)

    async def broadcast_to_user(self, user_id: int, message: str):
        """
        Broadcast a message to a specific user across all their active sessions.
        Useful for notifications like incoming calls.

        Args:
            user_id: The user ID to send the message to
            message: The message to send
        """
        user_channel = f"user_{user_id}"
        print(f"[broadcast_to_user] Broadcasting to user {user_id} (channel: '{user_channel}')")
        await self.broadcast(message, user_channel)

    async def disconnect_all(self):
        print("[ConnectionManager] Disconnecting all clients...")
        for session_id in list(self.active_connections.keys()):
            for connection in self.active_connections[session_id]:
                try:
                    await connection["websocket"].close(code=1000)
                except Exception as e:
                    print(f"Error closing websocket for session {session_id}: {e}")
        self.active_connections.clear()
        print("[ConnectionManager] All clients disconnected.")

    def has_user_connection(self, session_id: str) -> bool:
        """
        Checks if a session has at least one active user (not agent) connection.
        Returns True if there's an active user connection, False otherwise.
        """
        if session_id not in self.active_connections:
            return False

        user_connections = [c for c in self.active_connections[session_id] if c["user_type"] == "user"]
        return len(user_connections) > 0

    def get_connection_status(self, session_id: str) -> bool:
        """
        Gets the real-time connection status for a session.
        Returns True if client is connected, False otherwise.
        """
        return self.has_user_connection(session_id)

    def update_activity(self, session_id: str, websocket: WebSocket):
        """Update the last activity timestamp for a specific websocket connection"""
        if session_id in self.last_activity:
            ws_id = id(websocket)
            self.last_activity[session_id][ws_id] = time.time()

    def is_preview_session(self, session_id: str) -> bool:
        """Check if session is a preview session based on ID prefix"""
        return session_id.startswith("preview_session_")

    def get_session_timeout(self, session_id: str) -> int:
        """Get timeout in seconds based on session type"""
        from app.core.config import settings
        if self.is_preview_session(session_id):
            return settings.WS_PREVIEW_SESSION_TIMEOUT
        return settings.WS_REGULAR_SESSION_TIMEOUT

    def get_inactive_connections(self) -> List[Tuple[str, WebSocket, float]]:
        """
        Returns list of (session_id, websocket, idle_time) for connections
        that have exceeded their timeout period
        """
        current_time = time.time()
        inactive = []

        for session_id in list(self.active_connections.keys()):
            timeout = self.get_session_timeout(session_id)

            if session_id in self.last_activity:
                for connection in self.active_connections[session_id]:
                    ws_id = id(connection["websocket"])
                    last_seen = self.last_activity[session_id].get(ws_id, current_time)
                    idle_time = current_time - last_seen

                    if idle_time > timeout:
                        inactive.append((session_id, connection["websocket"], idle_time))

        return inactive

manager = ConnectionManager()
