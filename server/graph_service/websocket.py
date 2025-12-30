import json
import logging
from collections import defaultdict

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from graph_service.events import GraphEvent

logger = logging.getLogger(__name__)


class WebSocketManager:
    """Manages WebSocket connections and broadcasts events to subscribed clients."""

    def __init__(self):
        # Maps group_id to set of WebSocket connections
        self.active_connections: dict[str, set[WebSocket]] = defaultdict(set)

    async def connect(self, websocket: WebSocket, group_id: str) -> None:
        """Accept a WebSocket connection and add it to the group."""
        await websocket.accept()
        self.active_connections[group_id].add(websocket)
        logger.info(
            f'WebSocket connected: group={group_id}, '
            f'total_in_group={len(self.active_connections[group_id])}'
        )

    async def disconnect(self, websocket: WebSocket, group_id: str) -> None:
        """Remove a WebSocket connection and cleanup empty groups."""
        if websocket in self.active_connections[group_id]:
            self.active_connections[group_id].remove(websocket)
            logger.info(
                f'WebSocket disconnected: group={group_id}, '
                f'remaining_in_group={len(self.active_connections[group_id])}'
            )

        # Cleanup empty groups
        if not self.active_connections[group_id]:
            del self.active_connections[group_id]
            logger.debug(f'Removed empty group: {group_id}')

    async def broadcast_to_group(self, group_id: str, message: str) -> None:
        """
        Send a message to all connections in a group.

        Automatically removes dead connections that fail to receive.
        """
        connections = self.active_connections.get(group_id, set())
        if not connections:
            logger.debug(f'No connections for group {group_id}, skipping broadcast')
            return

        logger.debug(f'Broadcasting to group {group_id}: {len(connections)} connection(s)')

        # Track dead connections to remove after iteration
        dead_connections = set()

        for websocket in connections:
            try:
                await websocket.send_text(message)
            except Exception as e:
                logger.warning(
                    f'Failed to send to WebSocket in group {group_id}: {type(e).__name__}: {e}'
                )
                dead_connections.add(websocket)

        # Remove dead connections
        for websocket in dead_connections:
            await self.disconnect(websocket, group_id)

    async def broadcast_to_all(self, message: str) -> None:
        """
        Broadcast a message to all connected clients regardless of group.

        Used for global events like queue status updates.
        """
        total_connections = sum(len(conns) for conns in self.active_connections.values())
        if total_connections == 0:
            logger.debug('No active connections, skipping broadcast')
            return

        logger.debug(f'Broadcasting to all groups: {total_connections} connection(s) across {len(self.active_connections)} group(s)')

        # Track dead connections by group
        dead_by_group: dict[str, set[WebSocket]] = defaultdict(set)

        for group_id, connections in self.active_connections.items():
            for websocket in connections:
                try:
                    await websocket.send_text(message)
                except Exception as e:
                    logger.warning(
                        f'Failed to send to WebSocket in group {group_id}: {type(e).__name__}: {e}'
                    )
                    dead_by_group[group_id].add(websocket)

        # Remove dead connections
        for group_id, dead_connections in dead_by_group.items():
            for websocket in dead_connections:
                await self.disconnect(websocket, group_id)

    async def handle_graph_event(self, event: GraphEvent) -> None:
        """
        EventBus callback: convert event to JSON and broadcast to group.

        This method is registered as a subscriber with the EventBus and
        receives all graph modification events.

        Special handling:
        - group_id='*' broadcasts to all connected clients
        - Otherwise broadcasts only to the specified group
        """
        message = json.dumps(event.to_dict())

        if event.group_id == '*':
            # Broadcast to all connected clients
            await self.broadcast_to_all(message)
        else:
            # Broadcast only to specific group
            await self.broadcast_to_group(event.group_id, message)


# Singleton instance
_ws_manager: WebSocketManager | None = None


def get_ws_manager() -> WebSocketManager:
    """Get the singleton WebSocketManager instance."""
    global _ws_manager
    if _ws_manager is None:
        _ws_manager = WebSocketManager()
    return _ws_manager


# WebSocket router
router = APIRouter()


@router.websocket('/ws/{group_id}')
async def websocket_endpoint(websocket: WebSocket, group_id: str):
    """
    WebSocket endpoint for real-time graph notifications.

    Clients connect to /ws/{group_id} and receive JSON notifications
    for all graph modifications within that group.

    Example usage:
        const ws = new WebSocket('ws://localhost:8000/ws/my-group');
        ws.onmessage = (event) => {
            const notification = JSON.parse(event.data);
            console.log(notification.event_type, notification.data);
        };
    """
    manager = get_ws_manager()
    await manager.connect(websocket, group_id)

    try:
        # Keep connection alive and listen for client messages
        # (currently we don't expect clients to send messages, but we need to keep the loop running)
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        logger.info(f'WebSocket disconnected normally: group={group_id}')
    except Exception as e:
        logger.error(f'WebSocket error for group {group_id}: {type(e).__name__}: {e}')
    finally:
        await manager.disconnect(websocket, group_id)
