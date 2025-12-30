import asyncio
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)


@dataclass
class GraphEvent:
    """Event representing a graph modification."""
    event_type: str
    group_id: str
    data: dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        """Convert event to JSON-serializable dictionary."""
        return {
            'event_type': self.event_type,
            'group_id': self.group_id,
            'data': self.data,
            'timestamp': self.timestamp.isoformat(),
        }


EventCallback = Callable[[GraphEvent], Awaitable[None]]


class EventBus:
    """In-process async event bus using fire-and-forget pattern."""

    def __init__(self):
        self._subscribers: list[EventCallback] = []

    async def subscribe(self, callback: EventCallback) -> None:
        """Register a callback to receive all events."""
        if callback not in self._subscribers:
            self._subscribers.append(callback)
            logger.info(f'Subscribed callback: {callback.__name__}')

    async def unsubscribe(self, callback: EventCallback) -> None:
        """Unregister a callback."""
        if callback in self._subscribers:
            self._subscribers.remove(callback)
            logger.info(f'Unsubscribed callback: {callback.__name__}')

    async def publish(self, event_type: str, group_id: str, data: dict[str, Any]) -> None:
        """
        Publish an event to all subscribers using fire-and-forget pattern.

        This method does not block - it creates async tasks for each subscriber
        and returns immediately. Exceptions in subscribers are logged but don't
        affect the publisher or other subscribers.
        """
        event = GraphEvent(event_type=event_type, group_id=group_id, data=data)

        logger.debug(
            f'Publishing event: type={event_type}, group={group_id}, '
            f'subscribers={len(self._subscribers)}'
        )

        for callback in self._subscribers:
            # Fire-and-forget: create task and don't await
            task = asyncio.create_task(self._safe_callback(callback, event))
            # Prevent task from being garbage collected
            task.add_done_callback(lambda t: None)

    async def _safe_callback(self, callback: EventCallback, event: GraphEvent) -> None:
        """
        Execute callback with error handling.

        Exceptions are logged but don't propagate to prevent one subscriber
        from affecting others.
        """
        try:
            await callback(event)
        except Exception as e:
            logger.error(
                f'Error in event callback {callback.__name__}: {type(e).__name__}: {e}',
                exc_info=True,
            )


# Singleton instance
_event_bus: EventBus | None = None


def get_event_bus() -> EventBus:
    """Get the singleton EventBus instance."""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus
