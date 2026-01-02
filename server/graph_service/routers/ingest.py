import asyncio
import logging
from functools import partial
from typing import Annotated

from fastapi import APIRouter, Depends, status
from graphiti_core.nodes import EpisodeType  # type: ignore
from graphiti_core.utils.maintenance.graph_data_operations import clear_data  # type: ignore

from graph_service.dto import AddEntityNodeRequest, AddMessagesRequest, Message, Result
from graph_service.entity_types import ENTITY_TYPES
from graph_service.events import get_event_bus
from graph_service.zep_graphiti import (
    ZepGraphiti,
    get_graphiti_from_body,
    get_graphiti_from_path,
    get_graphiti_from_query,
)

logger = logging.getLogger(__name__)


class AsyncWorker:
    def __init__(self):
        self.queue = asyncio.Queue()
        self.task = None

    async def emit_queue_status(self):
        """Emit queue status event to all WebSocket clients."""
        queue_size = self.queue.qsize()
        event_bus = get_event_bus()
        # Broadcast to all groups (group_id='*' is a special broadcast group)
        await event_bus.publish(
            event_type='queue.status',
            group_id='*',  # Broadcast to all connected clients
            data={
                'queue_size': queue_size,
                'is_processing': queue_size > 0,
            },
        )

    async def worker(self):
        logger.info('AsyncWorker started - ready to process jobs')
        while True:
            try:
                queue_size = self.queue.qsize()
                logger.debug(f'AsyncWorker - Waiting for job... (queue size: {queue_size})')
                job = await self.queue.get()

                remaining = self.queue.qsize()
                logger.info(f'Processing job (queue: {remaining} remaining)')
                logger.debug(f'Got a job: (size of remaining queue: {remaining})')

                try:
                    # Let LLM_TIMEOUT and EMBEDDING_TIMEOUT handle timeouts
                    await job()
                    logger.info(
                        f'Job completed successfully (queue: {self.queue.qsize()} remaining)'
                    )
                    logger.debug('AsyncWorker - Job completed successfully')
                except Exception as e:
                    logger.error(f'AsyncWorker - ERROR in job execution: {type(e).__name__}: {e}')
                    import traceback

                    traceback.print_exc()
                finally:
                    self.queue.task_done()
                    # Emit queue status after job completes
                    await self.emit_queue_status()
            except asyncio.CancelledError:
                logger.info('AsyncWorker stopped')
                logger.debug('AsyncWorker - Worker loop cancelled, exiting')
                break
            except Exception as e:
                logger.error(f'AsyncWorker - FATAL ERROR in worker loop: {type(e).__name__}: {e}')
                import traceback

                traceback.print_exc()
                break
        logger.debug('AsyncWorker - Worker loop exited')

    async def start(self):
        self.task = asyncio.create_task(self.worker())

    async def stop(self):
        if self.task:
            self.task.cancel()
            await self.task
        while not self.queue.empty():
            self.queue.get_nowait()


async_worker = AsyncWorker()


router = APIRouter()


@router.post('/messages', status_code=status.HTTP_202_ACCEPTED)
async def add_messages(
    request: AddMessagesRequest,
    graphiti: Annotated[ZepGraphiti, Depends(get_graphiti_from_body)],
):
    logger.debug(
        f'POST /messages - Received {len(request.messages)} message(s) for group_id={request.group_id}'
    )

    async def add_messages_task(m: Message):
        logger.debug(f'Task - Processing message: uuid={m.uuid}, role={m.role_type}')
        logger.debug(
            f'Task - Calling graphiti.add_episode_with_events() with group_id={request.group_id}'
        )

        # Extract session_id from source_description if not explicitly provided
        if m.session_id is None and m.source_description:
            from graphiti_core.utils.session_utils import extract_session_id

            extracted_id, cleaned_desc = extract_session_id(m.source_description)
            session_id = extracted_id
            source_desc = cleaned_desc
        else:
            session_id = m.session_id
            source_desc = m.source_description

        try:
            await graphiti.add_episode_with_events(
                uuid=m.uuid,
                group_id=request.group_id,
                name=m.name,
                episode_body=f'[{m.role_type}]: {m.content}',
                reference_time=m.timestamp,
                source=EpisodeType.message,
                source_description=source_desc,
                session_id=session_id,
                entity_types=ENTITY_TYPES,
            )
            logger.debug('Task - add_episode_with_events() completed successfully')
        except Exception as e:
            logger.error(f'Task - add_episode() raised exception: {type(e).__name__}: {e}')
            raise
        logger.debug(f'Task - Message processed: uuid={m.uuid}')

    for m in request.messages:
        await async_worker.queue.put(partial(add_messages_task, m))
        logger.debug(
            f'POST /messages - Queued message {m.uuid} (queue size now: {async_worker.queue.qsize()})'
        )

    queue_size = async_worker.queue.qsize()
    logger.info(f'Added {len(request.messages)} message(s) to queue (total queued: {queue_size})')

    # Emit queue status after adding jobs
    await async_worker.emit_queue_status()

    return Result(message='Messages added to processing queue', success=True)


@router.post('/entity-node', status_code=status.HTTP_201_CREATED)
async def add_entity_node(
    request: AddEntityNodeRequest,
    graphiti: Annotated[ZepGraphiti, Depends(get_graphiti_from_body)],
):
    node = await graphiti.save_entity_node(
        uuid=request.uuid,
        group_id=request.group_id,
        name=request.name,
        summary=request.summary,
    )
    return node


@router.delete('/entity-edge/{uuid}', status_code=status.HTTP_200_OK)
async def delete_entity_edge(
    uuid: str,
    group_id: str,
    graphiti: Annotated[ZepGraphiti, Depends(get_graphiti_from_query)],
):
    await graphiti.delete_entity_edge(uuid)
    return Result(message='Entity Edge deleted', success=True)


@router.delete('/group/{group_id}', status_code=status.HTTP_200_OK)
async def delete_group(
    group_id: str,
    graphiti: Annotated[ZepGraphiti, Depends(get_graphiti_from_path)],
):
    await graphiti.delete_group(group_id)
    return Result(message='Group deleted', success=True)


@router.delete('/episode/{uuid}', status_code=status.HTTP_200_OK)
async def delete_episode(
    uuid: str,
    group_id: str,
    graphiti: Annotated[ZepGraphiti, Depends(get_graphiti_from_query)],
):
    await graphiti.delete_episodic_node(uuid)
    return Result(message='Episode deleted', success=True)


@router.post('/clear', status_code=status.HTTP_200_OK)
async def clear(
    group_id: str,
    graphiti: Annotated[ZepGraphiti, Depends(get_graphiti_from_query)],
):
    await clear_data(graphiti.driver)
    await graphiti.build_indices_and_constraints()
    return Result(message='Graph cleared', success=True)
