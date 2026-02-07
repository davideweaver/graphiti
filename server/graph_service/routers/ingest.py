import asyncio
import logging
from functools import partial
from typing import Annotated

from fastapi import APIRouter, Depends, Query, status
from graphiti_core.nodes import EpisodeType  # type: ignore
from graphiti_core.utils.maintenance.graph_data_operations import clear_data  # type: ignore

from graph_service.dto import (
    AddContentRequest,
    AddEntityNodeRequest,
    AddMessagesRequest,
    Message,
    Result,
    UpdateEntityEdgeRequest,
)
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
        self.processing_count = 0
        self._processing_lock = asyncio.Lock()  # Thread-safe counter updates

    async def emit_queue_status(self):
        """Emit queue status event to all WebSocket clients."""
        queue_size = self.queue.qsize()
        async with self._processing_lock:
            processing_count = self.processing_count
        total_pending = queue_size + processing_count

        event_bus = get_event_bus()
        # Broadcast to all groups (group_id='*' is a special broadcast group)
        await event_bus.publish(
            event_type='queue.status',
            group_id='*',  # Broadcast to all connected clients
            data={
                'queue_size': queue_size,  # Items waiting
                'processing_count': processing_count,  # Items being processed
                'total_pending': total_pending,  # Total work remaining
                'is_processing': total_pending > 0,
            },
        )

    async def worker(self):
        logger.info('AsyncWorker started - ready to process jobs')
        while True:
            try:
                queue_size = self.queue.qsize()
                logger.debug(f'AsyncWorker - Waiting for job... (queue size: {queue_size})')
                job = await self.queue.get()

                # Increment processing count and emit status
                async with self._processing_lock:
                    self.processing_count += 1

                remaining = self.queue.qsize()
                logger.info(
                    f'Processing job (queue: {remaining} waiting, '
                    f'processing: {self.processing_count})'
                )

                # Emit status after incrementing processing count
                await self.emit_queue_status()

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
                    # Decrement processing count
                    async with self._processing_lock:
                        self.processing_count -= 1
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

        # Extract or use project_name
        # Priority: 1) Explicit project_name in payload, 2) Parse from source_description
        if m.project_name is not None:
            # Use provided project name, lowercase it
            project_name = m.project_name.lower()
        elif source_desc:
            # Try to extract from source_description
            from graphiti_core.utils.project_utils import extract_project_name

            extracted_project, cleaned_source = extract_project_name(source_desc)
            project_name = extracted_project  # Already lowercased by extract_project_name
            # Update source_desc to cleaned version (without project suffix)
            source_desc = cleaned_source
        else:
            project_name = None

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
                project_name=project_name,
                project_path=m.project_path,
                programmatic=m.programmatic,
                skip_extraction=request.skip_extraction,
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


@router.post('/content', status_code=status.HTTP_202_ACCEPTED)
async def add_content(
    request: AddContentRequest,
    graphiti: Annotated[ZepGraphiti, Depends(get_graphiti_from_body)],
):
    """Add content with source tracking.

    This endpoint:
    1. Creates a Source node to track where the content came from
    2. Links Source to Project via PART_OF_PROJECT relationship
    3. Creates an episode from the content (async)
    4. Links episode to Source via FROM_SOURCE relationship
    5. Links episode to Project via IN_PROJECT relationship (handled by Graphiti)
    6. Extracts facts and entities via Graphiti (async)

    Returns the source_uuid for tracking extraction progress.
    """
    import uuid as uuid_lib
    from datetime import datetime, timezone

    logger.debug(
        f'POST /content - Received content for group_id={request.group_id}, '
        f'source_name={request.source_name}, source_type={request.source_type}'
    )

    # Generate UUID for Source entity
    source_uuid = str(uuid_lib.uuid4())

    # Determine project name (lowercase, default to "_general" if None)
    project_name = request.project_name.lower() if request.project_name else '_general'

    source_summary = f'{request.source_type.title()} source: {request.source_name}'

    # Create Source node using direct Cypher query with custom labels and attributes
    # Sources are NOT entities - they're metadata nodes like Episodes
    # We can't use save_entity_node() because it doesn't support custom labels or attributes
    now_iso = datetime.now(timezone.utc).isoformat()

    create_source_query = '''
        CREATE (s:Source {
            uuid: $uuid,
            name: $name,
            group_id: $group_id,
            summary: $summary,
            source_type: $source_type,
            created_at: $created_at
        })
        SET s += $metadata
        RETURN s
    '''

    await graphiti.driver.execute_query(
        create_source_query,
        uuid=source_uuid,
        name=request.source_name,
        group_id=request.group_id,
        summary=source_summary,
        source_type=request.source_type,
        created_at=now_iso,
        metadata=request.source_metadata,
    )

    logger.info(f'Created Source entity: uuid={source_uuid}, name={request.source_name}')

    # Link Source to Project
    try:
        from graphiti_core.utils.maintenance.project_operations import (
            get_or_create_project_node,
            link_source_to_project,
        )

        # Get or create project node
        project_node = await get_or_create_project_node(
            driver=graphiti.driver,
            group_id=request.group_id,
            project_name=project_name,
            project_path=None,
        )

        # Save project node
        await project_node.save(graphiti.driver)

        # Link source to project
        await link_source_to_project(
            driver=graphiti.driver,
            source_uuid=source_uuid,
            project_uuid=project_node.uuid,
        )

        logger.info(f'Linked source {source_uuid} to project {project_name}')

    except Exception as e:
        # Don't block content ingestion if project link fails
        logger.error(
            f'Failed to link source {source_uuid} to project {project_name}: '
            f'{type(e).__name__}: {e}'
        )

    # Queue content processing task
    async def process_content_task():
        logger.debug(f'Task - Processing content from source: {source_uuid}')

        try:
            # Create episode from content
            # Let Graphiti generate the UUID by passing None
            results = await graphiti.add_episode_with_events(
                uuid=None,  # Let Graphiti generate UUID
                group_id=request.group_id,
                name=f'Content from {request.source_name}',
                episode_body=request.content,
                reference_time=datetime.now(timezone.utc),
                source=EpisodeType.message,
                source_description=f'{request.source_type}:{request.source_name}',
                session_id=None,  # Content imports don't use sessions
                project_name=project_name,
                project_path=None,
                skip_extraction=False,  # Always extract facts/entities
                entity_types=ENTITY_TYPES,
                previous_episode_uuids=[],  # No previous context for external content
            )

            episode_uuid = results.episode.uuid
            logger.info(
                f'Created episode: uuid={episode_uuid} for source={source_uuid}'
            )

            # Create FROM_SOURCE relationship (episode → source)
            # Note: This requires adding the relationship via a custom Cypher query
            # since graphiti doesn't have a built-in method for this
            await graphiti.driver.execute_query(
                '''
                MATCH (e:Episodic {uuid: $episode_uuid, group_id: $group_id})
                MATCH (s:Source {uuid: $source_uuid, group_id: $group_id})
                CREATE (e)-[:FROM_SOURCE {created_at: $created_at}]->(s)
                ''',
                episode_uuid=episode_uuid,
                source_uuid=source_uuid,
                group_id=request.group_id,
                created_at=datetime.now(timezone.utc).isoformat(),
            )

            logger.info(f'Created FROM_SOURCE relationship: {episode_uuid} -> {source_uuid}')

            logger.debug(f'Task - Content processing complete for source: {source_uuid}')

        except Exception as e:
            logger.error(
                f'Task - Failed to process content from source {source_uuid}: '
                f'{type(e).__name__}: {e}'
            )
            raise

    # Queue the processing task
    await async_worker.queue.put(process_content_task)
    logger.debug(
        f'POST /content - Queued content processing (queue size now: {async_worker.queue.qsize()})'
    )

    # Emit queue status
    await async_worker.emit_queue_status()

    return {
        'source_uuid': source_uuid,
        'message': 'Content queued for processing',
        'success': True,
    }


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


@router.patch('/entity-edge/{uuid}', status_code=status.HTTP_200_OK)
async def update_entity_edge(
    uuid: str,
    request: UpdateEntityEdgeRequest,
    graphiti: Annotated[ZepGraphiti, Depends(get_graphiti_from_body)],
):
    """Update an entity edge (fact) by UUID.

    Currently supports updating the fact text.

    Args:
        uuid: UUID of the entity edge to update
        request: Update request with fact text and group_id

    Returns:
        Result with success status
    """
    await graphiti.update_entity_edge(uuid, request.fact, request.group_id)
    return Result(message='Entity Edge updated', success=True)


@router.delete('/entity-edge/{uuid}', status_code=status.HTTP_200_OK)
async def delete_entity_edge(
    uuid: str,
    group_id: str,
    graphiti: Annotated[ZepGraphiti, Depends(get_graphiti_from_query)],
):
    await graphiti.delete_entity_edge(uuid)
    return Result(message='Entity Edge deleted', success=True)


@router.post('/group/{group_id}/backup', status_code=status.HTTP_200_OK)
async def backup_group(
    group_id: str,
    graphiti: Annotated[ZepGraphiti, Depends(get_graphiti_from_path)],
    target_group_id: str = Query(..., description='Target group ID for the backup'),
):
    """Create a backup of a graph with a new group_id.

    This copies the entire FalkorDB database and updates all group_id properties
    in the copied data to match the new database name.

    Args:
        group_id: The group ID to backup (source)
        target_group_id: The new group ID for the backup

    Returns:
        Success message with backup statistics
    """
    stats = await graphiti.backup_group(group_id, target_group_id)
    return {
        'message': f'Backup created: {target_group_id}',
        'success': True,
        'stats': stats,
    }


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


@router.delete('/entities/{group_id}/{uuid}', status_code=status.HTTP_200_OK)
async def delete_entity(
    group_id: str,
    uuid: str,
    graphiti: Annotated[ZepGraphiti, Depends(get_graphiti_from_path)],
):
    """Delete an entity node by UUID.

    This will cascade-delete all edges (RELATES_TO relationships) connected to this entity.
    A WebSocket event will be emitted for real-time updates.

    Args:
        group_id: The group ID
        uuid: UUID of the entity to delete

    Returns:
        Success confirmation message

    Raises:
        HTTPException: 404 if entity not found
    """
    await graphiti.delete_entity_node(uuid)
    return Result(message='Entity deleted', success=True)


@router.post('/clear', status_code=status.HTTP_200_OK)
async def clear(
    group_id: str,
    graphiti: Annotated[ZepGraphiti, Depends(get_graphiti_from_query)],
):
    await clear_data(graphiti.driver)
    await graphiti.build_indices_and_constraints()
    return Result(message='Graph cleared', success=True)
