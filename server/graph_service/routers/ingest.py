import asyncio
from functools import partial

from fastapi import APIRouter, status
from graphiti_core.nodes import EpisodeType  # type: ignore
from graphiti_core.utils.maintenance.graph_data_operations import clear_data  # type: ignore

from graph_service.dto import AddEntityNodeRequest, AddMessagesRequest, Message, Result
from graph_service.entity_types import ENTITY_TYPES
from graph_service.zep_graphiti import ZepGraphitiDep


class AsyncWorker:
    def __init__(self):
        self.queue = asyncio.Queue()
        self.task = None

    async def worker(self):
        print('[AsyncWorker] Worker loop started')
        while True:
            try:
                print(f'[AsyncWorker] Waiting for job... (queue size: {self.queue.qsize()})')
                job = await self.queue.get()
                print(f'[AsyncWorker] Got a job! Executing... (remaining: {self.queue.qsize()})')
                try:
                    # Add 60s timeout to prevent jobs from hanging indefinitely
                    await asyncio.wait_for(job(), timeout=60.0)
                    print('[AsyncWorker] Job completed successfully')
                except asyncio.TimeoutError:
                    print('[AsyncWorker] ERROR: Job timed out after 60 seconds')
                except Exception as e:
                    print(f'[AsyncWorker] ERROR in job execution: {type(e).__name__}: {e}')
                    import traceback
                    traceback.print_exc()
                finally:
                    self.queue.task_done()
            except asyncio.CancelledError:
                print('[AsyncWorker] Worker loop cancelled, exiting')
                break
            except Exception as e:
                print(f'[AsyncWorker] FATAL ERROR in worker loop: {type(e).__name__}: {e}')
                import traceback
                traceback.print_exc()
                break
        print('[AsyncWorker] Worker loop exited')

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
    graphiti: ZepGraphitiDep,
):
    print(f'[POST /messages] Received {len(request.messages)} message(s) for group_id={request.group_id}')

    async def add_messages_task(m: Message):
        print(f'[Task] Processing message: uuid={m.uuid}, role={m.role_type}')
        print(f'[Task] Calling graphiti.add_episode() with group_id={request.group_id}')
        try:
            await graphiti.add_episode(
                uuid=m.uuid,
                group_id=request.group_id,
                name=m.name,
                episode_body=f'[{m.role_type}]: {m.content}',
                reference_time=m.timestamp,
                source=EpisodeType.message,
                source_description=m.source_description,
                entity_types=ENTITY_TYPES,
            )
            print(f'[Task] add_episode() completed successfully')
        except Exception as e:
            print(f'[Task] add_episode() raised exception: {type(e).__name__}: {e}')
            raise
        print(f'[Task] Message processed: uuid={m.uuid}')

    for m in request.messages:
        await async_worker.queue.put(partial(add_messages_task, m))
        print(f'[POST /messages] Queued message {m.uuid} (queue size now: {async_worker.queue.qsize()})')

    return Result(message='Messages added to processing queue', success=True)


@router.post('/entity-node', status_code=status.HTTP_201_CREATED)
async def add_entity_node(
    request: AddEntityNodeRequest,
    graphiti: ZepGraphitiDep,
):
    node = await graphiti.save_entity_node(
        uuid=request.uuid,
        group_id=request.group_id,
        name=request.name,
        summary=request.summary,
    )
    return node


@router.delete('/entity-edge/{uuid}', status_code=status.HTTP_200_OK)
async def delete_entity_edge(uuid: str, graphiti: ZepGraphitiDep):
    await graphiti.delete_entity_edge(uuid)
    return Result(message='Entity Edge deleted', success=True)


@router.delete('/group/{group_id}', status_code=status.HTTP_200_OK)
async def delete_group(group_id: str, graphiti: ZepGraphitiDep):
    await graphiti.delete_group(group_id)
    return Result(message='Group deleted', success=True)


@router.delete('/episode/{uuid}', status_code=status.HTTP_200_OK)
async def delete_episode(uuid: str, graphiti: ZepGraphitiDep):
    await graphiti.delete_episodic_node(uuid)
    return Result(message='Episode deleted', success=True)


@router.post('/clear', status_code=status.HTTP_200_OK)
async def clear(
    graphiti: ZepGraphitiDep,
):
    await clear_data(graphiti.driver)
    await graphiti.build_indices_and_constraints()
    return Result(message='Graph cleared', success=True)
