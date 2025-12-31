import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from graph_service.config import get_settings
from graph_service.events import get_event_bus
from graph_service.routers import ingest, retrieve
from graph_service.routers.ingest import async_worker
from graph_service.websocket import get_ws_manager, router as websocket_router
from graph_service.zep_graphiti import close_connection_pool, initialize_connection_pool


# Configure application logging from LOG_LEVEL environment variable
log_level_name = os.getenv('LOG_LEVEL', 'INFO').upper()
log_level = getattr(logging, log_level_name, logging.INFO)
logging.basicConfig(
    level=log_level,
    format='%(levelname)s:     %(name)s - %(message)s',
    force=True  # Override any existing configuration
)

# Suppress noisy HTTP request logs from httpx
logging.getLogger('httpx').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
logger.info(f'Application logging configured with level: {log_level_name}')


async def periodic_queue_status_broadcaster():
    """
    Periodically broadcast queue status to all WebSocket clients.

    Runs every 10 seconds to keep clients informed of processing status.
    """
    logger.info('Starting periodic queue status broadcaster')
    try:
        while True:
            await asyncio.sleep(10)  # Wait 10 seconds between broadcasts
            await async_worker.emit_queue_status()
    except asyncio.CancelledError:
        logger.info('Periodic queue status broadcaster stopped')
        raise


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Startup: Initialize connection pool with shared LLM/embedder
    settings = get_settings()
    await initialize_connection_pool(settings)
    # Startup: Start async worker for background job processing
    await async_worker.start()
    # Startup: Register WebSocketManager with EventBus
    event_bus = get_event_bus()
    ws_manager = get_ws_manager()
    await event_bus.subscribe(ws_manager.handle_graph_event)
    # Startup: Start periodic queue status broadcaster
    queue_broadcaster_task = asyncio.create_task(periodic_queue_status_broadcaster())
    yield
    # Shutdown: Stop periodic queue status broadcaster
    queue_broadcaster_task.cancel()
    try:
        await queue_broadcaster_task
    except asyncio.CancelledError:
        pass
    # Shutdown: Unsubscribe WebSocketManager
    await event_bus.unsubscribe(ws_manager.handle_graph_event)
    # Shutdown: Stop async worker
    await async_worker.stop()
    # Shutdown: Close connection pool
    await close_connection_pool()


app = FastAPI(lifespan=lifespan)


app.include_router(retrieve.router)
app.include_router(ingest.router)
app.include_router(websocket_router)


@app.get('/healthcheck')
async def healthcheck():
    return JSONResponse(content={'status': 'healthy'}, status_code=200)
