from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from graph_service.config import get_settings
from graph_service.events import get_event_bus
from graph_service.routers import ingest, retrieve
from graph_service.routers.ingest import async_worker
from graph_service.websocket import get_ws_manager, router as websocket_router
from graph_service.zep_graphiti import close_graphiti_singleton, initialize_graphiti_singleton


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Startup: Initialize singleton Graphiti instance
    settings = get_settings()
    await initialize_graphiti_singleton(settings)
    # Startup: Start async worker for background job processing
    await async_worker.start()
    # Startup: Register WebSocketManager with EventBus
    event_bus = get_event_bus()
    ws_manager = get_ws_manager()
    await event_bus.subscribe(ws_manager.handle_graph_event)
    yield
    # Shutdown: Unsubscribe WebSocketManager
    await event_bus.unsubscribe(ws_manager.handle_graph_event)
    # Shutdown: Stop async worker
    await async_worker.stop()
    # Shutdown: Close singleton Graphiti instance
    await close_graphiti_singleton()


app = FastAPI(lifespan=lifespan)


app.include_router(retrieve.router)
app.include_router(ingest.router)
app.include_router(websocket_router)


@app.get('/healthcheck')
async def healthcheck():
    return JSONResponse(content={'status': 'healthy'}, status_code=200)
