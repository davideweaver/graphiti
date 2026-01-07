import asyncio
import logging
import os

from fastapi import HTTPException, Path, Query, Request
from graphiti_core import Graphiti  # type: ignore
from graphiti_core.edges import EntityEdge  # type: ignore
from graphiti_core.errors import EdgeNotFoundError, GroupsEdgesNotFoundError, NodeNotFoundError
from graphiti_core.llm_client import LLMClient  # type: ignore
from graphiti_core.nodes import EntityNode, EpisodicNode  # type: ignore

from graph_service.config import Settings
from graph_service.dto import EntityNodeResponse, FactResult
from graph_service.events import get_event_bus

logger = logging.getLogger(__name__)


def create_graph_driver(uri: str, user: str, password: str, database: str = 'dave-weaver'):
    """Create appropriate graph driver based on URI scheme."""
    if uri.startswith('falkordb://'):
        from falkordb.asyncio import FalkorDB
        from graphiti_core.driver.falkordb_driver import FalkorDriver

        # Extract host and port from URI
        host_port = uri.replace('falkordb://', '').split('/')[0]
        parts = host_port.split(':')
        host = parts[0]
        port = int(parts[1]) if len(parts) > 1 else 6379
        logger.info(
            f'Creating FalkorDriver for {host}:{port} using graph "{database}" with timeouts'
        )
        # Create FalkorDB client with timeout configuration
        falkor_client = FalkorDB(
            host=host,
            port=port,
            username=user if user != 'default' else None,
            password=password if password != 'password' else None,
            socket_timeout=30.0,  # 30s timeout for operations
            socket_connect_timeout=10.0,  # 10s timeout for connection
        )
        return FalkorDriver(host=host, port=port, database=database, falkor_db=falkor_client)
    else:
        from graphiti_core.driver.neo4j_driver import Neo4jDriver

        logger.info(f'Creating Neo4jDriver for {uri}')
        return Neo4jDriver(uri, user, password)


class ZepGraphiti(Graphiti):
    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        llm_client: LLMClient | None = None,
        database: str = 'dave-weaver',
    ):
        driver = create_graph_driver(uri, user, password, database=database)

        # Use OpenAIGenericClient for llama.cpp/Ollama compatibility if not provided
        if llm_client is None:
            from graphiti_core.llm_client.config import LLMConfig
            from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient

            llm_config = LLMConfig(
                api_key=os.getenv('OPENAI_API_KEY', 'not-needed'),
                model=os.getenv('OPENAI_MODEL', 'meta-llama-3.1-8b-instruct-q4_k_m'),
                base_url=os.getenv('OPENAI_BASE_URL', 'http://172.16.0.114:9002/v1'),
            )
            # Set timeout for local LLM inference (default: 120 seconds)
            timeout = float(os.getenv('LLM_TIMEOUT', '120.0'))
            llm_client = OpenAIGenericClient(config=llm_config, max_tokens=16384)
            llm_client.client.timeout = timeout
            logger.info(
                f'Using OpenAIGenericClient with base_url={llm_config.base_url}, model={llm_config.model}'
            )

        # Configure local embedder using dedicated llama.cpp embedding server
        from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig

        embedder_config = OpenAIEmbedderConfig(
            api_key='not-needed',
            embedding_model=os.getenv('EMBEDDING_MODEL', 'nomic-embed-text-v1.5.Q8_0'),
            embedding_dim=int(os.getenv('EMBEDDING_DIM', '768')),
            base_url=os.getenv('EMBEDDING_BASE_URL', 'http://172.16.0.114:9003/v1'),
        )
        # Set timeout for local embedding inference (default: 60 seconds)
        embedding_timeout = float(os.getenv('EMBEDDING_TIMEOUT', '60.0'))
        embedder = OpenAIEmbedder(config=embedder_config)
        embedder.client.timeout = embedding_timeout
        logger.info(
            f'Using local embedder: {embedder.config.embedding_model} ({embedder.config.embedding_dim}d) at {embedder.config.base_url}'
        )

        super().__init__(
            uri=None,
            user=None,
            password=None,
            llm_client=llm_client,
            embedder=embedder,
            graph_driver=driver,
        )

    async def save_entity_node(self, name: str, uuid: str, group_id: str, summary: str = ''):
        new_node = EntityNode(
            name=name,
            uuid=uuid,
            group_id=group_id,
            summary=summary,
        )
        await new_node.generate_name_embedding(self.embedder)
        await new_node.save(self.driver)

        # Emit event for WebSocket notifications
        event_bus = get_event_bus()
        await event_bus.publish(
            event_type='entity.created',
            group_id=group_id,
            data={
                'uuid': uuid,
                'name': name,
                'summary': summary,
                'labels': new_node.labels,
                'created_at': new_node.created_at.isoformat(),
            },
        )

        return new_node

    async def get_entity_edge(self, uuid: str):
        try:
            edge = await EntityEdge.get_by_uuid(self.driver, uuid)
            return edge
        except EdgeNotFoundError as e:
            raise HTTPException(status_code=404, detail=e.message) from e

    async def delete_group(self, group_id: str):
        try:
            edges = await EntityEdge.get_by_group_ids(self.driver, [group_id])
        except GroupsEdgesNotFoundError:
            logger.warning(f'No edges found for group {group_id}')
            edges = []

        nodes = await EntityNode.get_by_group_ids(self.driver, [group_id])

        episodes = await EpisodicNode.get_by_group_ids(self.driver, [group_id])

        for edge in edges:
            await edge.delete(self.driver)

        for node in nodes:
            await node.delete(self.driver)

        for episode in episodes:
            await episode.delete(self.driver)

        # Emit event for WebSocket notifications
        event_bus = get_event_bus()
        await event_bus.publish(
            event_type='group.deleted',
            group_id=group_id,
            data={
                'deleted_edges': len(edges),
                'deleted_nodes': len(nodes),
                'deleted_episodes': len(episodes),
            },
        )

    async def delete_entity_edge(self, uuid: str):
        try:
            edge = await EntityEdge.get_by_uuid(self.driver, uuid)
            group_id = edge.group_id  # Capture before delete
            await edge.delete(self.driver)

            # Emit event for WebSocket notifications
            event_bus = get_event_bus()
            await event_bus.publish(
                event_type='edge.deleted',
                group_id=group_id,
                data={'uuid': uuid},
            )
        except EdgeNotFoundError as e:
            raise HTTPException(status_code=404, detail=e.message) from e

    async def delete_episodic_node(self, uuid: str):
        try:
            episode = await EpisodicNode.get_by_uuid(self.driver, uuid)
            group_id = episode.group_id  # Capture before delete
            await episode.delete(self.driver)

            # Emit event for WebSocket notifications
            event_bus = get_event_bus()
            await event_bus.publish(
                event_type='episode.deleted',
                group_id=group_id,
                data={'uuid': uuid},
            )
        except NodeNotFoundError as e:
            raise HTTPException(status_code=404, detail=e.message) from e

    async def add_episode_with_events(self, **kwargs):
        """
        Wrapper around add_episode that emits events for created entities/edges.

        This method ensures WebSocket notifications are sent after the async worker
        completes processing of episodic memories.
        """
        group_id = kwargs['group_id']

        # Call parent method
        results = await self.add_episode(**kwargs)

        # Emit event for created episode
        event_bus = get_event_bus()
        await event_bus.publish(
            event_type='episode.created',
            group_id=group_id,
            data={
                'uuid': results.episode.uuid,
                'name': results.episode.name,
                'session_id': results.episode.session_id,
            },
        )

        return results


# Connection pool - maps group_id to Graphiti instance
_graphiti_pool: dict[str, ZepGraphiti] = {}
_pool_locks: dict[str, asyncio.Lock] = {}
_pool_lock = asyncio.Lock()  # Master lock for pool structure
_settings: Settings | None = None


async def initialize_connection_pool(settings: Settings):
    """Initialize the connection pool (stores settings for instance creation)."""
    global _settings

    if _settings is not None:
        logger.warning('Connection pool already initialized, skipping')
        return

    logger.info('Initializing Graphiti connection pool...')
    _settings = settings
    logger.info('Connection pool initialized successfully')


async def get_or_create_graphiti_instance(group_id: str) -> ZepGraphiti:
    """Get or create a Graphiti instance for the given group_id (graph name)."""
    # Fast path: instance already exists
    if group_id in _graphiti_pool:
        return _graphiti_pool[group_id]

    # Get or create lock for this group_id
    async with _pool_lock:
        if group_id not in _pool_locks:
            _pool_locks[group_id] = asyncio.Lock()
        group_lock = _pool_locks[group_id]

    # Create instance with per-graph lock (double-check pattern)
    async with group_lock:
        # Double-check: another coroutine may have created it
        if group_id in _graphiti_pool:
            return _graphiti_pool[group_id]

        logger.info(f'Creating new Graphiti instance for graph "{group_id}"')

        # Create instance with its own LLM client and embedder (no sharing)
        instance = ZepGraphiti(
            uri=_settings.neo4j_uri,
            user=_settings.neo4j_user,
            password=_settings.neo4j_password,
            database=group_id,  # Use group_id as FalkorDB database name
            llm_client=None,  # Each instance creates its own client (no concurrent state issues)
        )

        # Apply settings overrides to this instance's LLM client
        if _settings.openai_base_url is not None:
            instance.llm_client.config.base_url = _settings.openai_base_url
        if _settings.openai_api_key is not None:
            instance.llm_client.config.api_key = _settings.openai_api_key
        if _settings.model_name is not None:
            instance.llm_client.model = _settings.model_name

        logger.info(
            f'Instance LLM client: base_url={instance.llm_client.config.base_url}, model={instance.llm_client.model}'
        )
        logger.info(
            f'Instance embedder: {instance.embedder.config.embedding_model} ({instance.embedder.config.embedding_dim}d)'
        )

        # Build indices on first access
        logger.info(f'Building indices for graph "{group_id}"...')
        await instance.build_indices_and_constraints()

        _graphiti_pool[group_id] = instance
        logger.info(f'Graph "{group_id}" ready (pool size: {len(_graphiti_pool)})')
        return instance


async def close_connection_pool():
    """Close all Graphiti instances in the connection pool."""
    logger.info(f'Closing connection pool ({len(_graphiti_pool)} graphs)...')

    for group_id, instance in _graphiti_pool.items():
        logger.info(f'Closing graph "{group_id}"...')
        await instance.close()

    _graphiti_pool.clear()
    _pool_locks.clear()
    logger.info('Connection pool closed')


async def get_graphiti_from_body(request: Request) -> ZepGraphiti:
    """Dependency to extract group_id from request body and return Graphiti instance."""
    try:
        body = await request.json()
        group_id = body.get('group_id')
        if not group_id:
            raise HTTPException(status_code=400, detail='group_id is required in request body')
        return await get_or_create_graphiti_instance(group_id)
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=400, detail=f'Failed to parse request body: {e}')


async def get_graphiti_from_path(group_id: str = Path(...)) -> ZepGraphiti:
    """Dependency to get Graphiti instance from path parameter."""
    return await get_or_create_graphiti_instance(group_id)


async def get_graphiti_from_query(group_id: str = Query(...)) -> ZepGraphiti:
    """Dependency to get Graphiti instance from query parameter."""
    return await get_or_create_graphiti_instance(group_id)


def get_fact_result_from_edge(
    edge: EntityEdge,
    similarity_score: float | None = None,
    source_entity: EntityNodeResponse | None = None,
    target_entity: EntityNodeResponse | None = None,
    episodes: list | None = None,
):
    """
    Convert an EntityEdge to a FactResult.

    Args:
        edge: The EntityEdge to convert
        similarity_score: Optional similarity/reranker score (0.0-1.0) indicating relevance
        source_entity: Optional full details of the source entity
        target_entity: Optional full details of the target entity
        episodes: Optional list of episodes that created/mentioned this fact

    Returns:
        FactResult with all edge attributes and optional provenance data
    """
    return FactResult(
        uuid=edge.uuid,
        name=edge.name,
        fact=edge.fact,
        valid_at=edge.valid_at,
        invalid_at=edge.invalid_at,
        created_at=edge.created_at,
        expired_at=edge.expired_at,
        similarity_score=similarity_score,
        source_node_uuid=edge.source_node_uuid if (source_entity or target_entity or episodes) else None,
        target_node_uuid=edge.target_node_uuid if (source_entity or target_entity or episodes) else None,
        source_entity=source_entity,
        target_entity=target_entity,
        episodes=episodes,
    )


def get_entity_node_response(node: EntityNode):
    return EntityNodeResponse(
        uuid=node.uuid,
        name=node.name,
        group_id=node.group_id,
        summary=node.summary,
        labels=node.labels,
        attributes=node.attributes,
        created_at=node.created_at,
    )
