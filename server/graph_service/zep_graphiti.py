import logging
import os
from typing import Annotated

from fastapi import Depends, HTTPException
from graphiti_core import Graphiti  # type: ignore
from graphiti_core.edges import EntityEdge  # type: ignore
from graphiti_core.errors import EdgeNotFoundError, GroupsEdgesNotFoundError, NodeNotFoundError
from graphiti_core.llm_client import LLMClient  # type: ignore
from graphiti_core.nodes import EntityNode, EpisodicNode  # type: ignore

from graph_service.config import ZepEnvDep
from graph_service.dto import FactResult

logger = logging.getLogger(__name__)


def create_graph_driver(uri: str, user: str, password: str):
    """Create appropriate graph driver based on URI scheme."""
    if uri.startswith('falkordb://'):
        from graphiti_core.driver.falkordb_driver import FalkorDriver
        # Extract host and port from URI
        host_port = uri.replace('falkordb://', '').split('/')[0]
        parts = host_port.split(':')
        host = parts[0]
        port = int(parts[1]) if len(parts) > 1 else 6379
        logger.info(f'Creating FalkorDriver for {host}:{port} (no auth)')
        # FalkorDB typically doesn't use authentication, so don't pass credentials
        return FalkorDriver(host=host, port=port)
    else:
        from graphiti_core.driver.neo4j_driver import Neo4jDriver
        logger.info(f'Creating Neo4jDriver for {uri}')
        return Neo4jDriver(uri, user, password)


class ZepGraphiti(Graphiti):
    def __init__(self, uri: str, user: str, password: str, llm_client: LLMClient | None = None):
        driver = create_graph_driver(uri, user, password)

        # Use OpenAIGenericClient for llama.cpp/Ollama compatibility if not provided
        if llm_client is None:
            from graphiti_core.llm_client.config import LLMConfig
            from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient

            llm_config = LLMConfig(
                api_key=os.getenv("OPENAI_API_KEY", "not-needed"),
                model=os.getenv("OPENAI_MODEL", "meta-llama-3.1-8b-instruct-q4_k_m"),
                base_url=os.getenv("OPENAI_BASE_URL", "http://172.16.0.114:9002/v1"),
            )
            # Set timeout for local LLM inference (default: 120 seconds)
            timeout = float(os.getenv("LLM_TIMEOUT", "120.0"))
            llm_client = OpenAIGenericClient(config=llm_config, max_tokens=16384)
            llm_client.client.timeout = timeout
            logger.info(f'Using OpenAIGenericClient with base_url={llm_config.base_url}, model={llm_config.model}')

        # Configure local embedder using dedicated llama.cpp embedding server
        from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig

        embedder_config = OpenAIEmbedderConfig(
            api_key="not-needed",
            embedding_model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text-v1.5.Q8_0"),
            embedding_dim=int(os.getenv("EMBEDDING_DIM", "768")),
            base_url=os.getenv("EMBEDDING_BASE_URL", "http://172.16.0.114:9003/v1"),
        )
        # Set timeout for local embedding inference (default: 60 seconds)
        embedding_timeout = float(os.getenv("EMBEDDING_TIMEOUT", "60.0"))
        embedder = OpenAIEmbedder(config=embedder_config)
        embedder.client.timeout = embedding_timeout
        logger.info(f'Using local embedder: {embedder.config.embedding_model} ({embedder.config.embedding_dim}d) at {embedder.config.base_url}')

        super().__init__(uri=None, user=None, password=None, llm_client=llm_client, embedder=embedder, graph_driver=driver)

    async def save_entity_node(self, name: str, uuid: str, group_id: str, summary: str = ''):
        new_node = EntityNode(
            name=name,
            uuid=uuid,
            group_id=group_id,
            summary=summary,
        )
        await new_node.generate_name_embedding(self.embedder)
        await new_node.save(self.driver)
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

    async def delete_entity_edge(self, uuid: str):
        try:
            edge = await EntityEdge.get_by_uuid(self.driver, uuid)
            await edge.delete(self.driver)
        except EdgeNotFoundError as e:
            raise HTTPException(status_code=404, detail=e.message) from e

    async def delete_episodic_node(self, uuid: str):
        try:
            episode = await EpisodicNode.get_by_uuid(self.driver, uuid)
            await episode.delete(self.driver)
        except NodeNotFoundError as e:
            raise HTTPException(status_code=404, detail=e.message) from e


async def get_graphiti(settings: ZepEnvDep):
    client = ZepGraphiti(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
    )
    if settings.openai_base_url is not None:
        client.llm_client.config.base_url = settings.openai_base_url
    if settings.openai_api_key is not None:
        client.llm_client.config.api_key = settings.openai_api_key
    if settings.model_name is not None:
        client.llm_client.model = settings.model_name

    try:
        yield client
    finally:
        await client.close()


async def initialize_graphiti(settings: ZepEnvDep):
    client = ZepGraphiti(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
    )
    await client.build_indices_and_constraints()


def get_fact_result_from_edge(edge: EntityEdge):
    return FactResult(
        uuid=edge.uuid,
        name=edge.name,
        fact=edge.fact,
        valid_at=edge.valid_at,
        invalid_at=edge.invalid_at,
        created_at=edge.created_at,
        expired_at=edge.expired_at,
    )


ZepGraphitiDep = Annotated[ZepGraphiti, Depends(get_graphiti)]
