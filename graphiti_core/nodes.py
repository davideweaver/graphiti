"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from time import time
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field
from typing_extensions import LiteralString

from graphiti_core.driver.driver import (
    GraphDriver,
    GraphProvider,
)
from graphiti_core.embedder import EmbedderClient
from graphiti_core.errors import NodeNotFoundError
from graphiti_core.helpers import parse_db_date
from graphiti_core.models.nodes.node_db_queries import (
    COMMUNITY_NODE_RETURN,
    COMMUNITY_NODE_RETURN_NEPTUNE,
    EPISODIC_NODE_RETURN,
    EPISODIC_NODE_RETURN_NEPTUNE,
    SESSION_NODE_RETURN,
    SESSION_NODE_RETURN_NEPTUNE,
    get_community_node_save_query,
    get_entity_node_return_query,
    get_entity_node_save_query,
    get_episode_node_save_query,
    get_project_node_save_query,
    get_session_node_save_query,
)
from graphiti_core.utils.datetime_utils import utc_now

logger = logging.getLogger(__name__)


class EpisodeType(Enum):
    """
    Enumeration of different types of episodes that can be processed.

    This enum defines the various sources or formats of episodes that the system
    can handle. It's used to categorize and potentially handle different types
    of input data differently.

    Attributes:
    -----------
    message : str
        Represents a standard message-type episode. The content for this type
        should be formatted as "actor: content". For example, "user: Hello, how are you?"
        or "assistant: I'm doing well, thank you for asking."
    json : str
        Represents an episode containing a JSON string object with structured data.
    text : str
        Represents a plain text episode.
    """

    message = 'message'
    json = 'json'
    text = 'text'

    @staticmethod
    def from_str(episode_type: str):
        if episode_type == 'message':
            return EpisodeType.message
        if episode_type == 'json':
            return EpisodeType.json
        if episode_type == 'text':
            return EpisodeType.text
        logger.error(f'Episode type: {episode_type} not implemented')
        raise NotImplementedError


class Node(BaseModel, ABC):
    uuid: str = Field(default_factory=lambda: str(uuid4()))
    name: str = Field(description='name of the node')
    group_id: str = Field(description='partition of the graph')
    labels: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: utc_now())

    @abstractmethod
    async def save(self, driver: GraphDriver): ...

    async def delete(self, driver: GraphDriver):
        if driver.graph_operations_interface:
            return await driver.graph_operations_interface.node_delete(self, driver)

        match driver.provider:
            case GraphProvider.NEO4J:
                records, _, _ = await driver.execute_query(
                    """
                    MATCH (n {uuid: $uuid})
                    WHERE n:Entity OR n:Episodic OR n:Community
                    OPTIONAL MATCH (n)-[r]-()
                    WITH collect(r.uuid) AS edge_uuids, n
                    DETACH DELETE n
                    RETURN edge_uuids
                    """,
                    uuid=self.uuid,
                )

            case GraphProvider.KUZU:
                for label in ['Episodic', 'Community']:
                    await driver.execute_query(
                        f"""
                        MATCH (n:{label} {{uuid: $uuid}})
                        DETACH DELETE n
                        """,
                        uuid=self.uuid,
                    )
                # Entity edges are actually nodes in Kuzu, so simple `DETACH DELETE` will not work.
                # Explicitly delete the "edge" nodes first, then the entity node.
                await driver.execute_query(
                    """
                    MATCH (n:Entity {uuid: $uuid})-[:RELATES_TO]->(e:RelatesToNode_)
                    DETACH DELETE e
                    """,
                    uuid=self.uuid,
                )
                await driver.execute_query(
                    """
                    MATCH (n:Entity {uuid: $uuid})
                    DETACH DELETE n
                    """,
                    uuid=self.uuid,
                )
            case _:  # FalkorDB, Neptune
                for label in ['Entity', 'Episodic', 'Community']:
                    await driver.execute_query(
                        f"""
                        MATCH (n:{label} {{uuid: $uuid}})
                        DETACH DELETE n
                        """,
                        uuid=self.uuid,
                    )

        logger.debug(f'Deleted Node: {self.uuid}')

    def __hash__(self):
        return hash(self.uuid)

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.uuid == other.uuid
        return False

    @classmethod
    async def delete_by_group_id(cls, driver: GraphDriver, group_id: str, batch_size: int = 100):
        if driver.graph_operations_interface:
            return await driver.graph_operations_interface.node_delete_by_group_id(
                cls, driver, group_id, batch_size
            )

        match driver.provider:
            case GraphProvider.NEO4J:
                async with driver.session() as session:
                    await session.run(
                        """
                        MATCH (n:Entity|Episodic|Community {group_id: $group_id})
                        CALL (n) {
                            DETACH DELETE n
                        } IN TRANSACTIONS OF $batch_size ROWS
                        """,
                        group_id=group_id,
                        batch_size=batch_size,
                    )

            case GraphProvider.KUZU:
                for label in ['Episodic', 'Community']:
                    await driver.execute_query(
                        f"""
                        MATCH (n:{label} {{group_id: $group_id}})
                        DETACH DELETE n
                        """,
                        group_id=group_id,
                    )
                # Entity edges are actually nodes in Kuzu, so simple `DETACH DELETE` will not work.
                # Explicitly delete the "edge" nodes first, then the entity node.
                await driver.execute_query(
                    """
                    MATCH (n:Entity {group_id: $group_id})-[:RELATES_TO]->(e:RelatesToNode_)
                    DETACH DELETE e
                    """,
                    group_id=group_id,
                )
                await driver.execute_query(
                    """
                    MATCH (n:Entity {group_id: $group_id})
                    DETACH DELETE n
                    """,
                    group_id=group_id,
                )
            case _:  # FalkorDB, Neptune
                for label in ['Entity', 'Episodic', 'Community']:
                    await driver.execute_query(
                        f"""
                        MATCH (n:{label} {{group_id: $group_id}})
                        DETACH DELETE n
                        """,
                        group_id=group_id,
                    )

    @classmethod
    async def delete_by_uuids(cls, driver: GraphDriver, uuids: list[str], batch_size: int = 100):
        if driver.graph_operations_interface:
            return await driver.graph_operations_interface.node_delete_by_uuids(
                cls, driver, uuids, group_id=None, batch_size=batch_size
            )

        match driver.provider:
            case GraphProvider.FALKORDB:
                for label in ['Entity', 'Episodic', 'Community']:
                    await driver.execute_query(
                        f"""
                        MATCH (n:{label})
                        WHERE n.uuid IN $uuids
                        DETACH DELETE n
                        """,
                        uuids=uuids,
                    )
            case GraphProvider.KUZU:
                for label in ['Episodic', 'Community']:
                    await driver.execute_query(
                        f"""
                        MATCH (n:{label})
                        WHERE n.uuid IN $uuids
                        DETACH DELETE n
                        """,
                        uuids=uuids,
                    )
                # Entity edges are actually nodes in Kuzu, so simple `DETACH DELETE` will not work.
                # Explicitly delete the "edge" nodes first, then the entity node.
                await driver.execute_query(
                    """
                    MATCH (n:Entity)-[:RELATES_TO]->(e:RelatesToNode_)
                    WHERE n.uuid IN $uuids
                    DETACH DELETE e
                    """,
                    uuids=uuids,
                )
                await driver.execute_query(
                    """
                    MATCH (n:Entity)
                    WHERE n.uuid IN $uuids
                    DETACH DELETE n
                    """,
                    uuids=uuids,
                )
            case _:  # Neo4J, Neptune
                async with driver.session() as session:
                    # Collect all edge UUIDs before deleting nodes
                    await session.run(
                        """
                        MATCH (n:Entity|Episodic|Community)
                        WHERE n.uuid IN $uuids
                        MATCH (n)-[r]-()
                        RETURN collect(r.uuid) AS edge_uuids
                        """,
                        uuids=uuids,
                    )

                    # Now delete the nodes in batches
                    await session.run(
                        """
                        MATCH (n:Entity|Episodic|Community)
                        WHERE n.uuid IN $uuids
                        CALL (n) {
                            DETACH DELETE n
                        } IN TRANSACTIONS OF $batch_size ROWS
                        """,
                        uuids=uuids,
                        batch_size=batch_size,
                    )

    @classmethod
    async def get_by_uuid(cls, driver: GraphDriver, uuid: str): ...

    @classmethod
    async def get_by_uuids(cls, driver: GraphDriver, uuids: list[str]): ...


class EpisodicNode(Node):
    source: EpisodeType = Field(description='source type')
    source_description: str = Field(description='description of the data source')
    content: str = Field(description='raw episode data')
    valid_at: datetime = Field(
        description='datetime of when the original document was created',
    )
    entity_edges: list[str] = Field(
        description='list of entity edges referenced in this episode',
        default_factory=list,
    )
    session_id: str | None = Field(
        default=None,
        description='session UUID for grouping related episodes',
    )
    project_name: str | None = Field(
        default=None,
        description='project name (lowercased) for grouping related episodes',
    )

    async def save(self, driver: GraphDriver):
        if driver.graph_operations_interface:
            return await driver.graph_operations_interface.episodic_node_save(self, driver)

        episode_args = {
            'uuid': self.uuid,
            'name': self.name,
            'group_id': self.group_id,
            'source_description': self.source_description,
            'content': self.content,
            'entity_edges': self.entity_edges,
            'created_at': self.created_at,
            'valid_at': self.valid_at,
            'source': self.source.value,
            'session_id': self.session_id,
            'project_name': self.project_name,
        }

        result = await driver.execute_query(
            get_episode_node_save_query(driver.provider), **episode_args
        )

        logger.debug(f'Saved Node to Graph: {self.uuid}')

        return result

    @classmethod
    async def get_by_uuid(cls, driver: GraphDriver, uuid: str):
        records, _, _ = await driver.execute_query(
            """
            MATCH (e:Episodic {uuid: $uuid})
            RETURN
            """
            + (
                EPISODIC_NODE_RETURN_NEPTUNE
                if driver.provider == GraphProvider.NEPTUNE
                else EPISODIC_NODE_RETURN
            ),
            uuid=uuid,
            routing_='r',
        )

        episodes = [get_episodic_node_from_record(record) for record in records]

        if len(episodes) == 0:
            raise NodeNotFoundError(uuid)

        return episodes[0]

    @classmethod
    async def get_by_uuids(cls, driver: GraphDriver, uuids: list[str]):
        records, _, _ = await driver.execute_query(
            """
            MATCH (e:Episodic)
            WHERE e.uuid IN $uuids
            RETURN DISTINCT
            """
            + (
                EPISODIC_NODE_RETURN_NEPTUNE
                if driver.provider == GraphProvider.NEPTUNE
                else EPISODIC_NODE_RETURN
            ),
            uuids=uuids,
            routing_='r',
        )

        episodes = [get_episodic_node_from_record(record) for record in records]

        return episodes

    @classmethod
    async def get_by_group_ids(
        cls,
        driver: GraphDriver,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ):
        cursor_query: LiteralString = 'AND e.uuid < $uuid' if uuid_cursor else ''
        limit_query: LiteralString = 'LIMIT $limit' if limit is not None else ''

        records, _, _ = await driver.execute_query(
            """
            MATCH (e:Episodic)
            WHERE e.group_id IN $group_ids
            """
            + cursor_query
            + """
            RETURN DISTINCT
            """
            + (
                EPISODIC_NODE_RETURN_NEPTUNE
                if driver.provider == GraphProvider.NEPTUNE
                else EPISODIC_NODE_RETURN
            )
            + """
            ORDER BY uuid DESC
            """
            + limit_query,
            group_ids=group_ids,
            uuid=uuid_cursor,
            limit=limit,
            routing_='r',
        )

        episodes = [get_episodic_node_from_record(record) for record in records]

        return episodes

    @classmethod
    async def get_by_entity_node_uuid(cls, driver: GraphDriver, entity_node_uuid: str):
        records, _, _ = await driver.execute_query(
            """
            MATCH (e:Episodic)-[r:MENTIONS]->(n:Entity {uuid: $entity_node_uuid})
            RETURN DISTINCT
            """
            + (
                EPISODIC_NODE_RETURN_NEPTUNE
                if driver.provider == GraphProvider.NEPTUNE
                else EPISODIC_NODE_RETURN
            ),
            entity_node_uuid=entity_node_uuid,
            routing_='r',
        )

        episodes = [get_episodic_node_from_record(record) for record in records]

        return episodes


class EntityNode(Node):
    name_embedding: list[float] | None = Field(default=None, description='embedding of the name')
    summary: str = Field(description='regional summary of surrounding edges', default_factory=str)
    attributes: dict[str, Any] = Field(
        default={}, description='Additional attributes of the node. Dependent on node labels'
    )

    async def generate_name_embedding(self, embedder: EmbedderClient):
        start = time()
        text = self.name.replace('\n', ' ')
        self.name_embedding = await embedder.create(input_data=[text])
        end = time()
        logger.debug(f'embedded {text} in {end - start} ms')

        return self.name_embedding

    async def load_name_embedding(self, driver: GraphDriver):
        if driver.graph_operations_interface:
            return await driver.graph_operations_interface.node_load_embeddings(self, driver)

        if driver.provider == GraphProvider.NEPTUNE:
            query: LiteralString = """
                MATCH (n:Entity {uuid: $uuid})
                RETURN [x IN split(n.name_embedding, ",") | toFloat(x)] as name_embedding
            """

        else:
            query: LiteralString = """
                MATCH (n:Entity {uuid: $uuid})
                RETURN n.name_embedding AS name_embedding
            """
        records, _, _ = await driver.execute_query(
            query,
            uuid=self.uuid,
            routing_='r',
        )

        if len(records) == 0:
            raise NodeNotFoundError(self.uuid)

        self.name_embedding = records[0]['name_embedding']

    async def save(self, driver: GraphDriver):
        if driver.graph_operations_interface:
            return await driver.graph_operations_interface.node_save(self, driver)

        entity_data: dict[str, Any] = {
            'uuid': self.uuid,
            'name': self.name,
            'name_embedding': self.name_embedding,
            'group_id': self.group_id,
            'summary': self.summary,
            'created_at': self.created_at,
        }

        if driver.provider == GraphProvider.KUZU:
            entity_data['attributes'] = json.dumps(self.attributes)
            entity_data['labels'] = list(set(self.labels + ['Entity']))
            result = await driver.execute_query(
                get_entity_node_save_query(driver.provider, labels=''),
                **entity_data,
            )
        else:
            entity_data.update(self.attributes or {})
            labels = ':'.join(self.labels + ['Entity'])

            result = await driver.execute_query(
                get_entity_node_save_query(driver.provider, labels),
                entity_data=entity_data,
            )

        logger.debug(f'Saved Node to Graph: {self.uuid}')

        return result

    @classmethod
    async def get_by_uuid(cls, driver: GraphDriver, uuid: str):
        records, _, _ = await driver.execute_query(
            """
            MATCH (n:Entity {uuid: $uuid})
            RETURN
            """
            + get_entity_node_return_query(driver.provider),
            uuid=uuid,
            routing_='r',
        )

        nodes = [get_entity_node_from_record(record, driver.provider) for record in records]

        if len(nodes) == 0:
            raise NodeNotFoundError(uuid)

        return nodes[0]

    @classmethod
    async def get_by_uuids(cls, driver: GraphDriver, uuids: list[str]):
        records, _, _ = await driver.execute_query(
            """
            MATCH (n:Entity)
            WHERE n.uuid IN $uuids
            RETURN
            """
            + get_entity_node_return_query(driver.provider),
            uuids=uuids,
            routing_='r',
        )

        nodes = [get_entity_node_from_record(record, driver.provider) for record in records]

        return nodes

    @classmethod
    async def count_by_group_ids(
        cls,
        driver: GraphDriver,
        group_ids: list[str],
        name_filter: str | None = None,
        label_filter: str | None = None,
        created_after: str | None = None,
        created_before: str | None = None,
    ) -> int:
        """Count entities matching the given filters."""
        # Build WHERE clauses (same logic as get_by_group_ids)
        where_clauses = ['n.group_id IN $group_ids']
        query_params: dict[str, Any] = {
            'group_ids': group_ids,
        }

        # Add filters
        if name_filter:
            where_clauses.append('toLower(n.name) CONTAINS toLower($name_filter)')
            query_params['name_filter'] = name_filter

        if label_filter:
            where_clauses.append('$label_filter IN labels(n)')
            query_params['label_filter'] = label_filter

        if created_after:
            where_clauses.append('n.created_at >= $created_after')
            query_params['created_after'] = created_after

        if created_before:
            where_clauses.append('n.created_at <= $created_before')
            query_params['created_before'] = created_before

        where_query = ' AND '.join(where_clauses)

        query = f"""
            MATCH (n:Entity)
            WHERE {where_query}
            RETURN count(n) AS total
        """

        records, _, _ = await driver.execute_query(
            query,
            **query_params,
            routing_='r',
        )

        return records[0]['total'] if records else 0

    @classmethod
    async def get_by_group_ids(
        cls,
        driver: GraphDriver,
        group_ids: list[str],
        limit: int | None = None,
        offset: int = 0,
        with_embeddings: bool = False,
        sort_by: str = 'uuid',
        sort_order: str = 'desc',
        name_filter: str | None = None,
        label_filter: str | None = None,
        created_after: str | None = None,
        created_before: str | None = None,
    ):
        # Build WHERE clauses
        where_clauses = ['n.group_id IN $group_ids']
        query_params: dict[str, Any] = {
            'group_ids': group_ids,
        }

        # NOTE: Using offset-based pagination due to FalkorDB bug
        # FalkorDB's WHERE clause is broken when combined with ORDER BY
        # No cursor logic needed - we use SKIP instead

        # Add filters
        if name_filter:
            where_clauses.append('toLower(n.name) CONTAINS toLower($name_filter)')
            query_params['name_filter'] = name_filter

        if label_filter:
            where_clauses.append('$label_filter IN labels(n)')
            query_params['label_filter'] = label_filter

        if created_after:
            where_clauses.append('n.created_at >= $created_after')
            query_params['created_after'] = created_after

        if created_before:
            where_clauses.append('n.created_at <= $created_before')
            query_params['created_before'] = created_before

        where_query = ' AND '.join(where_clauses)

        # Build ORDER BY clause
        sort_field_map = {
            'uuid': 'n.uuid',
            'name': 'n.name',
            'created_at': 'n.created_at',
        }
        sort_field = sort_field_map.get(sort_by, 'n.uuid')
        sort_direction = sort_order.upper()

        # Use normal ordering with offset pagination
        order_query = f'ORDER BY {sort_field} {sort_direction}, n.uuid {sort_direction}'

        # Build SKIP and LIMIT
        skip_query: LiteralString = f'SKIP {offset}' if offset > 0 else ''
        limit_query: LiteralString = 'LIMIT $limit' if limit is not None else ''
        if limit is not None:
            query_params['limit'] = limit

        with_embeddings_query: LiteralString = (
            """,
            n.name_embedding AS name_embedding
            """
            if with_embeddings
            else ''
        )

        query = (
            f"""
            MATCH (n:Entity)
            WHERE {where_query}
            RETURN
            """
            + get_entity_node_return_query(driver.provider)
            + with_embeddings_query
            + f"""
            {order_query}
            {skip_query}
            """
            + limit_query
        )

        import logging

        logger = logging.getLogger(__name__)
        logger.debug(f'Cypher query: {query}')
        logger.debug(f'Query params: {query_params}')

        records, _, _ = await driver.execute_query(
            query,
            **query_params,
            routing_='r',
        )

        nodes = [get_entity_node_from_record(record, driver.provider) for record in records]

        return nodes


class SessionNode(Node):
    """Node representing a conversation session with aggregated metadata."""

    session_id: str = Field(description='UUID of the session')
    summary: str = Field(description='Rolling summary of session content', default='')
    intents: list[str] = Field(
        description='List of user intents/topics in chronological order', default_factory=list
    )
    episode_count: int = Field(description='Number of episodes in session', default=0)
    first_episode_date: datetime = Field(description='Timestamp of first episode')
    last_episode_date: datetime = Field(description='Timestamp of most recent episode')
    source_descriptions: list[str] = Field(
        description='Unique source descriptions', default_factory=list
    )
    programmatic: bool = Field(
        default=False,
        description='Whether this session is programmatically-generated (automated, imported, background) or human-interactive',
    )

    async def save(self, driver: GraphDriver):
        session_data: dict[str, Any] = {
            'uuid': self.uuid,
            'name': self.name,
            'session_id': self.session_id,
            'group_id': self.group_id,
            'summary': self.summary,
            'intents': self.intents,
            'episode_count': self.episode_count,
            'first_episode_date': self.first_episode_date,
            'last_episode_date': self.last_episode_date,
            'source_descriptions': self.source_descriptions,
            'created_at': self.created_at,
            'programmatic': self.programmatic,
        }

        result = await driver.execute_query(
            get_session_node_save_query(driver.provider),
            **session_data,
        )

        logger.debug(f'Saved SessionNode to Graph: {self.uuid} (session_id: {self.session_id})')

        return result

    @classmethod
    async def get_by_session_id(cls, driver: GraphDriver, group_id: str, session_id: str):
        """Retrieve a session node by session_id and group_id."""
        records, _, _ = await driver.execute_query(
            """
            MATCH (s:Session {session_id: $session_id, group_id: $group_id})
            RETURN
            """
            + (
                SESSION_NODE_RETURN_NEPTUNE
                if driver.provider == GraphProvider.NEPTUNE
                else SESSION_NODE_RETURN
            ),
            session_id=session_id,
            group_id=group_id,
            routing_='r',
        )

        if len(records) == 0:
            return None

        return get_session_node_from_record(records[0])

    @classmethod
    async def get_by_uuid(cls, driver: GraphDriver, uuid: str):
        """Retrieve a session node by UUID."""
        records, _, _ = await driver.execute_query(
            """
            MATCH (s:Session {uuid: $uuid})
            RETURN
            """
            + (
                SESSION_NODE_RETURN_NEPTUNE
                if driver.provider == GraphProvider.NEPTUNE
                else SESSION_NODE_RETURN
            ),
            uuid=uuid,
            routing_='r',
        )

        if len(records) == 0:
            raise NodeNotFoundError(uuid)

        return get_session_node_from_record(records[0])


class ProjectNode(Node):
    """Minimal node representing a project. Metadata computed via queries."""

    name: str = Field(description='Project name (lowercased)')
    project_path: str | None = Field(default=None, description='File system path of the project')

    async def save(self, driver: GraphDriver):
        project_data: dict[str, Any] = {
            'uuid': self.uuid,
            'name': self.name,
            'group_id': self.group_id,
            'created_at': self.created_at,
            'project_path': self.project_path,
        }

        result = await driver.execute_query(
            get_project_node_save_query(driver.provider),
            **project_data,
        )

        logger.debug(f'Saved ProjectNode to Graph: {self.uuid} (name: {self.name})')

        return result

    @classmethod
    async def get_by_name(cls, driver: GraphDriver, group_id: str, name: str):
        """Retrieve a project node by name and group_id."""
        records, _, _ = await driver.execute_query(
            """
            MATCH (p:Project {name: $name, group_id: $group_id})
            RETURN p.uuid AS uuid, p.name AS name, p.group_id AS group_id,
                   p.created_at AS created_at, p.project_path AS project_path
            """,
            name=name,
            group_id=group_id,
            routing_='r',
        )

        if len(records) == 0:
            return None

        return ProjectNode(
            uuid=records[0]['uuid'],
            name=records[0]['name'],
            group_id=records[0]['group_id'],
            created_at=parse_db_date(records[0]['created_at']),
            project_path=records[0].get('project_path'),
        )

    @classmethod
    async def get_by_uuid(cls, driver: GraphDriver, uuid: str):
        """Retrieve a project node by UUID."""
        records, _, _ = await driver.execute_query(
            """
            MATCH (p:Project {uuid: $uuid})
            RETURN p.uuid AS uuid, p.name AS name, p.group_id AS group_id,
                   p.created_at AS created_at, p.project_path AS project_path
            """,
            uuid=uuid,
            routing_='r',
        )

        if len(records) == 0:
            raise NodeNotFoundError(uuid)

        return ProjectNode(
            uuid=records[0]['uuid'],
            name=records[0]['name'],
            group_id=records[0]['group_id'],
            created_at=parse_db_date(records[0]['created_at']),
            project_path=records[0].get('project_path'),
        )


class CommunityNode(Node):
    name_embedding: list[float] | None = Field(default=None, description='embedding of the name')
    summary: str = Field(description='region summary of member nodes', default_factory=str)

    async def save(self, driver: GraphDriver):
        if driver.provider == GraphProvider.NEPTUNE:
            await driver.save_to_aoss(  # pyright: ignore reportAttributeAccessIssue
                'communities',
                [{'name': self.name, 'uuid': self.uuid, 'group_id': self.group_id}],
            )
        result = await driver.execute_query(
            get_community_node_save_query(driver.provider),  # type: ignore
            uuid=self.uuid,
            name=self.name,
            group_id=self.group_id,
            summary=self.summary,
            name_embedding=self.name_embedding,
            created_at=self.created_at,
        )

        logger.debug(f'Saved Node to Graph: {self.uuid}')

        return result

    async def generate_name_embedding(self, embedder: EmbedderClient):
        start = time()
        text = self.name.replace('\n', ' ')
        self.name_embedding = await embedder.create(input_data=[text])
        end = time()
        logger.debug(f'embedded {text} in {end - start} ms')

        return self.name_embedding

    async def load_name_embedding(self, driver: GraphDriver):
        if driver.provider == GraphProvider.NEPTUNE:
            query: LiteralString = """
                MATCH (c:Community {uuid: $uuid})
                RETURN [x IN split(c.name_embedding, ",") | toFloat(x)] as name_embedding
            """
        else:
            query: LiteralString = """
            MATCH (c:Community {uuid: $uuid})
            RETURN c.name_embedding AS name_embedding
            """

        records, _, _ = await driver.execute_query(
            query,
            uuid=self.uuid,
            routing_='r',
        )

        if len(records) == 0:
            raise NodeNotFoundError(self.uuid)

        self.name_embedding = records[0]['name_embedding']

    @classmethod
    async def get_by_uuid(cls, driver: GraphDriver, uuid: str):
        records, _, _ = await driver.execute_query(
            """
            MATCH (c:Community {uuid: $uuid})
            RETURN
            """
            + (
                COMMUNITY_NODE_RETURN_NEPTUNE
                if driver.provider == GraphProvider.NEPTUNE
                else COMMUNITY_NODE_RETURN
            ),
            uuid=uuid,
            routing_='r',
        )

        nodes = [get_community_node_from_record(record) for record in records]

        if len(nodes) == 0:
            raise NodeNotFoundError(uuid)

        return nodes[0]

    @classmethod
    async def get_by_uuids(cls, driver: GraphDriver, uuids: list[str]):
        records, _, _ = await driver.execute_query(
            """
            MATCH (c:Community)
            WHERE c.uuid IN $uuids
            RETURN
            """
            + (
                COMMUNITY_NODE_RETURN_NEPTUNE
                if driver.provider == GraphProvider.NEPTUNE
                else COMMUNITY_NODE_RETURN
            ),
            uuids=uuids,
            routing_='r',
        )

        communities = [get_community_node_from_record(record) for record in records]

        return communities

    @classmethod
    async def get_by_group_ids(
        cls,
        driver: GraphDriver,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ):
        cursor_query: LiteralString = 'AND c.uuid < $uuid' if uuid_cursor else ''
        limit_query: LiteralString = 'LIMIT $limit' if limit is not None else ''

        records, _, _ = await driver.execute_query(
            """
            MATCH (c:Community)
            WHERE c.group_id IN $group_ids
            """
            + cursor_query
            + """
            RETURN
            """
            + (
                COMMUNITY_NODE_RETURN_NEPTUNE
                if driver.provider == GraphProvider.NEPTUNE
                else COMMUNITY_NODE_RETURN
            )
            + """
            ORDER BY c.uuid DESC
            """
            + limit_query,
            group_ids=group_ids,
            uuid=uuid_cursor,
            limit=limit,
            routing_='r',
        )

        communities = [get_community_node_from_record(record) for record in records]

        return communities


# Node helpers
def get_episodic_node_from_record(record: Any) -> EpisodicNode:
    created_at = parse_db_date(record['created_at'])
    valid_at = parse_db_date(record['valid_at'])

    if created_at is None:
        raise ValueError(f'created_at cannot be None for episode {record.get("uuid", "unknown")}')
    if valid_at is None:
        raise ValueError(f'valid_at cannot be None for episode {record.get("uuid", "unknown")}')

    return EpisodicNode(
        content=record['content'],
        created_at=created_at,
        valid_at=valid_at,
        uuid=record['uuid'],
        group_id=record['group_id'],
        source=EpisodeType.from_str(record['source']),
        name=record['name'],
        source_description=record['source_description'],
        entity_edges=record['entity_edges'],
        session_id=record.get('session_id'),
    )


def get_entity_node_from_record(record: Any, provider: GraphProvider) -> EntityNode:
    if provider == GraphProvider.KUZU:
        attributes = json.loads(record['attributes']) if record['attributes'] else {}
    else:
        attributes = record['attributes']
        attributes.pop('uuid', None)
        attributes.pop('name', None)
        attributes.pop('group_id', None)
        attributes.pop('name_embedding', None)
        attributes.pop('summary', None)
        attributes.pop('created_at', None)
        attributes.pop('labels', None)

    labels = record.get('labels', [])
    group_id = record.get('group_id')
    if 'Entity_' + group_id.replace('-', '') in labels:
        labels.remove('Entity_' + group_id.replace('-', ''))

    entity_node = EntityNode(
        uuid=record['uuid'],
        name=record['name'],
        name_embedding=record.get('name_embedding'),
        group_id=group_id,
        labels=labels,
        created_at=parse_db_date(record['created_at']),  # type: ignore
        summary=record['summary'],
        attributes=attributes,
    )

    return entity_node


def get_community_node_from_record(record: Any) -> CommunityNode:
    return CommunityNode(
        uuid=record['uuid'],
        name=record['name'],
        group_id=record['group_id'],
        name_embedding=record['name_embedding'],
        created_at=parse_db_date(record['created_at']),  # type: ignore
        summary=record['summary'],
    )


def get_session_node_from_record(record: Any) -> SessionNode:
    """Parse a SessionNode from a database record."""
    created_at = parse_db_date(record['created_at'])
    first_episode_date = parse_db_date(record['first_episode_date'])
    last_episode_date = parse_db_date(record['last_episode_date'])

    if created_at is None:
        raise ValueError(f'created_at cannot be None for session {record.get("uuid", "unknown")}')
    if first_episode_date is None:
        raise ValueError(
            f'first_episode_date cannot be None for session {record.get("uuid", "unknown")}'
        )
    if last_episode_date is None:
        raise ValueError(
            f'last_episode_date cannot be None for session {record.get("uuid", "unknown")}'
        )

    source_descriptions = record.get('source_descriptions', [])
    # Handle case where source_descriptions might be a string (from database)
    if isinstance(source_descriptions, str):
        source_descriptions = [source_descriptions] if source_descriptions else []

    intents = record.get('intents', [])
    # Handle case where intents might be missing (backwards compatibility) or None
    if intents is None:
        intents = []

    return SessionNode(
        uuid=record['uuid'],
        name=record['name'],
        session_id=record['session_id'],
        group_id=record['group_id'],
        created_at=created_at,
        summary=record.get('summary', ''),
        intents=intents,
        episode_count=record.get('episode_count', 0),
        first_episode_date=first_episode_date,
        last_episode_date=last_episode_date,
        source_descriptions=source_descriptions,
        programmatic=record.get('programmatic', False),
    )


async def create_entity_node_embeddings(embedder: EmbedderClient, nodes: list[EntityNode]):
    # filter out falsey values from nodes
    filtered_nodes = [node for node in nodes if node.name]

    if not filtered_nodes:
        return

    name_embeddings = await embedder.create_batch([node.name for node in filtered_nodes])
    for node, name_embedding in zip(filtered_nodes, name_embeddings, strict=True):
        node.name_embedding = name_embedding
