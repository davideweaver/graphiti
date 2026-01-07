import base64
import json
import logging
from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status

logger = logging.getLogger(__name__)
from graphiti_core.errors import NodeNotFoundError  # type: ignore
from graphiti_core.nodes import EntityNode  # type: ignore
from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_RRF  # type: ignore
from graphiti_core.search.search_filters import (  # type: ignore
    ComparisonOperator,
    DateFilter,
    SearchFilters,
)

from graph_service.dto import (
    DaySessionCount,
    EntityListResponse,
    GetMemoryRequest,
    GetMemoryResponse,
    Message,
    SearchQuery,
    SearchResults,
    SessionListResponse,
    SessionResponse,
    SessionStatsByDayResponse,
)
from graph_service.zep_graphiti import (
    ZepGraphiti,
    get_entity_node_response,
    get_fact_result_from_edge,
    get_graphiti_from_body,
    get_graphiti_from_path,
    get_graphiti_from_query,
)

router = APIRouter()


def extract_preview(content: str | None, max_chars: int = 200) -> str | None:
    """Extract first couple sentences or max_chars from content."""
    if not content:
        return None

    # Truncate to max_chars
    preview = content[:max_chars].strip()

    # Try to cut at sentence boundary if we truncated
    if len(content) > max_chars:
        # Find last sentence ending within the preview
        for ending in ['. ', '! ', '? ']:
            last_sentence = preview.rfind(ending)
            if last_sentence > 0:
                preview = preview[:last_sentence + 1]
                break
        else:
            # No sentence ending found, add ellipsis
            preview = preview.rstrip() + '...'

    return preview


@router.post('/search', status_code=status.HTTP_200_OK)
async def search(
    query: SearchQuery,
    graphiti: Annotated[ZepGraphiti, Depends(get_graphiti_from_body)],
):
    # Build search filters for date range
    search_filter = SearchFilters()
    if query.start_date or query.end_date:
        date_filters = []
        if query.start_date:
            date_filters.append(
                DateFilter(
                    date=query.start_date, comparison_operator=ComparisonOperator.greater_than_equal
                )
            )
        if query.end_date:
            date_filters.append(
                DateFilter(
                    date=query.end_date, comparison_operator=ComparisonOperator.less_than_equal
                )
            )
        search_filter.valid_at = [date_filters]

    # Use search_() for full SearchResults with similarity scores
    # (search() only returns edges, discarding scores)
    search_results = await graphiti.search_(
        query=query.query,
        config=EDGE_HYBRID_SEARCH_RRF,  # Hybrid search with RRF reranking
        group_ids=[query.group_id],
        search_filter=search_filter,
    )

    # Extract edges and their corresponding similarity scores
    edges = search_results.edges
    scores = search_results.edge_reranker_scores

    # Zip edges with scores (handle case where scores list might be empty or shorter)
    facts = []
    for idx, edge in enumerate(edges):
        score = scores[idx] if idx < len(scores) else None
        facts.append(get_fact_result_from_edge(edge, score))

    return SearchResults(
        facts=facts,
    )


@router.get('/entity-edge/{uuid}/related-facts', status_code=status.HTTP_200_OK)
async def get_related_facts(
    uuid: str,
    group_id: str,
    graphiti: Annotated[ZepGraphiti, Depends(get_graphiti_from_query)],
):
    """Get fact provenance and lifecycle information.

    Returns:
    - superseded_by: Facts that invalidated/replaced this fact (created around when this was invalidated)
    - supersedes: Facts that this fact invalidated/replaced (invalidated around when this was created)
    - related: Other facts involving the same entities for context
    """
    from datetime import timedelta

    # First get the fact to know what entities and timestamps we're working with
    entity_edge = await graphiti.get_entity_edge(uuid)

    from graphiti_core.helpers import parse_db_date
    created_at = entity_edge.created_at
    invalid_at = entity_edge.invalid_at

    # Calculate time windows in Python (FalkorDB doesn't support datetime() function)
    time_window = timedelta(hours=1)

    # Find superseding facts (created around when this fact was invalidated)
    # Look within 1 hour window
    superseded_by_query = None
    superseded_by_params = {}
    if invalid_at:
        invalid_at_before = (invalid_at - time_window).isoformat()
        invalid_at_after = (invalid_at + time_window).isoformat()
        superseded_by_query = """
            MATCH ()-[r:RELATES_TO]->()
            WHERE r.group_id = $group_id
              AND r.uuid <> $edge_uuid
              AND r.name = $rel_name
              AND (r.source_node_uuid = $source_uuid OR r.target_node_uuid = $source_uuid)
              AND r.created_at >= $created_at_min
              AND r.created_at <= $created_at_max
            RETURN r.uuid AS uuid, r.name AS name, r.fact AS fact,
                   r.valid_at AS valid_at, r.invalid_at AS invalid_at,
                   r.created_at AS created_at, r.expired_at AS expired_at
            ORDER BY r.created_at ASC
            LIMIT 10
        """
        superseded_by_params = {
            'created_at_min': invalid_at_before,
            'created_at_max': invalid_at_after,
        }

    # Find superseded facts (invalidated around when this fact was created)
    created_at_before = (created_at - time_window).isoformat()
    created_at_after = (created_at + time_window).isoformat()
    supersedes_query = """
        MATCH ()-[r:RELATES_TO]->()
        WHERE r.group_id = $group_id
          AND r.uuid <> $edge_uuid
          AND r.name = $rel_name
          AND (r.source_node_uuid = $source_uuid OR r.target_node_uuid = $source_uuid)
          AND r.invalid_at IS NOT NULL
          AND r.invalid_at >= $invalid_at_min
          AND r.invalid_at <= $invalid_at_max
        RETURN r.uuid AS uuid, r.name AS name, r.fact AS fact,
               r.valid_at AS valid_at, r.invalid_at AS invalid_at,
               r.created_at AS created_at, r.expired_at AS expired_at
        ORDER BY r.invalid_at DESC
        LIMIT 10
    """
    supersedes_params = {
        'invalid_at_min': created_at_before,
        'invalid_at_max': created_at_after,
    }

    # Find other related facts for context
    related_query = """
        MATCH ()-[r:RELATES_TO]->()
        WHERE r.group_id = $group_id
          AND r.uuid <> $edge_uuid
          AND (r.source_node_uuid = $source_uuid
               OR r.target_node_uuid = $source_uuid
               OR r.source_node_uuid = $target_uuid
               OR r.target_node_uuid = $target_uuid)
        RETURN DISTINCT r.uuid AS uuid, r.name AS name, r.fact AS fact,
               r.valid_at AS valid_at, r.invalid_at AS invalid_at,
               r.created_at AS created_at, r.expired_at AS expired_at
        ORDER BY r.created_at DESC
        LIMIT 20
    """

    try:
        from graphiti_core.helpers import parse_db_date

        def records_to_facts(records):
            """Convert query records to FactResult objects"""
            facts = []
            for record in records:
                facts.append(
                    get_fact_result_from_edge(
                        type('Edge', (), {
                            'uuid': record['uuid'],
                            'name': record['name'],
                            'fact': record['fact'],
                            'valid_at': parse_db_date(record['valid_at']) if record.get('valid_at') else None,
                            'invalid_at': parse_db_date(record['invalid_at']) if record.get('invalid_at') else None,
                            'created_at': parse_db_date(record['created_at']) if record.get('created_at') else None,
                            'expired_at': parse_db_date(record['expired_at']) if record.get('expired_at') else None,
                            'source_node_uuid': '',
                            'target_node_uuid': '',
                        })()
                    )
                )
            return facts

        # Execute superseded_by query (facts that replaced this one)
        superseded_by_facts = []
        if superseded_by_query and invalid_at:
            records, _, _ = await graphiti.driver.execute_query(
                superseded_by_query,
                edge_uuid=uuid,
                group_id=group_id,
                source_uuid=entity_edge.source_node_uuid,
                rel_name=entity_edge.name,
                **superseded_by_params,
            )
            superseded_by_facts = records_to_facts(records)

        # Execute supersedes query (facts that this one replaced)
        supersedes_facts = []
        records, _, _ = await graphiti.driver.execute_query(
            supersedes_query,
            edge_uuid=uuid,
            group_id=group_id,
            source_uuid=entity_edge.source_node_uuid,
            rel_name=entity_edge.name,
            **supersedes_params,
        )
        supersedes_facts = records_to_facts(records)

        # Execute related query (other facts involving same entities)
        related_facts = []
        records, _, _ = await graphiti.driver.execute_query(
            related_query,
            edge_uuid=uuid,
            group_id=group_id,
            source_uuid=entity_edge.source_node_uuid,
            target_uuid=entity_edge.target_node_uuid,
        )
        related_facts = records_to_facts(records)

        return {
            'superseded_by': superseded_by_facts,
            'supersedes': supersedes_facts,
            'related': related_facts,
        }

    except Exception as e:
        logger.error(f'Failed to fetch related facts for edge {uuid}: {e}')
        import traceback
        logger.error(traceback.format_exc())
        return {
            'superseded_by': [],
            'supersedes': [],
            'related': [],
        }


@router.get('/entity-edge/{uuid}', status_code=status.HTTP_200_OK)
async def get_entity_edge(
    uuid: str,
    group_id: str,
    graphiti: Annotated[ZepGraphiti, Depends(get_graphiti_from_query)],
    include_entities: bool = Query(False, description='Include source and target entity details'),
    include_episodes: bool = Query(False, description='Include related episodes'),
    include_related: bool = Query(False, description='Include related facts'),
):
    """Get entity edge (fact) details with optional provenance information.

    Args:
        uuid: The UUID of the entity edge
        group_id: The group ID
        include_entities: If true, includes full details of source and target entities
        include_episodes: If true, includes episodes that created/mentioned this fact

    Returns:
        FactResult with optional provenance data
    """
    entity_edge = await graphiti.get_entity_edge(uuid)

    source_entity = None
    target_entity = None
    episodes = None

    # Fetch source and target entities if requested
    if include_entities:
        try:
            source_node = await EntityNode.get_by_uuid(graphiti.driver, entity_edge.source_node_uuid)
            source_entity = get_entity_node_response(source_node)
        except NodeNotFoundError:
            logger.warning(f'Source entity not found: {entity_edge.source_node_uuid}')

        try:
            target_node = await EntityNode.get_by_uuid(graphiti.driver, entity_edge.target_node_uuid)
            target_entity = get_entity_node_response(target_node)
        except NodeNotFoundError:
            logger.warning(f'Target entity not found: {entity_edge.target_node_uuid}')

    # Fetch related episodes if requested
    if include_episodes:
        from graph_service.dto.retrieve import EpisodeResponse
        from graphiti_core.helpers import parse_db_date

        # Query episodes that mention either the source or target entity
        # Note: This is a heuristic - we find episodes that mention the entities in this relationship
        episodes_query = """
            MATCH (edge:Relation {uuid: $edge_uuid})
            MATCH (ep:Episodic)-[:MENTIONS]->(entity:Entity)
            WHERE entity.uuid IN [$source_uuid, $target_uuid]
            RETURN DISTINCT ep.uuid AS uuid, ep.name AS name, ep.content AS content,
                   ep.source_description AS source_description, ep.session_id AS session_id,
                   ep.timestamp AS timestamp, ep.valid_at AS valid_at,
                   ep.created_at AS created_at, ep.group_id AS group_id
            ORDER BY ep.valid_at DESC
            LIMIT 10
        """

        try:
            records, _, _ = await graphiti.driver.execute_query(
                episodes_query,
                edge_uuid=uuid,
                source_uuid=entity_edge.source_node_uuid,
                target_uuid=entity_edge.target_node_uuid,
            )

            episodes = [
                EpisodeResponse(
                    uuid=record['uuid'],
                    name=record['name'],
                    content=record['content'],
                    source_description=record['source_description'],
                    session_id=record['session_id'],
                    timestamp=parse_db_date(record['timestamp']),
                    valid_at=parse_db_date(record['valid_at']),
                    created_at=parse_db_date(record['created_at']),
                    group_id=record['group_id'],
                )
                for record in records
            ]
        except Exception as e:
            logger.error(f'Failed to fetch episodes for edge {uuid}: {e}')
            episodes = []

    return get_fact_result_from_edge(
        entity_edge,
        source_entity=source_entity,
        target_entity=target_entity,
        episodes=episodes,
    )


@router.get('/episodes/{group_id}', status_code=status.HTTP_200_OK)
async def get_episodes(
    group_id: str,
    graphiti: Annotated[ZepGraphiti, Depends(get_graphiti_from_path)],
    last_n: int = Query(..., description='Number of most recent episodes to retrieve'),
    start_date: datetime | None = Query(
        None, description='Filter episodes with valid_at >= this datetime (ISO 8601)'
    ),
    end_date: datetime | None = Query(
        None, description='Filter episodes with valid_at <= this datetime (ISO 8601)'
    ),
    session_id: str | None = Query(
        None, description='Filter episodes by session UUID from source_description'
    ),
):
    episodes = await graphiti.retrieve_episodes(
        group_ids=[group_id],
        last_n=last_n,
        reference_time=datetime.now(timezone.utc),
        start_date=start_date,
        end_date=end_date,
        session_id=session_id,
    )
    return episodes


@router.post('/get-memory', status_code=status.HTTP_200_OK)
async def get_memory(
    request: GetMemoryRequest,
    graphiti: Annotated[ZepGraphiti, Depends(get_graphiti_from_body)],
):
    combined_query = compose_query_from_messages(request.messages)
    search_results = await graphiti.search_(
        query=combined_query,
        config=EDGE_HYBRID_SEARCH_RRF,  # Hybrid search with RRF reranking
        group_ids=[request.group_id],
    )

    # Extract edges and scores
    edges = search_results.edges
    scores = search_results.edge_reranker_scores

    # Convert to facts with scores
    facts = []
    for idx, edge in enumerate(edges):
        score = scores[idx] if idx < len(scores) else None
        facts.append(get_fact_result_from_edge(edge, score))

    return GetMemoryResponse(facts=facts)


def compose_query_from_messages(messages: list[Message]):
    combined_query = ''
    for message in messages:
        combined_query += f'{message.role_type or ""}({message.role or ""}): {message.content}\n'
    return combined_query


@router.get('/entities/{group_id}/{uuid}', status_code=status.HTTP_200_OK)
async def get_entity(
    group_id: str,
    uuid: str,
    graphiti: Annotated[ZepGraphiti, Depends(get_graphiti_from_path)],
):
    """Get a single entity node by UUID."""
    try:
        entity = await EntityNode.get_by_uuid(graphiti.driver, uuid)
        return get_entity_node_response(entity)
    except NodeNotFoundError:
        raise HTTPException(status_code=404, detail=f'Entity not found: {uuid}')


@router.get('/entities/{group_id}', status_code=status.HTTP_200_OK)
async def list_entities(
    group_id: str,
    limit: int = Query(50, ge=1, le=500, description='Maximum number of entities to return'),
    cursor: str | None = Query(
        None, description='Pagination cursor (base64-encoded composite cursor)'
    ),
    with_embeddings: bool = Query(False, description='Include name embeddings in response'),
    sort_by: str = Query(
        'uuid',
        pattern='^(uuid|name|created_at)$',
        description='Field to sort by: uuid, name, or created_at',
    ),
    sort_order: str = Query('desc', pattern='^(asc|desc)$', description='Sort order: asc or desc'),
    name_filter: str | None = Query(
        None, description='Filter entities by name (case-insensitive substring match)'
    ),
    label: str | None = Query(None, description='Filter entities by label/type'),
    created_after: datetime | None = Query(
        None, description='Filter entities created after this datetime (ISO 8601)'
    ),
    created_before: datetime | None = Query(
        None, description='Filter entities created before this datetime (ISO 8601)'
    ),
    graphiti: Annotated[ZepGraphiti, Depends(get_graphiti_from_path)] = None,
):
    """List entities for a group with pagination, sorting, and filtering.

    Supports:
    - Sorting by uuid, name, or created_at (ascending or descending)
    - Filtering by name (substring), label (entity type), and date range
    - Cursor-based pagination with composite cursors for non-UUID sorts
    """
    # Parse offset from cursor (offset-based pagination due to FalkorDB bug)
    # FalkorDB has a bug where WHERE + ORDER BY breaks comparison operators
    offset = 0
    if cursor:
        try:
            cursor_data = json.loads(base64.b64decode(cursor).decode('utf-8'))
            offset = cursor_data.get('offset', 0)
        except (ValueError, KeyError) as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f'Invalid cursor format: {e}',
            ) from e

    # Fetch entities with sorting and filtering
    # Convert datetime objects to ISO strings for database comparison
    created_after_str = created_after.isoformat() if created_after else None
    created_before_str = created_before.isoformat() if created_before else None

    # Get total count of entities matching the filters
    total_count = await EntityNode.count_by_group_ids(
        graphiti.driver,
        group_ids=[group_id],
        name_filter=name_filter,
        label_filter=label,
        created_after=created_after_str,
        created_before=created_before_str,
    )

    entities = await EntityNode.get_by_group_ids(
        graphiti.driver,
        group_ids=[group_id],
        limit=limit,
        offset=offset,
        with_embeddings=with_embeddings,
        sort_by=sort_by,
        sort_order=sort_order,
        name_filter=name_filter,
        label_filter=label,
        created_after=created_after_str,
        created_before=created_before_str,
    )

    # Generate next cursor with new offset
    next_cursor = None
    if entities and len(entities) == limit:
        next_offset = offset + limit
        cursor_obj = {'offset': next_offset}
        next_cursor = base64.b64encode(json.dumps(cursor_obj).encode('utf-8')).decode('utf-8')

    return EntityListResponse(
        entities=[get_entity_node_response(e) for e in entities],
        total=total_count,
        has_more=len(entities) == limit,
        cursor=next_cursor,
    )


@router.post('/entities/by-uuids', status_code=status.HTTP_200_OK)
async def get_entities_by_uuids(
    uuids: list[str],
    group_id: str,
    graphiti: Annotated[ZepGraphiti, Depends(get_graphiti_from_query)],
):
    """Get multiple entities by their UUIDs (batch retrieval)."""
    if not uuids:
        return EntityListResponse(
            entities=[],
            total=0,
            has_more=False,
            cursor=None,
        )

    entities = await EntityNode.get_by_uuids(graphiti.driver, uuids)

    return EntityListResponse(
        entities=[get_entity_node_response(e) for e in entities],
        total=len(entities),
        has_more=False,
        cursor=None,
    )


@router.get('/entities/{group_id}/{uuid}/relationships', status_code=status.HTTP_200_OK)
async def get_entity_relationships(
    group_id: str,
    uuid: str,
    graphiti: Annotated[ZepGraphiti, Depends(get_graphiti_from_path)],
):
    """Get entities related to this entity via RELATES_TO relationships."""
    # Query for entities with RELATES_TO relationships (both directions)
    query = f"""
    MATCH (e:Entity {{uuid: '{uuid}'}})-[:RELATES_TO]-(related:Entity)
    RETURN DISTINCT related.uuid as uuid, related.name as name,
           related.group_id as group_id, related.summary as summary,
           labels(related) as labels, related.created_at as created_at
    LIMIT 50
    """

    try:
        # Use driver's execute_query method (works for both FalkorDB and Neo4j)
        result = await graphiti.driver.execute_query(query)

        if result is None:
            return EntityListResponse(entities=[], total=0, has_more=False, cursor=None)

        records, header, _ = result

        # Parse query results into entity objects
        related_entities = []
        for record in records:
            entity_data = {
                'uuid': record.get('uuid', ''),
                'name': record.get('name', ''),
                'group_id': record.get('group_id', ''),
                'summary': record.get('summary', ''),
                'labels': record.get('labels', ['Entity']),
                'attributes': {},
                'created_at': record.get('created_at', ''),
            }
            related_entities.append(entity_data)

        return EntityListResponse(
            entities=related_entities,
            total=len(related_entities),
            has_more=False,
            cursor=None,
        )
    except Exception as e:
        import logging

        logging.error(f'Error fetching relationships: {e}')
        # If query fails, return empty list
        return EntityListResponse(entities=[], total=0, has_more=False, cursor=None)


@router.get('/sessions/{group_id}', status_code=status.HTTP_200_OK)
async def list_sessions(
    group_id: str,
    graphiti: Annotated[ZepGraphiti, Depends(get_graphiti_from_path)],
    limit: int = Query(50, ge=1, le=500, description='Maximum number of sessions to return'),
    cursor: str | None = Query(None, description='Pagination cursor (base64-encoded offset)'),
    created_after: datetime | None = Query(None, description='Filter episodes created after (ISO 8601)'),
    created_before: datetime | None = Query(None, description='Filter episodes created before (ISO 8601)'),
    valid_after: datetime | None = Query(None, description='Filter episodes occurred after (ISO 8601)'),
    valid_before: datetime | None = Query(None, description='Filter episodes occurred before (ISO 8601)'),
    sort_order: str = Query('desc', pattern='^(asc|desc)$', description='Sort by last episode date'),
):
    """List all sessions in a group with metadata.

    Sessions are groups of related episodes identified by session_id.
    Returns metadata including episode count and date range for each session.
    """
    # Parse offset from cursor
    offset = 0
    if cursor:
        try:
            cursor_data = json.loads(base64.b64decode(cursor).decode('utf-8'))
            offset = cursor_data.get('offset', 0)
        except (ValueError, KeyError) as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f'Invalid cursor format: {e}',
            ) from e

    # Build WHERE clauses
    where_clauses = ['e.group_id = $group_id', 'e.session_id IS NOT NULL']
    query_params = {'group_id': group_id, 'limit': limit, 'offset': offset}

    # Add date filters
    if created_after:
        where_clauses.append('e.created_at >= $created_after')
        query_params['created_after'] = created_after.isoformat()
    if created_before:
        where_clauses.append('e.created_at <= $created_before')
        query_params['created_before'] = created_before.isoformat()
    if valid_after:
        where_clauses.append('e.valid_at >= $valid_after')
        query_params['valid_after'] = valid_after.isoformat()
    if valid_before:
        where_clauses.append('e.valid_at <= $valid_before')
        query_params['valid_before'] = valid_before.isoformat()

    where_query = ' AND '.join(where_clauses)
    order_direction = 'DESC' if sort_order == 'desc' else 'ASC'

    # Count total sessions (for pagination metadata)
    count_query = f"""
        MATCH (e:Episodic)
        WHERE {where_query}
        RETURN count(DISTINCT e.session_id) AS total
    """
    count_result, _, _ = await graphiti.driver.execute_query(count_query, **query_params)
    total_count = count_result[0]['total'] if count_result else 0

    # Fetch sessions with metadata, optionally joining with SessionNode for summaries and projects
    sessions_query = f"""
        MATCH (e:Episodic)
        WHERE {where_query}
        WITH e.session_id AS session_id,
             count(e) AS episode_count,
             min(e.valid_at) AS first_episode_date,
             max(e.valid_at) AS last_episode_date,
             collect(DISTINCT e.source_description) AS source_descriptions
        OPTIONAL MATCH (s:Session {{session_id: session_id, group_id: $group_id}})
        OPTIONAL MATCH (s)-[:PART_OF_PROJECT]->(p:Project)
        OPTIONAL MATCH (first_ep:Episodic {{session_id: session_id, group_id: $group_id}})
        WHERE first_ep.valid_at = first_episode_date
        WITH session_id, episode_count, first_episode_date, last_episode_date,
             source_descriptions, s.summary AS summary,
             collect(DISTINCT p.name) AS project_names,
             head(collect(first_ep.content)) AS first_episode_content
        RETURN session_id, episode_count, first_episode_date,
               last_episode_date, source_descriptions, summary,
               project_names, first_episode_content
        ORDER BY last_episode_date {order_direction}
        SKIP $offset
        LIMIT $limit
    """

    records, _, _ = await graphiti.driver.execute_query(sessions_query, **query_params)

    # Parse results into DTOs
    from graphiti_core.helpers import parse_db_date

    sessions = []
    for record in records:
        # Filter out None values from project_names
        project_names = [p for p in record.get('project_names', []) if p is not None]

        sessions.append(
            SessionResponse(
                session_id=record['session_id'],
                episode_count=record['episode_count'],
                first_episode_date=parse_db_date(record['first_episode_date']),
                last_episode_date=parse_db_date(record['last_episode_date']),
                source_descriptions=record['source_descriptions'],
                summary=record.get('summary'),
                project_name=project_names[0] if project_names else None,
                first_episode_preview=extract_preview(record.get('first_episode_content')),
            )
        )

    # Generate next cursor
    next_cursor = None
    if len(sessions) == limit:
        next_offset = offset + limit
        cursor_obj = {'offset': next_offset}
        next_cursor = base64.b64encode(json.dumps(cursor_obj).encode('utf-8')).decode('utf-8')

    return SessionListResponse(
        sessions=sessions,
        total=total_count,
        has_more=len(sessions) == limit,
        cursor=next_cursor,
    )


@router.get('/sessions/{group_id}/stats/by-day', status_code=status.HTTP_200_OK)
async def get_session_stats_by_day(
    group_id: str,
    graphiti: Annotated[ZepGraphiti, Depends(get_graphiti_from_path)],
    created_after: datetime | None = Query(None, description='Filter episodes created after (ISO 8601)'),
    created_before: datetime | None = Query(None, description='Filter episodes created before (ISO 8601)'),
    valid_after: datetime | None = Query(None, description='Filter episodes occurred after (ISO 8601)'),
    valid_before: datetime | None = Query(None, description='Filter episodes occurred before (ISO 8601)'),
):
    """Get count of unique sessions per day.

    Aggregates session activity by day, useful for visualizing session patterns over time.
    Uses episode valid_at (when episode occurred) for day grouping.
    """
    # Build WHERE clauses (same as list endpoint)
    where_clauses = ['e.group_id = $group_id', 'e.session_id IS NOT NULL']
    query_params = {'group_id': group_id}

    if created_after:
        where_clauses.append('e.created_at >= $created_after')
        query_params['created_after'] = created_after.isoformat()
    if created_before:
        where_clauses.append('e.created_at <= $created_before')
        query_params['created_before'] = created_before.isoformat()
    if valid_after:
        where_clauses.append('e.valid_at >= $valid_after')
        query_params['valid_after'] = valid_after.isoformat()
    if valid_before:
        where_clauses.append('e.valid_at <= $valid_before')
        query_params['valid_before'] = valid_before.isoformat()

    where_query = ' AND '.join(where_clauses)

    # Query: sessions per day (using valid_at)
    stats_query = f"""
        MATCH (e:Episodic)
        WHERE {where_query}
        WITH date(e.valid_at) AS day, e.session_id AS session_id
        WITH day, count(DISTINCT session_id) AS count
        RETURN toString(day) AS date, count
        ORDER BY day ASC
    """

    records, _, _ = await graphiti.driver.execute_query(stats_query, **query_params)

    # Parse results
    stats = [DaySessionCount(date=record['date'], count=record['count']) for record in records]

    return SessionStatsByDayResponse(stats=stats, total_days=len(stats))


@router.get('/sessions/{group_id}/{session_id}', status_code=status.HTTP_200_OK)
async def get_session(
    group_id: str,
    session_id: str,
    graphiti: Annotated[ZepGraphiti, Depends(get_graphiti_from_path)],
):
    """Get detailed session information including metadata and all episodes.

    Returns session metadata (summary, dates, episode count) plus full episode list.
    Returns 404 if session does not exist.
    """
    from graphiti_core.helpers import parse_db_date

    # Query session metadata from SessionNode with project information
    session_query = """
        MATCH (s:Session {session_id: $session_id, group_id: $group_id})
        OPTIONAL MATCH (s)-[:PART_OF_PROJECT]->(p:Project)
        OPTIONAL MATCH (first_ep:Episodic {session_id: $session_id, group_id: $group_id})
        WHERE first_ep.valid_at = s.first_episode_date
        WITH s, collect(DISTINCT p.name) AS project_names, head(collect(first_ep.content)) AS first_episode_content
        RETURN s.session_id AS session_id,
               s.summary AS summary,
               s.episode_count AS episode_count,
               s.first_episode_date AS first_episode_date,
               s.last_episode_date AS last_episode_date,
               s.source_descriptions AS source_descriptions,
               project_names,
               first_episode_content
    """

    records, _, _ = await graphiti.driver.execute_query(
        session_query,
        session_id=session_id,
        group_id=group_id
    )

    if not records:
        raise HTTPException(status_code=404, detail=f'Session not found: {session_id}')

    session_data = records[0]

    # Retrieve all episodes for this session
    episodes = await graphiti.retrieve_episodes(
        group_ids=[group_id],
        last_n=10000,
        reference_time=datetime.now(timezone.utc),
        session_id=session_id,
    )

    # Filter out None values from project_names
    project_names = [p for p in session_data.get('project_names', []) if p is not None]

    # Return session with metadata and episodes
    return {
        'session_id': session_data['session_id'],
        'summary': session_data.get('summary'),
        'episode_count': session_data.get('episode_count', 0),
        'first_episode_date': parse_db_date(session_data['first_episode_date']) if session_data.get('first_episode_date') else None,
        'last_episode_date': parse_db_date(session_data['last_episode_date']) if session_data.get('last_episode_date') else None,
        'source_descriptions': session_data.get('source_descriptions', []),
        'project_name': project_names[0] if project_names else None,
        'first_episode_preview': extract_preview(session_data.get('first_episode_content')),
        'episodes': episodes,
    }


@router.get('/projects/{group_id}', status_code=status.HTTP_200_OK)
async def list_projects(
    group_id: str,
    graphiti: Annotated[ZepGraphiti, Depends(get_graphiti_from_path)],
    limit: int = Query(50, ge=1, le=500, description='Maximum number of projects to return'),
    cursor: str | None = Query(None, description='Pagination cursor (base64-encoded offset)'),
    name_filter: str | None = Query(
        None, description='Filter projects by name (case-insensitive substring match)'
    ),
    min_episodes: int | None = Query(None, ge=1, description='Filter projects with at least N episodes'),
    sort_order: str = Query('desc', pattern='^(asc|desc)$', description='Sort order by last activity'),
):
    """List all projects in a group with computed metadata.

    Projects are identified by unique (name, group_id) pairs.
    Metadata (episode_count, session_count, dates) computed on-demand via graph queries.
    """
    from graphiti_core.helpers import parse_db_date

    # Parse offset from cursor
    offset = 0
    if cursor:
        try:
            cursor_data = json.loads(base64.b64decode(cursor).decode('utf-8'))
            offset = cursor_data.get('offset', 0)
        except (ValueError, KeyError) as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f'Invalid cursor format: {e}',
            ) from e

    # Build WHERE clauses
    where_clauses = ['p.group_id = $group_id']
    query_params = {'group_id': group_id, 'limit': limit, 'offset': offset}

    if name_filter:
        where_clauses.append('toLower(p.name) CONTAINS toLower($name_filter)')
        query_params['name_filter'] = name_filter

    where_query = ' AND '.join(where_clauses)
    order_direction = 'DESC' if sort_order == 'desc' else 'ASC'

    # Count total projects
    count_query = f"""
        MATCH (p:Project)
        WHERE {where_query}
        RETURN count(p) AS total
    """
    count_result, _, _ = await graphiti.driver.execute_query(count_query, **query_params)
    total_count = count_result[0]['total'] if count_result else 0

    # Fetch projects with computed metadata
    projects_query = f"""
        MATCH (p:Project)
        WHERE {where_query}
        OPTIONAL MATCH (p)<-[:IN_PROJECT]-(e:Episodic)
        WITH p,
             count(DISTINCT e) AS episode_count,
             min(e.valid_at) AS first_episode_date,
             max(e.valid_at) AS last_episode_date
        OPTIONAL MATCH (p)<-[:PART_OF_PROJECT]-(s:Session)
        WITH p, episode_count, first_episode_date, last_episode_date,
             count(DISTINCT s) AS session_count
        WHERE episode_count IS NULL OR episode_count >= $min_episodes
        RETURN p.uuid AS uuid, p.name AS name, p.created_at AS created_at, p.project_path AS project_path,
               COALESCE(episode_count, 0) AS episode_count,
               COALESCE(session_count, 0) AS session_count,
               first_episode_date, last_episode_date
        ORDER BY last_episode_date {order_direction}
        SKIP $offset
        LIMIT $limit
    """
    query_params['min_episodes'] = min_episodes if min_episodes else 0

    records, _, _ = await graphiti.driver.execute_query(projects_query, **query_params)

    # Parse results
    from graph_service.dto.retrieve import ProjectResponse

    projects = []
    for record in records:
        projects.append(
            ProjectResponse(
                uuid=record['uuid'],
                name=record['name'],
                created_at=parse_db_date(record['created_at']),
                project_path=record.get('project_path'),
                episode_count=record['episode_count'],
                session_count=record['session_count'],
                first_episode_date=parse_db_date(record['first_episode_date'])
                if record['first_episode_date']
                else None,
                last_episode_date=parse_db_date(record['last_episode_date'])
                if record['last_episode_date']
                else None,
            )
        )

    # Generate next cursor
    next_cursor = None
    if len(projects) == limit and offset + limit < total_count:
        next_offset = offset + limit
        cursor_obj = {'offset': next_offset}
        next_cursor = base64.b64encode(json.dumps(cursor_obj).encode('utf-8')).decode('utf-8')

    from graph_service.dto.retrieve import ProjectListResponse

    return ProjectListResponse(
        projects=projects,
        total=total_count,
        has_more=len(projects) == limit and offset + limit < total_count,
        cursor=next_cursor,
    )


@router.get('/projects/{group_id}/stats/by-day', status_code=status.HTTP_200_OK)
async def get_project_stats_by_day(
    group_id: str,
    graphiti: Annotated[ZepGraphiti, Depends(get_graphiti_from_path)],
    project_name: str = Query(..., description='Project name to get stats for'),
    start_date: datetime | None = Query(None, description='Start date for stats (ISO 8601)'),
    end_date: datetime | None = Query(None, description='End date for stats (ISO 8601)'),
):
    """Get daily activity statistics for a specific project.

    Returns episode counts grouped by day for the specified project.
    """
    # Query via relationship
    query_params = {'group_id': group_id, 'project_name': project_name}

    # Build optional WHERE clauses for date filtering
    date_filters = []
    if start_date:
        date_filters.append('e.valid_at >= $start_date')
        query_params['start_date'] = start_date.isoformat()

    if end_date:
        date_filters.append('e.valid_at <= $end_date')
        query_params['end_date'] = end_date.isoformat()

    where_clause = f"WHERE {' AND '.join(date_filters)}" if date_filters else ""

    # Query for daily stats via relationship
    stats_query = f"""
        MATCH (e:Episodic {{group_id: $group_id}})-[:IN_PROJECT]->(p:Project {{name: $project_name}})
        {where_clause}
        WITH date(e.valid_at) AS day, count(e) AS episode_count
        RETURN toString(day) AS date, episode_count
        ORDER BY day ASC
    """

    records, _, _ = await graphiti.driver.execute_query(stats_query, **query_params)

    # Parse results
    from graph_service.dto.retrieve import DayProjectActivity, ProjectStatsByDayResponse

    stats = []
    for record in records:
        stats.append(
            DayProjectActivity(
                date=record['date'],
                episode_count=record['episode_count'],
            )
        )

    return ProjectStatsByDayResponse(
        stats=stats,
        total_days=len(stats),
    )


@router.get('/projects/{group_id}/{project_name}/episodes', status_code=status.HTTP_200_OK)
async def get_project_episodes(
    group_id: str,
    project_name: str,
    graphiti: Annotated[ZepGraphiti, Depends(get_graphiti_from_path)],
    limit: int = Query(50, ge=1, le=500, description='Maximum number of episodes to return'),
    cursor: str | None = Query(None, description='Pagination cursor (base64-encoded offset)'),
):
    """Get episodes for a specific project.

    Returns paginated list of episodes belonging to the specified project.
    """
    # Parse offset from cursor
    offset = 0
    if cursor:
        try:
            cursor_data = json.loads(base64.b64decode(cursor).decode('utf-8'))
            offset = cursor_data.get('offset', 0)
        except (ValueError, KeyError) as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f'Invalid cursor format: {e}',
            ) from e

    # Query via relationship, not field
    query_params = {'group_id': group_id, 'project_name': project_name, 'limit': limit, 'offset': offset}

    # Count total episodes
    count_query = """
        MATCH (e:Episodic {group_id: $group_id})-[:IN_PROJECT]->(p:Project {name: $project_name})
        RETURN count(e) AS total
    """
    count_result, _, _ = await graphiti.driver.execute_query(count_query, **query_params)
    total_count = count_result[0]['total'] if count_result else 0

    # Fetch episodes
    episodes_query = """
        MATCH (e:Episodic {group_id: $group_id})-[:IN_PROJECT]->(p:Project {name: $project_name})
        RETURN e.uuid AS uuid, e.name AS name, e.content AS content,
               e.valid_at AS valid_at, e.session_id AS session_id,
               p.name AS project_name, e.source_description AS source_description
        ORDER BY e.valid_at DESC
        SKIP $offset
        LIMIT $limit
    """

    records, _, _ = await graphiti.driver.execute_query(episodes_query, **query_params)

    # Parse results
    from graphiti_core.helpers import parse_db_date

    episodes = []
    for record in records:
        episodes.append({
            'uuid': record['uuid'],
            'name': record.get('name', ''),
            'content': record.get('content', ''),
            'valid_at': parse_db_date(record['valid_at']) if record.get('valid_at') else None,
            'session_id': record.get('session_id'),
            'project_name': record.get('project_name'),
            'source_description': record.get('source_description', ''),
        })

    # Generate next cursor
    next_cursor = None
    if len(episodes) == limit and offset + limit < total_count:
        next_offset = offset + limit
        cursor_obj = {'offset': next_offset}
        next_cursor = base64.b64encode(json.dumps(cursor_obj).encode('utf-8')).decode('utf-8')

    return {
        'episodes': episodes,
        'total': total_count,
        'has_more': len(episodes) == limit and offset + limit < total_count,
        'cursor': next_cursor,
    }


@router.get('/projects/{group_id}/{project_name}/sessions', status_code=status.HTTP_200_OK)
async def get_project_sessions(
    group_id: str,
    project_name: str,
    graphiti: Annotated[ZepGraphiti, Depends(get_graphiti_from_path)],
    limit: int = Query(50, ge=1, le=500, description='Maximum number of sessions to return'),
    cursor: str | None = Query(None, description='Pagination cursor (base64-encoded offset)'),
):
    """Get sessions for a specific project.

    Returns paginated list of sessions linked to the specified project via PART_OF_PROJECT relationship.
    """
    # Parse offset from cursor
    offset = 0
    if cursor:
        try:
            cursor_data = json.loads(base64.b64decode(cursor).decode('utf-8'))
            offset = cursor_data.get('offset', 0)
        except (ValueError, KeyError) as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f'Invalid cursor format: {e}',
            ) from e

    query_params = {'group_id': group_id, 'project_name': project_name, 'limit': limit, 'offset': offset}

    # Count total sessions
    count_query = """
        MATCH (s:Session {group_id: $group_id})-[:PART_OF_PROJECT]->(p:Project {name: $project_name})
        RETURN count(s) AS total
    """
    count_result, _, _ = await graphiti.driver.execute_query(count_query, **query_params)
    total_count = count_result[0]['total'] if count_result else 0

    # Fetch sessions with metadata
    sessions_query = """
        MATCH (s:Session {group_id: $group_id})-[:PART_OF_PROJECT]->(p:Project {name: $project_name})
        OPTIONAL MATCH (e:Episodic {session_id: s.session_id, group_id: $group_id})
        WITH s, p,
             count(DISTINCT e) AS episode_count,
             min(e.valid_at) AS first_episode_date,
             max(e.valid_at) AS last_episode_date,
             collect(DISTINCT e.source_description) AS source_descriptions
        OPTIONAL MATCH (first_ep:Episodic {session_id: s.session_id, group_id: $group_id})
        WHERE first_ep.valid_at = first_episode_date
        WITH s, p, episode_count, first_episode_date, last_episode_date, source_descriptions,
             head(collect(first_ep.content)) AS first_episode_content
        RETURN s.session_id AS session_id,
               s.summary AS summary,
               episode_count,
               first_episode_date,
               last_episode_date,
               source_descriptions,
               p.name AS project_name,
               first_episode_content
        ORDER BY last_episode_date DESC
        SKIP $offset
        LIMIT $limit
    """

    records, _, _ = await graphiti.driver.execute_query(sessions_query, **query_params)

    # Parse results into DTOs
    from graphiti_core.helpers import parse_db_date

    sessions = []
    for record in records:
        sessions.append(
            SessionResponse(
                session_id=record['session_id'],
                episode_count=record['episode_count'] if record['episode_count'] else 0,
                first_episode_date=parse_db_date(record['first_episode_date']) if record.get('first_episode_date') else None,
                last_episode_date=parse_db_date(record['last_episode_date']) if record.get('last_episode_date') else None,
                source_descriptions=record.get('source_descriptions', []),
                summary=record.get('summary'),
                project_name=record.get('project_name'),
                first_episode_preview=extract_preview(record.get('first_episode_content')),
            )
        )

    # Generate next cursor
    next_cursor = None
    if len(sessions) == limit and offset + limit < total_count:
        next_offset = offset + limit
        cursor_obj = {'offset': next_offset}
        next_cursor = base64.b64encode(json.dumps(cursor_obj).encode('utf-8')).decode('utf-8')

    return SessionListResponse(
        sessions=sessions,
        total=total_count,
        has_more=len(sessions) == limit and offset + limit < total_count,
        cursor=next_cursor,
    )


@router.get('/projects/{group_id}/{project_name}/entities', status_code=status.HTTP_200_OK)
async def get_project_entities(
    group_id: str,
    project_name: str,
    graphiti: Annotated[ZepGraphiti, Depends(get_graphiti_from_path)],
    limit: int = Query(50, ge=1, le=500, description='Maximum number of entities to return'),
    cursor: str | None = Query(None, description='Pagination cursor (base64-encoded offset)'),
):
    """Get entities mentioned in episodes for a specific project.

    Returns paginated list of entities (Person, Organization, etc.) that are mentioned
    in episodes belonging to the specified project, ordered by mention count.
    """
    # Parse offset from cursor
    offset = 0
    if cursor:
        try:
            cursor_data = json.loads(base64.b64decode(cursor).decode('utf-8'))
            offset = cursor_data.get('offset', 0)
        except (ValueError, KeyError) as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f'Invalid cursor format: {e}',
            ) from e

    query_params = {'group_id': group_id, 'project_name': project_name, 'limit': limit, 'offset': offset}

    # Count total unique entities
    count_query = """
        MATCH (e:Episodic {group_id: $group_id})-[:IN_PROJECT]->(p:Project {name: $project_name})
        MATCH (e)-[:MENTIONS]->(entity:Entity)
        RETURN count(DISTINCT entity) AS total
    """
    count_result, _, _ = await graphiti.driver.execute_query(count_query, **query_params)
    total_count = count_result[0]['total'] if count_result else 0

    # Fetch entities with mention counts
    entities_query = """
        MATCH (e:Episodic {group_id: $group_id})-[:IN_PROJECT]->(p:Project {name: $project_name})
        MATCH (e)-[:MENTIONS]->(entity:Entity)
        WITH entity, count(DISTINCT e) AS mention_count
        RETURN entity.uuid AS uuid,
               entity.name AS name,
               entity.group_id AS group_id,
               entity.summary AS summary,
               labels(entity) AS labels,
               entity.created_at AS created_at,
               mention_count
        ORDER BY mention_count DESC, entity.name ASC
        SKIP $offset
        LIMIT $limit
    """

    records, _, _ = await graphiti.driver.execute_query(entities_query, **query_params)

    # Parse results
    from graphiti_core.helpers import parse_db_date

    entities = []
    for record in records:
        # Filter out 'Entity' from labels to get specific types
        entity_labels = [label for label in record.get('labels', []) if label != 'Entity']

        entities.append({
            'uuid': record['uuid'],
            'name': record['name'],
            'group_id': record['group_id'],
            'summary': record.get('summary', ''),
            'labels': entity_labels,
            'created_at': parse_db_date(record['created_at']) if record.get('created_at') else None,
            'mention_count': record['mention_count'],
        })

    # Generate next cursor
    next_cursor = None
    if len(entities) == limit and offset + limit < total_count:
        next_offset = offset + limit
        cursor_obj = {'offset': next_offset}
        next_cursor = base64.b64encode(json.dumps(cursor_obj).encode('utf-8')).decode('utf-8')

    return {
        'entities': entities,
        'total': total_count,
        'has_more': len(entities) == limit and offset + limit < total_count,
        'cursor': next_cursor,
    }
