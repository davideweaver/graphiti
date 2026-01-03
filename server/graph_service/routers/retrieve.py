import base64
import json
from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from graphiti_core.errors import NodeNotFoundError  # type: ignore
from graphiti_core.nodes import EntityNode  # type: ignore

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


@router.post('/search', status_code=status.HTTP_200_OK)
async def search(
    query: SearchQuery,
    graphiti: Annotated[ZepGraphiti, Depends(get_graphiti_from_body)],
):
    relevant_edges = await graphiti.search(
        group_ids=[query.group_id],  # Changed from query.group_ids
        query=query.query,
        num_results=query.max_facts,
    )
    facts = [get_fact_result_from_edge(edge) for edge in relevant_edges]
    return SearchResults(
        facts=facts,
    )


@router.get('/entity-edge/{uuid}', status_code=status.HTTP_200_OK)
async def get_entity_edge(
    uuid: str,
    group_id: str,
    graphiti: Annotated[ZepGraphiti, Depends(get_graphiti_from_query)],
):
    entity_edge = await graphiti.get_entity_edge(uuid)
    return get_fact_result_from_edge(entity_edge)


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
    result = await graphiti.search(
        group_ids=[request.group_id],
        query=combined_query,
        num_results=request.max_facts,
    )
    facts = [get_fact_result_from_edge(edge) for edge in result]
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

    # Fetch sessions with metadata, optionally joining with SessionNode for summaries
    sessions_query = f"""
        MATCH (e:Episodic)
        WHERE {where_query}
        WITH e.session_id AS session_id,
             count(e) AS episode_count,
             min(e.valid_at) AS first_episode_date,
             max(e.valid_at) AS last_episode_date,
             collect(DISTINCT e.source_description) AS source_descriptions
        OPTIONAL MATCH (s:Session {{session_id: session_id, group_id: $group_id}})
        RETURN session_id, episode_count, first_episode_date,
               last_episode_date, source_descriptions, s.summary AS summary
        ORDER BY last_episode_date {order_direction}
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
                episode_count=record['episode_count'],
                first_episode_date=parse_db_date(record['first_episode_date']),
                last_episode_date=parse_db_date(record['last_episode_date']),
                source_descriptions=record['source_descriptions'],
                summary=record.get('summary'),
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
    """Get all episodes for a specific session by UUID.

    Returns the full episode data for all episodes in the session.
    Returns 404 if session does not exist (no episodes found).
    """
    # Retrieve all episodes for this session
    # Use a large last_n value to get all episodes (no practical limit)
    episodes = await graphiti.retrieve_episodes(
        group_ids=[group_id],
        last_n=10000,  # Large enough to get all episodes in a session
        reference_time=datetime.now(timezone.utc),
        session_id=session_id,
    )

    if not episodes:
        raise HTTPException(status_code=404, detail=f'Session not found: {session_id}')

    return episodes
