from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from graphiti_core.errors import NodeNotFoundError  # type: ignore
from graphiti_core.nodes import EntityNode  # type: ignore

from graph_service.dto import (
    EntityListResponse,
    GetMemoryRequest,
    GetMemoryResponse,
    Message,
    SearchQuery,
    SearchResults,
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
    last_n: int,
    graphiti: Annotated[ZepGraphiti, Depends(get_graphiti_from_path)],
):
    episodes = await graphiti.retrieve_episodes(
        group_ids=[group_id], last_n=last_n, reference_time=datetime.now(timezone.utc)
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


@router.get('/entities/{uuid}', status_code=status.HTTP_200_OK)
async def get_entity(
    uuid: str,
    group_id: str,
    graphiti: Annotated[ZepGraphiti, Depends(get_graphiti_from_query)],
):
    """Get a single entity node by UUID."""
    try:
        entity = await EntityNode.get_by_uuid(graphiti.driver, uuid)
        return get_entity_node_response(entity)
    except NodeNotFoundError:
        raise HTTPException(status_code=404, detail=f'Entity not found: {uuid}')


@router.get('/entities', status_code=status.HTTP_200_OK)
async def list_entities(
    group_id: str = Query(..., description='The group ID to filter entities'),
    limit: int = Query(50, ge=1, le=500, description='Maximum number of entities to return'),
    cursor: str | None = Query(None, description='Pagination cursor (UUID of last entity)'),
    with_embeddings: bool = Query(False, description='Include name embeddings in response'),
    graphiti: Annotated[ZepGraphiti, Depends(get_graphiti_from_query)] = None,
):
    """List entities for a group with pagination."""
    entities = await EntityNode.get_by_group_ids(
        graphiti.driver,
        group_ids=[group_id],
        limit=limit,
        uuid_cursor=cursor,
        with_embeddings=with_embeddings,
    )

    return EntityListResponse(
        entities=[get_entity_node_response(e) for e in entities],
        total=len(entities),
        has_more=len(entities) == limit,
        cursor=entities[-1].uuid if entities else None,
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


@router.get('/entities/{uuid}/relationships', status_code=status.HTTP_200_OK)
async def get_entity_relationships(
    uuid: str,
    group_id: str,
    graphiti: Annotated[ZepGraphiti, Depends(get_graphiti_from_query)],
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
        logging.error(f"Error fetching relationships: {e}")
        # If query fails, return empty list
        return EntityListResponse(entities=[], total=0, has_more=False, cursor=None)
