"""Generic graph navigation router - thin layer over FalkorDB.

Provides unified API for navigating any node or edge type without requiring
type-specific endpoints. Executes raw Cypher queries directly against FalkorDB.
"""
import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from graphiti_core.helpers import parse_db_date  # type: ignore

from graph_service.dto import (
    EdgeConnectionsResponse,
    GraphConnection,
    GraphEdge,
    GraphNode,
    NodeConnectionsResponse,
)
from graph_service.zep_graphiti import (
    ZepGraphiti,
    get_graphiti_from_query,
)

router = APIRouter()
logger = logging.getLogger(__name__)


def serialize_node(node_data: Any, node_labels: list[str]) -> GraphNode:
    """Serialize a FalkorDB node into a GraphNode DTO.

    Args:
        node_data: Node data from Cypher query result (could be Node object or dict)
        node_labels: Node labels from labels(n)

    Returns:
        GraphNode with type, label, and metadata
    """
    # Convert Node object to dict if needed
    if hasattr(node_data, 'properties'):
        # FalkorDB Node object
        node_props = node_data.properties
    elif isinstance(node_data, dict):
        # Already a dict
        node_props = node_data
    else:
        # Try to convert to dict
        node_props = dict(node_data) if node_data else {}

    # Determine primary node type (first non-generic label)
    node_type = next((l for l in node_labels if l not in ['Node']), node_labels[0] if node_labels else 'Unknown')

    # Extract label (name field is common across node types)
    # Fallback: session_id for Sessions, uuid as last resort
    label = node_props.get('name', node_props.get('session_id', node_props.get('uuid', 'Unknown')))

    # Build metadata dict excluding uuid/session_id (already in top-level uuid field)
    metadata = {k: v for k, v in node_props.items() if k not in ['uuid', 'session_id']}

    # Extract UUID (all nodes should have uuid, including Sessions)
    node_uuid = node_props.get('uuid')

    return GraphNode(
        uuid=node_uuid,
        node_type=node_type,
        label=label,
        labels=node_labels,
        metadata=metadata,
        created_at=parse_db_date(node_props.get('created_at')) if node_props.get('created_at') else None
    )


def serialize_edge(
    edge_data: Any,
    edge_type: str,
    source_uuid: str | None = None,
    target_uuid: str | None = None
) -> GraphEdge:
    """Serialize a FalkorDB relationship into a GraphEdge DTO.

    Args:
        edge_data: Edge data from Cypher query result (could be Edge object or dict)
        edge_type: Relationship type from type(r)
        source_uuid: Source node UUID (from edge_props or query)
        target_uuid: Target node UUID (from edge_props or query)

    Returns:
        GraphEdge with type, label, and metadata
    """
    # Convert Edge object to dict if needed
    if hasattr(edge_data, 'properties'):
        # FalkorDB Edge object
        edge_props = edge_data.properties
    elif isinstance(edge_data, dict):
        # Already a dict
        edge_props = edge_data
    else:
        # Try to convert to dict
        edge_props = dict(edge_data) if edge_data else {}

    # Label is 'name' for RELATES_TO, or 'fact', or edge type as fallback
    label = edge_props.get('name', edge_props.get('fact', edge_type))

    # Extract source/target UUIDs
    if source_uuid is None:
        source_uuid = edge_props.get('source_node_uuid', '')
    if target_uuid is None:
        target_uuid = edge_props.get('target_node_uuid', '')

    # Metadata includes all properties except uuid/source/target
    metadata = {
        k: v for k, v in edge_props.items()
        if k not in ['uuid', 'source_node_uuid', 'target_node_uuid']
    }

    return GraphEdge(
        uuid=edge_props.get('uuid', ''),
        edge_type=edge_type,
        label=label,
        source_uuid=source_uuid,
        target_uuid=target_uuid,
        metadata=metadata,
        created_at=parse_db_date(edge_props.get('created_at')) if edge_props.get('created_at') else None
    )


@router.get('/graph/nodes/{uuid}', status_code=status.HTTP_200_OK)
async def get_graph_node(
    uuid: str,
    group_id: str = Query(..., description='Group ID for multi-tenant filtering'),
    graphiti: Annotated[ZepGraphiti, Depends(get_graphiti_from_query)] = None,
) -> GraphNode:
    """Get any node by UUID with auto-detected type.

    Tries standard UUID lookup first, then falls back to session_id lookup
    for Session nodes (which use session_id instead of uuid).

    Returns:
        GraphNode with unified interface regardless of node type
    """
    # Try standard UUID lookup
    query = """
    MATCH (n)
    WHERE n.uuid = $uuid AND n.group_id = $group_id
    RETURN n, labels(n) AS node_labels
    LIMIT 1
    """

    records, _, _ = await graphiti.driver.execute_query(query, uuid=uuid, group_id=group_id)

    # If no result, try Session node lookup (uses session_id not uuid)
    if not records:
        fallback_query = """
        MATCH (n:Session)
        WHERE n.session_id = $uuid AND n.group_id = $group_id
        RETURN n, labels(n) AS node_labels
        LIMIT 1
        """
        records, _, _ = await graphiti.driver.execute_query(fallback_query, uuid=uuid, group_id=group_id)

    if not records:
        raise HTTPException(
            status_code=404,
            detail=f'Node not found: {uuid} (group_id: {group_id})'
        )

    record = records[0]
    return serialize_node(record['n'], record['node_labels'])


@router.get('/graph/nodes/{uuid}/connections', status_code=status.HTTP_200_OK)
async def get_node_connections(
    uuid: str,
    group_id: str = Query(..., description='Group ID for multi-tenant filtering'),
    graphiti: Annotated[ZepGraphiti, Depends(get_graphiti_from_query)] = None,
) -> NodeConnectionsResponse:
    """Get all connections (incoming and outgoing) for any node.

    Returns bidirectional relationships with connected nodes. Works for all node types.

    Returns:
        Center node + list of connections with relationship details
    """
    # Try Session node first (matches by session_id)
    session_query = """
    MATCH (center:Session)
    WHERE center.session_id = $uuid AND center.group_id = $group_id

    OPTIONAL MATCH (center)-[r_out]->(connected_out)
    WHERE connected_out.group_id = $group_id

    OPTIONAL MATCH (connected_in)-[r_in]->(center)
    WHERE connected_in.group_id = $group_id

    RETURN
      center,
      labels(center) AS center_labels,
      collect(DISTINCT {
        node: connected_out,
        node_labels: labels(connected_out),
        relationship: r_out,
        rel_type: type(r_out),
        source_uuid: center.uuid,
        target_uuid: connected_out.uuid,
        direction: 'outgoing'
      }) AS outgoing,
      collect(DISTINCT {
        node: connected_in,
        node_labels: labels(connected_in),
        relationship: r_in,
        rel_type: type(r_in),
        source_uuid: connected_in.uuid,
        target_uuid: center.uuid,
        direction: 'incoming'
      }) AS incoming
    """

    records, _, _ = await graphiti.driver.execute_query(session_query, uuid=uuid, group_id=group_id)

    # If no Session found, try by UUID (for Entity, Episodic, Project, Community nodes)
    if not records:
        uuid_query = """
        MATCH (center)
        WHERE center.uuid = $uuid AND center.group_id = $group_id

        OPTIONAL MATCH (center)-[r_out]->(connected_out)
        WHERE connected_out.group_id = $group_id

        OPTIONAL MATCH (connected_in)-[r_in]->(center)
        WHERE connected_in.group_id = $group_id

        RETURN
          center,
          labels(center) AS center_labels,
          collect(DISTINCT {
            node: connected_out,
            node_labels: labels(connected_out),
            relationship: r_out,
            rel_type: type(r_out),
            source_uuid: center.uuid,
            target_uuid: connected_out.uuid,
            direction: 'outgoing'
          }) AS outgoing,
          collect(DISTINCT {
            node: connected_in,
            node_labels: labels(connected_in),
            relationship: r_in,
            rel_type: type(r_in),
            source_uuid: connected_in.uuid,
            target_uuid: center.uuid,
            direction: 'incoming'
          }) AS incoming
        """
        records, _, _ = await graphiti.driver.execute_query(uuid_query, uuid=uuid, group_id=group_id)

    if not records:
        raise HTTPException(
            status_code=404,
            detail=f'Node not found: {uuid} (group_id: {group_id})'
        )

    record = records[0]

    # Serialize center node
    center_node = serialize_node(record['center'], record['center_labels'])

    # Process connections (filter out null connections from OPTIONAL MATCH)
    connections: list[GraphConnection] = []

    # Outgoing connections
    for conn_data in record['outgoing']:
        if conn_data['node'] is not None and conn_data['relationship'] is not None:
            connected_node = serialize_node(conn_data['node'], conn_data['node_labels'])

            # Handle relationship (could be Edge object or dict)
            edge = serialize_edge(
                conn_data['relationship'],
                conn_data['rel_type'],
                source_uuid=conn_data.get('source_uuid'),
                target_uuid=conn_data.get('target_uuid')
            )

            connections.append(GraphConnection(
                node=connected_node,
                relationship=edge,
                direction='outgoing'
            ))

    # Incoming connections
    for conn_data in record['incoming']:
        if conn_data['node'] is not None and conn_data['relationship'] is not None:
            connected_node = serialize_node(conn_data['node'], conn_data['node_labels'])

            # Handle relationship (could be Edge object or dict)
            edge = serialize_edge(
                conn_data['relationship'],
                conn_data['rel_type'],
                source_uuid=conn_data.get('source_uuid'),
                target_uuid=conn_data.get('target_uuid')
            )

            connections.append(GraphConnection(
                node=connected_node,
                relationship=edge,
                direction='incoming'
            ))

    return NodeConnectionsResponse(
        center_node=center_node,
        connections=connections,
        total_connections=len(connections)
    )


@router.get('/graph/edges/{uuid}', status_code=status.HTTP_200_OK)
async def get_graph_edge(
    uuid: str,
    group_id: str = Query(..., description='Group ID for multi-tenant filtering'),
    graphiti: Annotated[ZepGraphiti, Depends(get_graphiti_from_query)] = None,
) -> GraphEdge:
    """Get any edge (relationship) by UUID.

    Currently supports RELATES_TO edges (facts). Other edge types can be added as needed.

    Returns:
        GraphEdge with source/target UUIDs and metadata
    """
    # Focus on RELATES_TO edges (facts) for MVP
    query = """
    MATCH ()-[r:RELATES_TO]-()
    WHERE r.uuid = $uuid AND r.group_id = $group_id
    RETURN r,
           type(r) AS edge_type,
           r.source_node_uuid AS source_uuid,
           r.target_node_uuid AS target_uuid
    LIMIT 1
    """

    records, _, _ = await graphiti.driver.execute_query(query, uuid=uuid, group_id=group_id)

    if not records:
        raise HTTPException(
            status_code=404,
            detail=f'Edge not found: {uuid} (group_id: {group_id})'
        )

    record = records[0]
    return serialize_edge(
        record['r'],
        record['edge_type'],
        source_uuid=record['source_uuid'],
        target_uuid=record['target_uuid']
    )


@router.get('/graph/edges/{uuid}/connections', status_code=status.HTTP_200_OK)
async def get_edge_connections(
    uuid: str,
    group_id: str = Query(..., description='Group ID for multi-tenant filtering'),
    graphiti: Annotated[ZepGraphiti, Depends(get_graphiti_from_query)] = None,
) -> EdgeConnectionsResponse:
    """Get source and target nodes for an edge (relationship).

    Returns the edge along with full details of its source and target nodes.

    Returns:
        Edge + source node + target node
    """
    query = """
    MATCH (source)-[r:RELATES_TO]->(target)
    WHERE r.uuid = $uuid AND r.group_id = $group_id
    RETURN r, type(r) AS edge_type,
           source, labels(source) AS source_labels,
           target, labels(target) AS target_labels
    LIMIT 1
    """

    records, _, _ = await graphiti.driver.execute_query(query, uuid=uuid, group_id=group_id)

    if not records:
        raise HTTPException(
            status_code=404,
            detail=f'Edge not found: {uuid} (group_id: {group_id})'
        )

    record = records[0]

    # Serialize source and target nodes first to get their UUIDs
    source = serialize_node(record['source'], record['source_labels'])
    target = serialize_node(record['target'], record['target_labels'])

    # Serialize edge with source/target UUIDs
    edge = serialize_edge(
        record['r'],
        record['edge_type'],
        source_uuid=source.uuid,
        target_uuid=target.uuid
    )

    return EdgeConnectionsResponse(
        edge=edge,
        source=source,
        target=target
    )
