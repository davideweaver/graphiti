"""Generic graph navigation DTOs for thin FalkorDB layer."""
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel


class GraphNode(BaseModel):
    """Generic node representation for any node type in the graph.

    Represents Entity, Episodic, Session, Project, or Community nodes
    with a unified interface regardless of specific node type.
    """
    uuid: str  # Node UUID (or session_id for Session nodes)
    node_type: str  # "Entity", "Episodic", "Session", "Project", "Community"
    label: str  # Primary label for display (name field)
    labels: list[str]  # All node labels (e.g., ["Person", "Entity"])
    metadata: dict[str, Any]  # All node properties (group_id, summary, content, etc.)
    created_at: datetime | None  # Common field across most node types


class GraphEdge(BaseModel):
    """Generic edge representation for any relationship in the graph.

    Represents RELATES_TO, MENTIONS, HAS_MEMBER, and other relationship types
    with a unified interface.
    """
    uuid: str  # Edge UUID
    edge_type: str  # "RELATES_TO", "MENTIONS", "HAS_MEMBER", "IN_PROJECT", "PART_OF_PROJECT"
    label: str  # Edge name/fact for display
    source_uuid: str  # Source node UUID
    target_uuid: str  # Target node UUID
    metadata: dict[str, Any]  # All edge properties (fact, valid_at, invalid_at, etc.)
    created_at: datetime | None


class GraphConnection(BaseModel):
    """A connected node and its relationship to the center node."""
    node: GraphNode
    relationship: GraphEdge
    direction: Literal["incoming", "outgoing"]  # Direction relative to center node


class NodeConnectionsResponse(BaseModel):
    """Response for GET /graph/nodes/{uuid}/connections"""
    center_node: GraphNode
    connections: list[GraphConnection]
    total_connections: int


class EdgeConnectionsResponse(BaseModel):
    """Response for GET /graph/edges/{uuid}/connections"""
    edge: GraphEdge
    source: GraphNode
    target: GraphNode
