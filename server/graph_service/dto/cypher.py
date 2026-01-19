from typing import Any

from pydantic import BaseModel, Field


class CypherQueryRequest(BaseModel):
    """Request model for executing raw Cypher queries."""

    query: str = Field(..., description='Cypher query to execute')
    parameters: dict[str, Any] = Field(
        default_factory=dict, description='Query parameters (use $param syntax in query)'
    )
    group_id: str = Field(..., description='Graph group ID to query against')

    model_config = {
        'json_schema_extra': {
            'examples': [
                {
                    'query': 'MATCH (e:Entity) WHERE e.group_id = $group_id RETURN e.name LIMIT 10',
                    'parameters': {},
                    'group_id': 'dave-weaver',
                }
            ]
        }
    }


class CypherQueryResult(BaseModel):
    """Response model for Cypher query results."""

    header: list[str] = Field(description='Column names')
    records: list[dict[str, Any]] = Field(description='Result rows as dicts')
    row_count: int = Field(description='Number of rows returned')


class NodeSchema(BaseModel):
    """Schema information for a node type."""

    label: str
    properties: dict[str, str]  # property_name: property_type
    count: int


class RelationshipSchema(BaseModel):
    """Schema information for a relationship type."""

    type: str
    properties: dict[str, str]
    source_labels: list[str]
    target_labels: list[str]
    count: int


class SchemaResponse(BaseModel):
    """Graph schema for LLM context."""

    group_id: str
    nodes: list[NodeSchema]
    relationships: list[RelationshipSchema]
    total_nodes: int
    total_relationships: int
