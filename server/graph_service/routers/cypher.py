import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from graph_service.dto.cypher import (
    CypherQueryRequest,
    CypherQueryResult,
    NodeSchema,
    RelationshipSchema,
    SchemaResponse,
)
from graph_service.zep_graphiti import (
    ZepGraphiti,
    get_graphiti_from_body,
    get_graphiti_from_query,
)

router = APIRouter(prefix='/cypher', tags=['cypher'])
logger = logging.getLogger(__name__)


@router.post('/query', status_code=status.HTTP_200_OK)
async def execute_cypher_query(
    request: CypherQueryRequest,
    graphiti: Annotated[ZepGraphiti, Depends(get_graphiti_from_body)] = None,
) -> CypherQueryResult:
    """
    Execute a raw Cypher query against the FalkorDB graph.

    **Important:**
    - Always include `WHERE n.group_id = $group_id` for multi-tenancy
    - Use $parameter_name syntax for parameters
    - Read-only queries recommended (MATCH, RETURN)
    - DateTime objects automatically converted to ISO strings

    **Example:**
    ```json
    {
      "query": "MATCH (e:Entity) WHERE e.group_id = $group_id AND e.name CONTAINS $term RETURN e.name, e.entity_type LIMIT 10",
      "parameters": {"term": "Python"},
      "group_id": "dave-weaver"
    }
    ```
    """
    try:
        logger.info(f'Executing Cypher query for group_id={request.group_id}')

        # Add group_id to parameters automatically
        params = {**request.parameters, 'group_id': request.group_id}

        # Execute query via FalkorDB driver
        records, header, _ = await graphiti.driver.execute_query(request.query, **params)

        logger.info(f'Query returned {len(records) if records else 0} rows')

        return CypherQueryResult(
            header=header or [],
            records=records or [],
            row_count=len(records) if records else 0,
        )

    except ValueError as e:
        # Query syntax errors, parameter issues
        logger.warning(f'Query validation error: {e}')
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f'Invalid query: {str(e)}'
        )
    except Exception as e:
        # Database errors, connection issues
        logger.error(f'Query execution failed: {e}', exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'Query execution failed: {str(e)}',
        )


@router.get('/schema', status_code=status.HTTP_200_OK)
async def get_graph_schema(
    group_id: str,
    graphiti: Annotated[ZepGraphiti, Depends(get_graphiti_from_query)] = None,
) -> SchemaResponse:
    """
    Get graph schema for LLM context generation.

    Returns node types, relationship types, properties, and counts.
    This information helps LLMs generate accurate Cypher queries.

    **Use case:** Pass this schema to an LLM with natural language query
    to generate appropriate Cypher syntax.
    """
    try:
        # Get node schema
        node_query = """
        MATCH (n)
        WHERE n.group_id = $group_id
        WITH labels(n) AS node_labels, n
        UNWIND node_labels AS label
        WITH label, keys(n) AS props
        RETURN label,
               collect(DISTINCT props) AS property_sets,
               count(*) AS node_count
        """

        node_records, _, _ = await graphiti.driver.execute_query(node_query, group_id=group_id)

        # Get relationship schema
        rel_query = """
        MATCH (source)-[r]->(target)
        WHERE source.group_id = $group_id
        WITH type(r) AS rel_type,
             labels(source) AS source_labels,
             labels(target) AS target_labels,
             keys(r) AS props,
             r
        RETURN rel_type,
               collect(DISTINCT props) AS property_sets,
               collect(DISTINCT source_labels) AS all_source_labels,
               collect(DISTINCT target_labels) AS all_target_labels,
               count(*) AS rel_count
        """

        rel_records, _, _ = await graphiti.driver.execute_query(rel_query, group_id=group_id)

        # Get total counts
        count_query = """
        MATCH (n)
        WHERE n.group_id = $group_id
        WITH count(n) AS node_count
        MATCH ()-[r]->()
        RETURN node_count, count(r) AS rel_count
        """

        count_records, _, _ = await graphiti.driver.execute_query(count_query, group_id=group_id)

        total_nodes = count_records[0]['node_count'] if count_records else 0
        total_rels = count_records[0]['rel_count'] if count_records else 0

        # Parse node schema
        nodes = []
        for record in node_records:
            # Flatten property sets and infer types
            all_props = {}
            for prop_set in record['property_sets']:
                for prop in prop_set:
                    all_props[prop] = 'string'  # FalkorDB doesn't expose type info easily

            nodes.append(
                NodeSchema(
                    label=record['label'], properties=all_props, count=record['node_count']
                )
            )

        # Parse relationship schema
        relationships = []
        for record in rel_records:
            all_props = {}
            for prop_set in record['property_sets']:
                for prop in prop_set:
                    all_props[prop] = 'string'

            # Flatten label sets
            source_labels = list(
                set(label for label_list in record['all_source_labels'] for label in label_list)
            )
            target_labels = list(
                set(label for label_list in record['all_target_labels'] for label in label_list)
            )

            relationships.append(
                RelationshipSchema(
                    type=record['rel_type'],
                    properties=all_props,
                    source_labels=source_labels,
                    target_labels=target_labels,
                    count=record['rel_count'],
                )
            )

        return SchemaResponse(
            group_id=group_id,
            nodes=nodes,
            relationships=relationships,
            total_nodes=total_nodes,
            total_relationships=total_rels,
        )

    except Exception as e:
        logger.error(f'Schema introspection failed: {e}', exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'Failed to retrieve schema: {str(e)}',
        )
