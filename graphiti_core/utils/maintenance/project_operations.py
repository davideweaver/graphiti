"""
Project node management operations.
Handles creation and linking of project nodes.
"""

import logging

from graphiti_core.driver.driver import GraphDriver
from graphiti_core.nodes import ProjectNode
from graphiti_core.utils.datetime_utils import utc_now

logger = logging.getLogger(__name__)


async def get_or_create_project_node(
    driver: GraphDriver,
    group_id: str,
    project_name: str,
    project_path: str | None = None,
) -> ProjectNode:
    """
    Get an existing ProjectNode or create a new one.

    Parameters
    ----------
    driver : GraphDriver
        The graph database driver
    group_id : str
        The group ID for the project
    project_name : str
        The name of the project (already lowercased)
    project_path : str | None, optional
        The file system path of the project. If provided, updates the project node.

    Returns
    -------
    ProjectNode
        Existing or newly created project node
    """
    # Try to fetch existing project node
    existing_project = await ProjectNode.get_by_name(driver, group_id, project_name)

    if existing_project is not None:
        # If project_path is provided, update the existing project
        if project_path is not None:
            existing_project.project_path = project_path
            await existing_project.save(driver)
            logger.debug(f'Updated ProjectNode {existing_project.uuid} with project_path: {project_path}')
        return existing_project

    # Create new project node
    logger.debug(f'Creating new ProjectNode for name: {project_name}, group_id: {group_id}')

    new_project = ProjectNode(
        name=project_name,
        group_id=group_id,
        project_path=project_path,
        created_at=utc_now(),
    )

    return new_project


async def link_episode_to_project(
    driver: GraphDriver,
    episode_uuid: str,
    project_uuid: str,
) -> None:
    """
    Create a relationship between an episode and its project.

    Creates a (Episode)-[:IN_PROJECT]->(Project) relationship in the graph.

    Parameters
    ----------
    driver : GraphDriver
        The graph database driver
    episode_uuid : str
        UUID of the episode
    project_uuid : str
        UUID of the project node

    Returns
    -------
    None
    """
    query = """
        MATCH (e:Episodic {uuid: $episode_uuid})
        MATCH (p:Project {uuid: $project_uuid})
        MERGE (e)-[:IN_PROJECT]->(p)
        RETURN e.uuid AS episode_uuid, p.uuid AS project_uuid
    """

    try:
        result = await driver.execute_query(
            query,
            episode_uuid=episode_uuid,
            project_uuid=project_uuid,
        )
        logger.debug(
            f'Linked episode {episode_uuid} to project {project_uuid}: '
            f'{len(result[0])} relationships created/verified'
        )
    except Exception as e:
        logger.error(
            f'Failed to link episode {episode_uuid} to project {project_uuid}: '
            f'{type(e).__name__}: {e}'
        )
        # Don't raise - link failure shouldn't block episode ingestion


async def link_session_to_project(
    driver: GraphDriver,
    session_uuid: str,
    project_uuid: str,
) -> None:
    """
    Create a relationship between a session and its project.

    Creates a (Session)-[:PART_OF_PROJECT]->(Project) relationship in the graph.

    Parameters
    ----------
    driver : GraphDriver
        The graph database driver
    session_uuid : str
        UUID of the session node
    project_uuid : str
        UUID of the project node

    Returns
    -------
    None
    """
    query = """
        MATCH (s:Session {uuid: $session_uuid})
        MATCH (p:Project {uuid: $project_uuid})
        MERGE (s)-[:PART_OF_PROJECT]->(p)
        RETURN s.uuid AS session_uuid, p.uuid AS project_uuid
    """

    try:
        result = await driver.execute_query(
            query,
            session_uuid=session_uuid,
            project_uuid=project_uuid,
        )
        logger.debug(
            f'Linked session {session_uuid} to project {project_uuid}: '
            f'{len(result[0])} relationships created/verified'
        )
    except Exception as e:
        logger.error(
            f'Failed to link session {session_uuid} to project {project_uuid}: '
            f'{type(e).__name__}: {e}'
        )
        # Don't raise - link failure shouldn't block operations
