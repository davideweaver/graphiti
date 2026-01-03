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

import logging
from datetime import datetime

from graphiti_core.driver.driver import GraphDriver
from graphiti_core.llm_client import LLMClient
from graphiti_core.nodes import EpisodicNode, SessionNode
from graphiti_core.prompts.summarize_sessions import (
    SessionSummary,
    create_initial_session_summary,
    summarize_session_incremental,
)
from graphiti_core.utils.datetime_utils import utc_now

logger = logging.getLogger(__name__)


async def get_or_create_session_node(
    driver: GraphDriver,
    group_id: str,
    session_id: str,
    episode_time: datetime,
    source_description: str | None = None,
) -> SessionNode:
    """
    Get an existing SessionNode or create a new one.

    Parameters
    ----------
    driver : GraphDriver
        The graph database driver
    group_id : str
        The group ID for the session
    session_id : str
        The UUID of the session
    episode_time : datetime
        The timestamp of the episode (used for first/last episode date)
    source_description : str | None
        The source description from the episode (optional)

    Returns
    -------
    SessionNode
        Existing or newly created session node
    """
    # Try to fetch existing session node
    existing_session = await SessionNode.get_by_session_id(driver, group_id, session_id)

    if existing_session is not None:
        return existing_session

    # Create new session node
    logger.debug(f'Creating new SessionNode for session_id: {session_id}, group_id: {group_id}')

    source_descriptions = [source_description] if source_description else []

    new_session = SessionNode(
        name=f'Session {session_id[:8]}',  # Short name for display
        session_id=session_id,
        group_id=group_id,
        summary='',  # Will be populated on first summary update
        episode_count=0,
        first_episode_date=episode_time,
        last_episode_date=episode_time,
        source_descriptions=source_descriptions,
        created_at=utc_now(),
    )

    return new_session


async def update_session_summary(
    llm_client: LLMClient,
    session_node: SessionNode,
    new_episode: EpisodicNode,
) -> SessionNode:
    """
    Update the session summary with a new episode.

    Uses incremental summarization if the session already has a summary,
    or creates an initial summary for the first episode.

    Parameters
    ----------
    llm_client : LLMClient
        The LLM client for generating summaries
    session_node : SessionNode
        The session node to update
    new_episode : EpisodicNode
        The new episode to incorporate into the summary

    Returns
    -------
    SessionNode
        Updated session node with new summary and metadata
    """
    # Update episode count
    session_node.episode_count += 1

    # Update last episode date
    if new_episode.valid_at > session_node.last_episode_date:
        session_node.last_episode_date = new_episode.valid_at

    # Update first episode date if this is earlier
    if new_episode.valid_at < session_node.first_episode_date:
        session_node.first_episode_date = new_episode.valid_at

    # Add source description if not already present
    if (
        new_episode.source_description
        and new_episode.source_description not in session_node.source_descriptions
    ):
        session_node.source_descriptions.append(new_episode.source_description)

    # Generate or update summary
    try:
        if session_node.episode_count == 1 or not session_node.summary:
            # First episode - create initial summary
            logger.debug(f'Creating initial summary for session: {session_node.session_id}')

            prompt_context = {
                'episode_content': new_episode.content,
                'source_description': new_episode.source_description,
                'session_id': session_node.session_id,
            }

            summary_response = await llm_client.generate_response(
                create_initial_session_summary(prompt_context),
                response_model=SessionSummary,
                prompt_name='summarize_sessions.create_initial',
            )

            session_node.summary = summary_response.get('summary', '')
            logger.debug(
                f'Created initial summary for session {session_node.session_id}: {session_node.summary}'
            )

        else:
            # Subsequent episodes - update existing summary
            logger.debug(
                f'Updating summary for session: {session_node.session_id} (episode {session_node.episode_count})'
            )

            prompt_context = {
                'previous_summary': session_node.summary,
                'new_episode_content': new_episode.content,
                'episode_count': session_node.episode_count - 1,  # Count before this episode
                'session_id': session_node.session_id,
            }

            summary_response = await llm_client.generate_response(
                summarize_session_incremental(prompt_context),
                response_model=SessionSummary,
                prompt_name='summarize_sessions.incremental',
            )

            session_node.summary = summary_response.get('summary', '')
            logger.debug(
                f'Updated summary for session {session_node.session_id}: {session_node.summary}'
            )

    except Exception as e:
        logger.error(
            f'Failed to generate/update summary for session {session_node.session_id}: {type(e).__name__}: {e}'
        )
        # Keep previous summary (or empty string if first episode)
        # Don't block episode ingestion due to summarization failure

    return session_node


async def link_episode_to_session(
    driver: GraphDriver,
    episode_uuid: str,
    session_uuid: str,
) -> None:
    """
    Create a relationship between an episode and its session.

    Creates a (Episode)-[:IN_SESSION]->(Session) relationship in the graph.

    Parameters
    ----------
    driver : GraphDriver
        The graph database driver
    episode_uuid : str
        UUID of the episode
    session_uuid : str
        UUID of the session node

    Returns
    -------
    None
    """
    query = """
        MATCH (e:Episodic {uuid: $episode_uuid})
        MATCH (s:Session {uuid: $session_uuid})
        MERGE (e)-[:IN_SESSION]->(s)
        RETURN e.uuid AS episode_uuid, s.uuid AS session_uuid
    """

    try:
        result = await driver.execute_query(
            query,
            episode_uuid=episode_uuid,
            session_uuid=session_uuid,
        )
        logger.debug(
            f'Linked episode {episode_uuid} to session {session_uuid}: {len(result[0])} relationships created/verified'
        )
    except Exception as e:
        logger.error(
            f'Failed to link episode {episode_uuid} to session {session_uuid}: {type(e).__name__}: {e}'
        )
        # Don't raise - link failure shouldn't block episode ingestion
