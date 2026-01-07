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
from graphiti_core.embedder.client import EmbedderClient
from graphiti_core.graphiti import extract_message_content, extract_role_type
from graphiti_core.llm_client import LLMClient
from graphiti_core.nodes import EpisodicNode, SessionNode
from graphiti_core.prompts.summarize_sessions import (
    IntentChange,
    SessionSummary,
    append_new_intent,
    create_initial_session_summary,
    detect_intent_change,
    extract_initial_intent,
    refine_existing_intent,
    summarize_session_incremental,
)
from graphiti_core.search.search_utils import calculate_cosine_similarity
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
    embedder: EmbedderClient | None = None,
) -> SessionNode:
    """
    Update the session summary with a new episode using intent-based approach.

    Only processes user messages. Assistant and system messages are ignored entirely.
    For user messages:
    - First user message: Extract initial intent (50-100 chars)
    - Subsequent user messages: Check intent change via embeddings
      - Same intent (similarity >= 0.7): Refine existing summary
      - Different intent (similarity < 0.7): Append new intent

    Parameters
    ----------
    llm_client : LLMClient
        The LLM client for generating summaries
    session_node : SessionNode
        The session node to update
    new_episode : EpisodicNode
        The new episode to incorporate into the summary
    embedder : EmbedderClient | None
        The embedder client for intent change detection (optional)

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

    # Extract role type and message content from episode
    role_type = extract_role_type(new_episode.content)
    message_content = extract_message_content(new_episode.content)

    # Only process user messages - skip assistant and system messages
    if role_type != 'user':
        logger.debug(
            f'Skipping summary update for {role_type} message in session {session_node.session_id}'
        )
        return session_node

    logger.debug(f'Processing user message for session {session_node.session_id}')

    # Generate or update summary
    try:
        # Check if this is the first user message (summary is empty)
        if not session_node.summary:
            # First user message - extract initial intent
            logger.debug(
                f'Extracting initial intent for session: {session_node.session_id} (episode {session_node.episode_count})'
            )

            prompt_context = {
                'user_message': message_content,
                'session_id': session_node.session_id,
            }

            summary_response = await llm_client.generate_response(
                extract_initial_intent(prompt_context),
                response_model=SessionSummary,
                prompt_name='summarize_sessions.extract_initial_intent',
            )

            session_node.summary = summary_response.get('summary', '')
            logger.debug(
                f'Extracted initial intent for session {session_node.session_id}: "{session_node.summary}"'
            )

        else:
            # Subsequent user messages - check for intent change
            logger.debug(
                f'Checking intent change for session: {session_node.session_id} (episode {session_node.episode_count})'
            )

            # Use embeddings to detect intent change if embedder available
            similarity = 0.0
            if embedder:
                try:
                    current_embedding = await embedder.create(session_node.summary)
                    new_embedding = await embedder.create(message_content)
                    similarity = calculate_cosine_similarity(current_embedding, new_embedding)
                    logger.debug(
                        f'Intent similarity for session {session_node.session_id}: {similarity:.2f}'
                    )
                except Exception as e:
                    logger.warning(
                        f'Failed to compute embedding similarity for session {session_node.session_id}: {e}. Defaulting to 0.0'
                    )
                    similarity = 0.0
            else:
                logger.debug(
                    f'No embedder provided for session {session_node.session_id}, defaulting similarity to 0.0'
                )

            # Intent change threshold: < 0.7 suggests different intent
            if similarity < 0.7:
                # Different intent detected - use LLM to confirm and extract new intent
                logger.debug(
                    f'Low similarity ({similarity:.2f}) detected for session {session_node.session_id}, checking intent change'
                )

                intent_context = {
                    'current_summary': session_node.summary,
                    'new_message': message_content,
                    'similarity': similarity,
                    'session_id': session_node.session_id,
                }

                intent_response = await llm_client.generate_response(
                    detect_intent_change(intent_context),
                    response_model=IntentChange,
                    prompt_name='summarize_sessions.detect_intent_change',
                )

                if intent_response.get('changed', False):
                    # Append new intent to summary
                    new_intent = intent_response.get('new_intent', '')
                    logger.debug(
                        f'Intent changed for session {session_node.session_id}, appending: "{new_intent}"'
                    )

                    append_context = {
                        'current_summary': session_node.summary,
                        'new_intent': new_intent,
                        'session_id': session_node.session_id,
                    }

                    summary_response = await llm_client.generate_response(
                        append_new_intent(append_context),
                        response_model=SessionSummary,
                        prompt_name='summarize_sessions.append_new_intent',
                    )

                    session_node.summary = summary_response.get('summary', '')
                    logger.debug(
                        f'Appended new intent for session {session_node.session_id}: "{session_node.summary}"'
                    )
                else:
                    # LLM says same intent despite low similarity - refine
                    logger.debug(
                        f'LLM determined same intent for session {session_node.session_id}, refining'
                    )

                    refine_context = {
                        'current_summary': session_node.summary,
                        'new_message': message_content,
                        'session_id': session_node.session_id,
                    }

                    summary_response = await llm_client.generate_response(
                        refine_existing_intent(refine_context),
                        response_model=SessionSummary,
                        prompt_name='summarize_sessions.refine_existing_intent',
                    )

                    session_node.summary = summary_response.get('summary', '')
                    logger.debug(
                        f'Refined intent for session {session_node.session_id}: "{session_node.summary}"'
                    )
            else:
                # Same intent (high similarity) - refine existing summary
                logger.debug(
                    f'High similarity ({similarity:.2f}) detected for session {session_node.session_id}, refining summary'
                )

                refine_context = {
                    'current_summary': session_node.summary,
                    'new_message': message_content,
                    'session_id': session_node.session_id,
                }

                summary_response = await llm_client.generate_response(
                    refine_existing_intent(refine_context),
                    response_model=SessionSummary,
                    prompt_name='summarize_sessions.refine_existing_intent',
                )

                session_node.summary = summary_response.get('summary', '')
                logger.debug(
                    f'Refined intent for session {session_node.session_id}: "{session_node.summary}"'
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
