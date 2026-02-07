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


def compress_intents_to_summary(intents: list[str]) -> str:
    """
    Compress a list of intents into a human-readable summary using template-based logic.

    Format: "Main intent. Also worked on topic1, topic2, and N other refinements"

    Parameters
    ----------
    intents : list[str]
        List of user intents/topics in chronological order

    Returns
    -------
    str
        Human-readable summary under 200 characters
    """
    if not intents:
        return ''

    # Clean up main intent (remove trailing periods/commas)
    main_intent = intents[0].rstrip('.,')

    if len(intents) == 1:
        return main_intent

    # Get additional intents (refinements)
    additional_intents = [intent.rstrip('.,') for intent in intents[1:]]

    if len(additional_intents) == 1:
        # One refinement - show it directly
        return f'{main_intent}. Also worked on {additional_intents[0]}'
    elif len(additional_intents) == 2:
        # Two refinements - show both
        return f'{main_intent}. Also worked on {additional_intents[0]} and {additional_intents[1]}'
    elif len(additional_intents) <= 4:
        # 3-4 refinements - list all with commas
        refinements_text = ', '.join(additional_intents[:-1]) + f', and {additional_intents[-1]}'
        return f'{main_intent}. Also worked on {refinements_text}'
    else:
        # 5+ refinements - show first 2-3 most significant and summarize
        # Take first 2 intents (likely most significant based on order)
        top_refinements = ', '.join(additional_intents[:2])
        remaining_count = len(additional_intents) - 2
        return f'{main_intent}. Also worked on {top_refinements}, and {remaining_count} other refinements'


async def get_or_create_session_node(
    driver: GraphDriver,
    group_id: str,
    session_id: str,
    episode_time: datetime,
    source_description: str | None = None,
    programmatic: bool = False,
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
    programmatic : bool
        Whether this session is programmatically-generated (automated, imported, background)
        or human-interactive. Defaults to False. Only used when creating a new session;
        ignored if session already exists.

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
        programmatic=programmatic,
    )

    return new_session


async def update_session_summary(
    llm_client: LLMClient,
    session_node: SessionNode,
    new_episode: EpisodicNode,
    embedder: EmbedderClient | None = None,
    session_llm_client: LLMClient | None = None,
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
    session_llm_client : LLMClient | None
        Optional dedicated LLM client for session summarization.
        If not provided, falls back to llm_client parameter.

    Returns
    -------
    SessionNode
        Updated session node with new summary and metadata
    """
    # Use session_llm_client if provided, otherwise fall back to llm_client
    summarization_client = session_llm_client or llm_client

    # DEBUG: Log which client is being used
    if session_llm_client and session_llm_client is not llm_client:
        logger.info(f'[DEBUG] Using DEDICATED session LLM for session {session_node.session_id}: base_url={summarization_client.config.base_url}, model={summarization_client.model}')
    else:
        logger.warning(f'[DEBUG] Using MAIN llm_client for session {session_node.session_id} (session_llm_client was None or same as main): base_url={summarization_client.config.base_url}, model={summarization_client.model}')

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
        # Check if this is the first user message (no intents yet)
        if not session_node.intents:
            # First user message - extract initial intent
            logger.debug(
                f'Extracting initial intent for session: {session_node.session_id} (episode {session_node.episode_count})'
            )

            prompt_context = {
                'user_message': message_content,
                'session_id': session_node.session_id,
            }

            summary_response = await summarization_client.generate_response(
                extract_initial_intent(prompt_context),
                response_model=SessionSummary,
                prompt_name='summarize_sessions.extract_initial_intent',
            )

            initial_intent = summary_response.get('summary', '')
            session_node.intents.append(initial_intent)
            session_node.summary = compress_intents_to_summary(session_node.intents)
            logger.debug(
                f'Extracted initial intent for session {session_node.session_id}: "{initial_intent}"'
            )
            logger.debug(
                f'Generated summary for session {session_node.session_id}: "{session_node.summary}"'
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

                intent_response = await summarization_client.generate_response(
                    detect_intent_change(intent_context),
                    response_model=IntentChange,
                    prompt_name='summarize_sessions.detect_intent_change',
                )

                if intent_response.get('changed', False):
                    # Append new intent to intents list
                    new_intent = intent_response.get('new_intent', '')
                    logger.debug(
                        f'Intent changed for session {session_node.session_id}, appending: "{new_intent}"'
                    )

                    session_node.intents.append(new_intent)
                    session_node.summary = compress_intents_to_summary(session_node.intents)
                    logger.debug(
                        f'Appended new intent for session {session_node.session_id} (total: {len(session_node.intents)})'
                    )
                    logger.debug(
                        f'Generated summary for session {session_node.session_id}: "{session_node.summary}"'
                    )
                else:
                    # LLM says same intent despite low similarity - no change needed
                    logger.debug(
                        f'LLM determined same intent for session {session_node.session_id}, no update needed'
                    )
            else:
                # Same intent (high similarity) - no change needed
                logger.debug(
                    f'High similarity ({similarity:.2f}) detected for session {session_node.session_id}, no update needed'
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
