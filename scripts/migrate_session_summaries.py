#!/usr/bin/env python3
"""
Migrate existing session summaries to use the new intent-based approach.

This script recalculates session summaries using the new intent-based logic that:
- Only processes user messages (ignores assistant/system)
- Extracts initial intent from first user message
- Uses embeddings to detect intent changes
- Preserves original intent and appends new topics

Usage:
    python scripts/migrate_session_summaries.py --group-id dave-weaver --date today
    python scripts/migrate_session_summaries.py --group-id dave-weaver --date 2024-01-07
    python scripts/migrate_session_summaries.py --group-id dave-weaver --all
"""

import argparse
import asyncio
import logging
import os
from datetime import datetime, timedelta

from dotenv import load_dotenv

load_dotenv()

from graphiti_core.driver.driver import GraphDriver
from graphiti_core.driver.falkordb_driver import FalkorDriver
from graphiti_core.driver.neo4j_driver import Neo4jDriver
from graphiti_core.embedder.openai import OpenAIEmbedder
from graphiti_core.graphiti import extract_message_content, extract_role_type
from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.nodes import SessionNode
from graphiti_core.prompts.summarize_sessions import (
    IntentChange,
    SessionSummary,
    append_new_intent,
    detect_intent_change,
    extract_initial_intent,
    refine_existing_intent,
)
from graphiti_core.search.search_utils import calculate_cosine_similarity
from graphiti_core.utils.datetime_utils import utc_now

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


async def get_sessions_for_date(
    driver: GraphDriver,
    group_id: str,
    start_date: datetime,
    end_date: datetime,
) -> list[SessionNode]:
    """Get all sessions with episodes in the date range."""
    query = """
        MATCH (s:Session {group_id: $group_id})
        WHERE s.first_episode_date >= $start_date
          AND s.first_episode_date < $end_date
        RETURN s.uuid AS uuid
        ORDER BY s.first_episode_date
    """

    results, _, _ = await driver.execute_query(
        query,
        group_id=group_id,
        start_date=start_date,
        end_date=end_date,
    )

    sessions = []
    for record in results:
        session = await SessionNode.get_by_uuid(driver, record['uuid'])
        if session:
            sessions.append(session)

    return sessions


async def get_session_episodes(driver: GraphDriver, session_uuid: str) -> list[dict]:
    """Get all episodes for a session in chronological order."""
    query = """
        MATCH (e:Episodic)-[:IN_SESSION]->(s:Session {uuid: $session_uuid})
        RETURN e.uuid AS uuid, e.content AS content, e.valid_at AS valid_at
        ORDER BY e.valid_at ASC
    """

    results, _, _ = await driver.execute_query(query, session_uuid=session_uuid)
    return [dict(record) for record in results]


async def recalculate_session_summary(
    session: SessionNode,
    episodes: list[dict],
    llm_client: OpenAIClient,
    embedder: OpenAIEmbedder,
) -> str:
    """
    Recalculate session summary using intent-based approach.

    Only processes user messages in chronological order.
    Returns the new summary.
    """
    new_summary = ''

    # Process episodes in order
    for episode in episodes:
        content = episode['content']

        # Extract role and message
        role_type = extract_role_type(content)
        message_content = extract_message_content(content)

        # Skip non-user messages
        if role_type != 'user':
            continue

        try:
            if not new_summary:
                # First user message - extract initial intent
                logger.debug(f'Extracting initial intent from: {message_content[:50]}...')

                prompt_context = {
                    'user_message': message_content,
                    'session_id': session.session_id,
                }

                response = await llm_client.generate_response(
                    extract_initial_intent(prompt_context),
                    response_model=SessionSummary,
                    prompt_name='migrate.extract_initial_intent',
                )

                new_summary = response.get('summary', '')
                logger.debug(f'Initial intent: "{new_summary}"')

            else:
                # Subsequent user message - check intent change
                logger.debug(f'Processing user message: {message_content[:50]}...')

                # Compute similarity
                current_embedding = await embedder.create(new_summary)
                new_embedding = await embedder.create(message_content)
                similarity = calculate_cosine_similarity(current_embedding, new_embedding)
                logger.debug(f'Similarity: {similarity:.2f}')

                if similarity < 0.7:
                    # Possible intent change - check with LLM
                    intent_context = {
                        'current_summary': new_summary,
                        'new_message': message_content,
                        'similarity': similarity,
                        'session_id': session.session_id,
                    }

                    intent_response = await llm_client.generate_response(
                        detect_intent_change(intent_context),
                        response_model=IntentChange,
                        prompt_name='migrate.detect_intent_change',
                    )

                    if intent_response.get('changed', False):
                        # Append new intent
                        new_intent = intent_response.get('new_intent', '')
                        logger.debug(f'Intent changed, appending: "{new_intent}"')

                        append_context = {
                            'current_summary': new_summary,
                            'new_intent': new_intent,
                            'session_id': session.session_id,
                        }

                        response = await llm_client.generate_response(
                            append_new_intent(append_context),
                            response_model=SessionSummary,
                            prompt_name='migrate.append_new_intent',
                        )

                        new_summary = response.get('summary', '')
                        logger.debug(f'New summary: "{new_summary}"')
                    else:
                        # Same intent - refine
                        logger.debug('Same intent, refining')

                        refine_context = {
                            'current_summary': new_summary,
                            'new_message': message_content,
                            'session_id': session.session_id,
                        }

                        response = await llm_client.generate_response(
                            refine_existing_intent(refine_context),
                            response_model=SessionSummary,
                            prompt_name='migrate.refine_existing_intent',
                        )

                        new_summary = response.get('summary', '')
                        logger.debug(f'Refined summary: "{new_summary}"')
                else:
                    # High similarity - refine
                    logger.debug('High similarity, refining')

                    refine_context = {
                        'current_summary': new_summary,
                        'new_message': message_content,
                        'session_id': session.session_id,
                    }

                    response = await llm_client.generate_response(
                        refine_existing_intent(refine_context),
                        response_model=SessionSummary,
                        prompt_name='migrate.refine_existing_intent',
                    )

                    new_summary = response.get('summary', '')
                    logger.debug(f'Refined summary: "{new_summary}"')

        except Exception as e:
            logger.error(f'Error processing episode: {e}')
            # Continue with current summary
            continue

    return new_summary


async def migrate_sessions(
    group_id: str,
    date: str | None = None,
    all_sessions: bool = False,
    dry_run: bool = False,
):
    """
    Migrate session summaries to use intent-based approach.

    Parameters
    ----------
    group_id : str
        The group ID to migrate sessions for
    date : str | None
        Date to migrate (YYYY-MM-DD format or 'today')
    all_sessions : bool
        If True, migrate all sessions regardless of date
    dry_run : bool
        If True, show what would be changed without making changes
    """
    # Initialize clients
    db_uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    db_user = os.getenv('NEO4J_USER', 'neo4j')
    db_password = os.getenv('NEO4J_PASSWORD')

    if not db_password:
        raise ValueError('NEO4J_PASSWORD environment variable is required')

    # Detect driver type from URI scheme
    if db_uri.startswith('falkordb://'):
        # Parse FalkorDB URI
        parts = db_uri.replace('falkordb://', '').split(':')
        host = parts[0]
        port = int(parts[1]) if len(parts) > 1 else 6379
        # Use group_id as database name (matching server behavior)
        database = group_id
        logger.info(f'Using FalkorDB driver: {host}:{port}, database: {database}')

        # Create FalkorDB client with timeout configuration
        from falkordb.asyncio import FalkorDB
        falkor_client = FalkorDB(
            host=host,
            port=port,
            username=db_user if db_user != 'default' else None,
            password=db_password if db_password != 'password' else None,
            socket_timeout=30.0,
            socket_connect_timeout=10.0,
        )
        driver = FalkorDriver(host=host, port=port, database=database, falkor_db=falkor_client)
    else:
        logger.info(f'Using Neo4j driver: {db_uri}')
        driver = Neo4jDriver(uri=db_uri, user=db_user, password=db_password)

    # Initialize LLM client (replace host.docker.internal with actual host IP)
    # The actual host IP is 172.16.0.114, not PORTAINER_HOST_IP which is for extra_hosts DNS
    llm_base_url = os.getenv('OPENAI_BASE_URL', 'http://172.16.0.114:9004/v1').replace(
        'host.docker.internal', '172.16.0.114'
    )
    llm_model = os.getenv('OPENAI_MODEL', 'gpt-oss')
    llm_api_key = os.getenv('OPENAI_API_KEY', 'not-needed')

    from graphiti_core.llm_client.config import LLMConfig
    from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient

    llm_config = LLMConfig(api_key=llm_api_key, model=llm_model, base_url=llm_base_url)
    llm_client = OpenAIGenericClient(config=llm_config, max_tokens=16384)
    llm_client.client.timeout = float(os.getenv('LLM_TIMEOUT', '120.0'))
    logger.info(f'Using LLM: {llm_model} at {llm_base_url}')

    # Initialize embedder (replace host.docker.internal with actual host IP)
    from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig

    embedder_base_url = os.getenv('EMBEDDING_BASE_URL', 'http://172.16.0.114:9003/v1').replace(
        'host.docker.internal', '172.16.0.114'
    )
    embedder_config = OpenAIEmbedderConfig(
        api_key='not-needed',
        embedding_model=os.getenv('EMBEDDING_MODEL', 'nomic-embed-text-v1.5.Q8_0'),
        embedding_dim=int(os.getenv('EMBEDDING_DIM', '768')),
        base_url=embedder_base_url,
    )
    embedder = OpenAIEmbedder(config=embedder_config)
    embedder.client.timeout = float(os.getenv('EMBEDDING_TIMEOUT', '60.0'))
    logger.info(f'Using embedder: {embedder_config.embedding_model} at {embedder_base_url}')

    try:
        # Determine date range
        if all_sessions:
            start_date = datetime(2000, 1, 1)
            end_date = datetime(2100, 1, 1)
            logger.info(f'Migrating ALL sessions for group_id: {group_id}')
        else:
            if date == 'today' or date is None:
                start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            else:
                start_date = datetime.strptime(date, '%Y-%m-%d')

            end_date = start_date + timedelta(days=1)
            logger.info(
                f'Migrating sessions for group_id: {group_id}, date: {start_date.date()}'
            )

        # Get sessions
        sessions = await get_sessions_for_date(driver, group_id, start_date, end_date)
        logger.info(f'Found {len(sessions)} sessions to migrate')

        if not sessions:
            logger.info('No sessions found to migrate')
            return

        # Process each session
        migrated_count = 0
        unchanged_count = 0
        error_count = 0

        for i, session in enumerate(sessions, 1):
            logger.info(
                f'\n[{i}/{len(sessions)}] Processing session {session.session_id[:8]}...'
            )
            logger.info(f'  Current summary: "{session.summary}"')

            try:
                # Get episodes
                episodes = await get_session_episodes(driver, session.uuid)
                logger.info(f'  Episodes: {len(episodes)}')

                if not episodes:
                    logger.warning('  No episodes found, skipping')
                    unchanged_count += 1
                    continue

                # Recalculate summary
                new_summary = await recalculate_session_summary(
                    session, episodes, llm_client, embedder
                )

                if new_summary == session.summary:
                    logger.info(f'  Summary unchanged: "{new_summary}"')
                    unchanged_count += 1
                    continue

                logger.info(f'  New summary: "{new_summary}"')

                if not dry_run:
                    # Update session in database
                    session.summary = new_summary
                    await session.save(driver)
                    logger.info('  ✓ Updated')
                    migrated_count += 1
                else:
                    logger.info('  [DRY RUN] Would update')
                    migrated_count += 1

            except Exception as e:
                logger.error(f'  Error migrating session: {e}')
                error_count += 1
                continue

        # Summary
        logger.info('\n' + '=' * 80)
        logger.info('MIGRATION COMPLETE')
        logger.info('=' * 80)
        logger.info(f'Total sessions: {len(sessions)}')
        logger.info(f'Migrated: {migrated_count}')
        logger.info(f'Unchanged: {unchanged_count}')
        logger.info(f'Errors: {error_count}')
        if dry_run:
            logger.info('\n[DRY RUN] No changes were made to the database')

    finally:
        await driver.close()


def main():
    parser = argparse.ArgumentParser(
        description='Migrate session summaries to intent-based approach'
    )
    parser.add_argument('--group-id', required=True, help='Group ID to migrate sessions for')
    parser.add_argument(
        '--date',
        help='Date to migrate (YYYY-MM-DD or "today", default: today)',
        default='today',
    )
    parser.add_argument(
        '--all', action='store_true', help='Migrate all sessions regardless of date'
    )
    parser.add_argument(
        '--dry-run', action='store_true', help='Show what would change without making changes'
    )

    args = parser.parse_args()

    asyncio.run(
        migrate_sessions(
            group_id=args.group_id,
            date=None if args.all else args.date,
            all_sessions=args.all,
            dry_run=args.dry_run,
        )
    )


if __name__ == '__main__':
    main()
