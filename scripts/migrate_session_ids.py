"""
Migration script to backfill session_id field for existing Episodic nodes.

This script extracts session UUIDs from the source_description field of existing
episodes and populates the new session_id field. It also removes the session
information from source_description to avoid redundancy.

Usage:
    # Dry run to preview changes
    uv run python scripts/migrate_session_ids.py --group-id dave-weaver --dry-run

    # Execute the migration
    uv run python scripts/migrate_session_ids.py --group-id dave-weaver

    # Migrate all episodes (no group filter)
    uv run python scripts/migrate_session_ids.py --all --dry-run
"""

import argparse
import asyncio
import logging
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from graphiti_core import Graphiti
from graphiti_core.embedder import OpenAIEmbedder
from graphiti_core.llm_client import LLMConfig, OpenAIClient
from graphiti_core.utils.session_utils import extract_session_id

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def migrate_episodes(group_id: str | None = None, dry_run: bool = True):
    """
    Migrate existing episodes to extract session_id from source_description.

    Args:
        group_id: Optional group_id to filter episodes. If None, migrates all.
        dry_run: If True, only preview changes without applying them.
    """
    # Initialize Graphiti (using environment variables for config)
    llm_config = LLMConfig(
        api_key=os.getenv('OPENAI_API_KEY'),
        model='gpt-4o-mini',
        temperature=0.0,
    )
    llm_client = OpenAIClient(llm_config)
    embedder = OpenAIEmbedder()

    try:
        graphiti = Graphiti(
            neo4j_uri=os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
            neo4j_user=os.getenv('NEO4J_USER', 'neo4j'),
            neo4j_password=os.getenv('NEO4J_PASSWORD', 'password'),
            llm_client=llm_client,
            embedder=embedder,
        )
        logger.info('Connected to Neo4j database')
    except Exception as e:
        logger.error(f'Failed to connect to database: {e}')
        return

    # Build query to find episodes with session info in source_description
    where_clauses = [
        "e.source_description =~ '(?i).*\\\\(session:.*\\\\).*'",
        '(e.session_id IS NULL OR e.session_id = "")',
    ]

    if group_id:
        where_clauses.append('e.group_id = $group_id')

    where_query = ' AND '.join(where_clauses)

    query = f"""
        MATCH (e:Episodic)
        WHERE {where_query}
        RETURN e.uuid AS uuid,
               e.source_description AS source_description
    """

    query_params = {'group_id': group_id} if group_id else {}

    try:
        result, _, _ = await graphiti.driver.execute_query(query, **query_params)
    except Exception as e:
        logger.error(f'Failed to query episodes: {e}')
        return

    episodes_to_update = []
    for record in result:
        uuid = record['uuid']
        source_desc = record['source_description']

        # Extract session_id and clean description
        session_id, cleaned_desc = extract_session_id(source_desc)

        if session_id:
            episodes_to_update.append(
                {
                    'uuid': uuid,
                    'session_id': session_id,
                    'source_description': cleaned_desc,
                }
            )

    logger.info(f'Found {len(episodes_to_update)} episodes to migrate')

    if len(episodes_to_update) == 0:
        logger.info('No episodes need migration. Exiting.')
        return

    if dry_run:
        logger.info('=' * 60)
        logger.info('DRY RUN - No changes will be made')
        logger.info('=' * 60)
        preview_count = min(5, len(episodes_to_update))
        logger.info(f'Showing first {preview_count} episodes:')
        logger.info('')
        for i, ep in enumerate(episodes_to_update[:preview_count], 1):
            logger.info(f'Episode {i}:')
            logger.info(f'  UUID: {ep["uuid"]}')
            logger.info(f'  Session ID: {ep["session_id"]}')
            logger.info(f'  New description: {ep["source_description"][:80]}...')
            logger.info('')
        if len(episodes_to_update) > preview_count:
            logger.info(f'... and {len(episodes_to_update) - preview_count} more episodes')
        return

    # Perform bulk update
    logger.info('Starting migration...')
    update_query = """
        UNWIND $episodes AS ep
        MATCH (e:Episodic {uuid: ep.uuid})
        SET e.session_id = ep.session_id,
            e.source_description = ep.source_description
    """

    try:
        await graphiti.driver.execute_query(update_query, episodes=episodes_to_update)
        logger.info(f'✓ Successfully migrated {len(episodes_to_update)} episodes')
    except Exception as e:
        logger.error(f'✗ Migration failed: {e}')
        return

    # Verify migration
    verify_query = """
        MATCH (e:Episodic)
        WHERE e.session_id IS NOT NULL
    """
    if group_id:
        verify_query += ' AND e.group_id = $group_id'
    verify_query += ' RETURN count(e) AS count'

    try:
        result, _, _ = await graphiti.driver.execute_query(
            verify_query, **({'group_id': group_id} if group_id else {})
        )
        count = result[0]['count']
        logger.info(f'Verification: {count} episodes now have session_id')
    except Exception as e:
        logger.warning(f'Verification failed: {e}')


def main():
    parser = argparse.ArgumentParser(description='Migrate session IDs for Episodic nodes')
    parser.add_argument('--group-id', type=str, help='Group ID to filter episodes (optional)')
    parser.add_argument(
        '--all', action='store_true', help='Migrate all episodes regardless of group_id'
    )
    parser.add_argument(
        '--dry-run', action='store_true', help='Preview changes without applying them'
    )

    args = parser.parse_args()

    if not args.group_id and not args.all:
        parser.error('Must specify either --group-id or --all')

    group_id = args.group_id if not args.all else None

    asyncio.run(migrate_episodes(group_id=group_id, dry_run=args.dry_run))


if __name__ == '__main__':
    main()
