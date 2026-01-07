#!/usr/bin/env python3
"""
Inspect a specific session's episodes to check for missing or detached content.

Usage:
    python scripts/inspect_session.py --group-id dave-weaver --session-id 558b2f3f
"""

import argparse
import asyncio
import logging
import os
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

from graphiti_core.driver.falkordb_driver import FalkorDriver

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


async def inspect_session(group_id: str, session_id_prefix: str):
    """
    Inspect a session and its episodes.

    Parameters
    ----------
    group_id : str
        The group ID
    session_id_prefix : str
        Prefix of the session_id to search for
    """
    # Initialize FalkorDB driver
    db_uri = 'falkordb://falkordb.appkit.local:6379'
    db_user = os.getenv('NEO4J_USER', 'default')
    db_password = os.getenv('NEO4J_PASSWORD', 'password')

    host = 'falkordb.appkit.local'
    port = 6379
    database = group_id

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

    try:
        # Find session by prefix
        query = """
            MATCH (s:Session {group_id: $group_id})
            WHERE s.session_id STARTS WITH $session_id_prefix
            RETURN s.uuid AS uuid, s.session_id AS session_id, s.summary AS summary,
                   s.episode_count AS episode_count,
                   s.first_episode_date AS first_episode_date,
                   s.last_episode_date AS last_episode_date
        """

        results, _, _ = await driver.execute_query(
            query,
            group_id=group_id,
            session_id_prefix=session_id_prefix,
        )

        if not results:
            logger.error(f'No session found with prefix: {session_id_prefix}')
            return

        session = results[0]
        logger.info('=' * 80)
        logger.info(f'SESSION: {session["session_id"][:8]}...')
        logger.info('=' * 80)
        logger.info(f'UUID: {session["uuid"]}')
        logger.info(f'Summary: "{session["summary"]}"')
        logger.info(f'Episode Count: {session["episode_count"]}')
        logger.info(f'First Episode: {session["first_episode_date"]}')
        logger.info(f'Last Episode: {session["last_episode_date"]}')
        logger.info('')

        # Get all episodes for this session
        episodes_query = """
            MATCH (e:Episodic)-[:IN_SESSION]->(s:Session {uuid: $session_uuid})
            RETURN e.uuid AS uuid, e.name AS name, e.content AS content,
                   e.valid_at AS valid_at, e.source_description AS source_description
            ORDER BY e.valid_at ASC
        """

        episode_results, _, _ = await driver.execute_query(
            episodes_query,
            session_uuid=session['uuid'],
        )

        logger.info(f'EPISODES ({len(episode_results)}):')
        logger.info('=' * 80)

        for i, episode in enumerate(episode_results, 1):
            logger.info(f'\n[{i}/{len(episode_results)}] Episode at {episode["valid_at"]}')
            logger.info(f'  UUID: {episode["uuid"]}')
            logger.info(f'  Name: {episode["name"]}')
            logger.info(f'  Source: {episode["source_description"]}')

            # Parse content to extract role and message
            import re
            content = episode['content']
            match = re.match(r'^\[(\w+)\]:\s*(.+)', content, re.DOTALL)
            if match:
                role = match.group(1)
                message = match.group(2)

                # Truncate long messages
                if len(message) > 200:
                    message = message[:200] + '...'

                logger.info(f'  Role: {role}')
                logger.info(f'  Message: {message}')
            else:
                logger.info(f'  Content: {content[:200]}...')

        # Check for orphaned episodes (episodes not linked to any session)
        logger.info('\n' + '=' * 80)
        logger.info('CHECKING FOR ORPHANED EPISODES')
        logger.info('=' * 80)

        orphan_query = """
            MATCH (e:Episodic {group_id: $group_id})
            WHERE NOT (e)-[:IN_SESSION]->(:Session)
            RETURN e.uuid AS uuid, e.name AS name, e.valid_at AS valid_at,
                   e.content AS content
            ORDER BY e.valid_at ASC
            LIMIT 20
        """

        orphan_results, _, _ = await driver.execute_query(
            orphan_query,
            group_id=group_id,
        )

        if orphan_results:
            logger.warning(f'Found {len(orphan_results)} orphaned episodes!')
            for i, episode in enumerate(orphan_results, 1):
                logger.info(f'\n[{i}] Orphaned episode at {episode["valid_at"]}')
                logger.info(f'  UUID: {episode["uuid"]}')
                logger.info(f'  Name: {episode["name"]}')

                # Parse content
                import re
                content = episode['content']
                match = re.match(r'^\[(\w+)\]:\s*(.+)', content, re.DOTALL)
                if match:
                    role = match.group(1)
                    message = match.group(2)[:200]
                    logger.info(f'  Role: {role}')
                    logger.info(f'  Message: {message}...')
        else:
            logger.info('No orphaned episodes found.')

    finally:
        await driver.close()


def main():
    parser = argparse.ArgumentParser(
        description='Inspect a session and check for missing/detached episodes'
    )
    parser.add_argument('--group-id', required=True, help='Group ID')
    parser.add_argument(
        '--session-id',
        required=True,
        help='Session ID prefix (e.g., 558b2f3f)',
    )

    args = parser.parse_args()

    asyncio.run(
        inspect_session(
            group_id=args.group_id,
            session_id_prefix=args.session_id,
        )
    )


if __name__ == '__main__':
    main()
