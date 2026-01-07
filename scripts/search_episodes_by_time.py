#!/usr/bin/env python3
"""
Search for episodes in a specific time range.

Usage:
    python scripts/search_episodes_by_time.py --group-id dave-weaver --start "2026-01-07T15:53:00" --end "2026-01-07T16:50:00"
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


async def search_episodes(group_id: str, start_time: str, end_time: str):
    """
    Search for episodes in a time range.

    Parameters
    ----------
    group_id : str
        The group ID
    start_time : str
        Start time in ISO format
    end_time : str
        End time in ISO format
    """
    # Parse times
    start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
    end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))

    # Initialize FalkorDB driver
    host = 'falkordb.appkit.local'
    port = 6379
    database = group_id

    db_user = os.getenv('NEO4J_USER', 'default')
    db_password = os.getenv('NEO4J_PASSWORD', 'password')

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
        logger.info('=' * 80)
        logger.info(f'SEARCHING EPISODES: {start_dt} to {end_dt}')
        logger.info('=' * 80)

        # Search for episodes in time range
        query = """
            MATCH (e:Episodic {group_id: $group_id})
            WHERE e.valid_at >= $start_time
              AND e.valid_at < $end_time
            RETURN e.uuid AS uuid, e.name AS name, e.content AS content,
                   e.valid_at AS valid_at, e.source_description AS source_description
            ORDER BY e.valid_at ASC
        """

        results, _, _ = await driver.execute_query(
            query,
            group_id=group_id,
            start_time=start_dt,
            end_time=end_dt,
        )

        logger.info(f'Found {len(results)} episodes in time range')
        logger.info('')

        if not results:
            logger.info('No episodes found in this time range.')
            return

        for i, episode in enumerate(results, 1):
            logger.info(f'[{i}/{len(results)}] Episode at {episode["valid_at"]}')
            logger.info(f'  UUID: {episode["uuid"]}')
            logger.info(f'  Name: {episode["name"]}')
            logger.info(f'  Source: {episode["source_description"]}')

            # Parse content
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

            # Check if linked to session
            session_query = """
                MATCH (e:Episodic {uuid: $episode_uuid})-[:IN_SESSION]->(s:Session)
                RETURN s.session_id AS session_id, s.uuid AS session_uuid
            """
            session_results, _, _ = await driver.execute_query(
                session_query,
                episode_uuid=episode['uuid'],
            )

            if session_results:
                session = session_results[0]
                logger.info(f'  → Linked to session: {session["session_id"][:8]}...')
            else:
                logger.info('  → NOT LINKED (orphaned)')

            logger.info('')

    finally:
        await driver.close()


def main():
    parser = argparse.ArgumentParser(description='Search for episodes in a time range')
    parser.add_argument('--group-id', required=True, help='Group ID')
    parser.add_argument('--start', required=True, help='Start time (ISO format)')
    parser.add_argument('--end', required=True, help='End time (ISO format)')

    args = parser.parse_args()

    asyncio.run(search_episodes(group_id=args.group_id, start_time=args.start, end_time=args.end))


if __name__ == '__main__':
    main()
