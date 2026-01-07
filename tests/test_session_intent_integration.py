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

from datetime import datetime
from unittest.mock import AsyncMock, Mock

import pytest

from graphiti_core.nodes import EpisodeType, EpisodicNode, SessionNode
from graphiti_core.utils.datetime_utils import utc_now
from graphiti_core.utils.maintenance.session_operations import update_session_summary


@pytest.fixture
def mock_llm_client():
    """Mock LLM client that returns reasonable responses."""
    client = Mock()

    async def mock_generate_response(messages, response_model, prompt_name):
        # Determine response based on prompt_name
        if 'extract_initial_intent' in prompt_name:
            return {'summary': 'User asked about Graphiti session summarization'}
        elif 'detect_intent_change' in prompt_name:
            # Simulate intent change detection based on message content
            message = messages[1].content
            if 'contactcenter' in message.lower():
                return {'changed': True, 'new_intent': 'the contactcenter repository'}
            else:
                return {'changed': False, 'new_intent': ''}
        elif 'append_new_intent' in prompt_name:
            return {
                'summary': 'User asked about Graphiti session summarization. User also asked about contactcenter repo.'
            }
        elif 'refine_existing_intent' in prompt_name:
            # Simulate refinement
            return {
                'summary': 'User asked about Graphiti session summarization, then drilled into intent-based weighting'
            }
        else:
            return {'summary': 'Default summary'}

    client.generate_response = AsyncMock(side_effect=mock_generate_response)
    return client


@pytest.fixture
def mock_embedder():
    """Mock embedder that returns similarity based on content."""
    embedder = Mock()

    async def mock_create(text):
        # Create consistent embeddings based on keywords
        if 'graphiti' in text.lower() or 'summarization' in text.lower():
            return [0.1, 0.8, 0.2, 0.5] * 256  # 1024 dims
        elif 'contactcenter' in text.lower() or 'repository' in text.lower():
            return [0.9, 0.1, 0.7, 0.3] * 256
        else:
            return [0.5, 0.5, 0.5, 0.5] * 256

    embedder.create = AsyncMock(side_effect=mock_create)
    return embedder


@pytest.mark.asyncio
async def test_first_user_message_extracts_intent(mock_llm_client, mock_embedder):
    """Test that first user message extracts initial intent."""
    session_node = SessionNode(
        name='Test Session',
        session_id='test-123',
        group_id='test-group',
        summary='',
        episode_count=0,
        first_episode_date=utc_now(),
        last_episode_date=utc_now(),
        source_descriptions=[],
        created_at=utc_now(),
    )

    episode = EpisodicNode(
        name='First message',
        group_id='test-group',
        content='[user]: Can you explain how session summarization works in Graphiti?',
        source=EpisodeType.message,
        source_description='test',
        valid_at=utc_now(),
        created_at=utc_now(),
    )

    result = await update_session_summary(mock_llm_client, session_node, episode, mock_embedder)

    assert result.episode_count == 1
    assert result.summary == 'User asked about Graphiti session summarization'
    mock_llm_client.generate_response.assert_called_once()
    call_args = mock_llm_client.generate_response.call_args
    assert 'extract_initial_intent' in call_args[1]['prompt_name']


@pytest.mark.asyncio
async def test_assistant_message_skipped(mock_llm_client, mock_embedder):
    """Test that assistant messages don't update summary."""
    session_node = SessionNode(
        name='Test Session',
        session_id='test-123',
        group_id='test-group',
        summary='User asked about authentication',
        episode_count=1,
        first_episode_date=utc_now(),
        last_episode_date=utc_now(),
        source_descriptions=[],
        created_at=utc_now(),
    )

    episode = EpisodicNode(
        name='Assistant response',
        group_id='test-group',
        content='[assistant]: Here is how authentication works...',
        source=EpisodeType.message,
        source_description='test',
        valid_at=utc_now(),
        created_at=utc_now(),
    )

    result = await update_session_summary(mock_llm_client, session_node, episode, mock_embedder)

    assert result.episode_count == 2
    assert result.summary == 'User asked about authentication'  # Unchanged
    mock_llm_client.generate_response.assert_not_called()  # No LLM call for assistant messages


@pytest.mark.asyncio
async def test_system_message_skipped(mock_llm_client, mock_embedder):
    """Test that system messages don't update summary."""
    session_node = SessionNode(
        name='Test Session',
        session_id='test-123',
        group_id='test-group',
        summary='User wants to debug error',
        episode_count=2,
        first_episode_date=utc_now(),
        last_episode_date=utc_now(),
        source_descriptions=[],
        created_at=utc_now(),
    )

    episode = EpisodicNode(
        name='System message',
        group_id='test-group',
        content='[system]: Session initialized',
        source=EpisodeType.message,
        source_description='test',
        valid_at=utc_now(),
        created_at=utc_now(),
    )

    result = await update_session_summary(mock_llm_client, session_node, episode, mock_embedder)

    assert result.episode_count == 3
    assert result.summary == 'User wants to debug error'  # Unchanged
    mock_llm_client.generate_response.assert_not_called()


@pytest.mark.asyncio
async def test_same_intent_high_similarity(mock_llm_client, mock_embedder):
    """Test that high similarity messages refine existing summary."""
    session_node = SessionNode(
        name='Test Session',
        session_id='test-123',
        group_id='test-group',
        summary='User asked about Graphiti session summarization',
        episode_count=2,
        first_episode_date=utc_now(),
        last_episode_date=utc_now(),
        source_descriptions=[],
        created_at=utc_now(),
    )

    episode = EpisodicNode(
        name='Follow-up',
        group_id='test-group',
        content='[user]: What about the intent-based weighting approach?',
        source=EpisodeType.message,
        source_description='test',
        valid_at=utc_now(),
        created_at=utc_now(),
    )

    result = await update_session_summary(mock_llm_client, session_node, episode, mock_embedder)

    assert result.episode_count == 3
    assert 'intent-based weighting' in result.summary
    # Should call refine_existing_intent
    mock_llm_client.generate_response.assert_called_once()
    call_args = mock_llm_client.generate_response.call_args
    assert 'refine_existing_intent' in call_args[1]['prompt_name']


@pytest.mark.asyncio
async def test_different_intent_low_similarity(mock_llm_client, mock_embedder):
    """Test that low similarity messages append new intent."""
    session_node = SessionNode(
        name='Test Session',
        session_id='test-123',
        group_id='test-group',
        summary='User asked about Graphiti session summarization',
        episode_count=3,
        first_episode_date=utc_now(),
        last_episode_date=utc_now(),
        source_descriptions=[],
        created_at=utc_now(),
    )

    episode = EpisodicNode(
        name='Topic change',
        group_id='test-group',
        content='[user]: Can you also look at the contactcenter repository?',
        source=EpisodeType.message,
        source_description='test',
        valid_at=utc_now(),
        created_at=utc_now(),
    )

    result = await update_session_summary(mock_llm_client, session_node, episode, mock_embedder)

    assert result.episode_count == 4
    assert 'contactcenter' in result.summary
    # Should call detect_intent_change then append_new_intent
    assert mock_llm_client.generate_response.call_count == 2
    call_names = [call[1]['prompt_name'] for call in mock_llm_client.generate_response.call_args_list]
    assert 'detect_intent_change' in call_names[0]
    assert 'append_new_intent' in call_names[1]


@pytest.mark.asyncio
async def test_no_embedder_defaults_to_intent_change(mock_llm_client):
    """Test that without embedder, similarity defaults to 0.0 (intent change path)."""
    session_node = SessionNode(
        name='Test Session',
        session_id='test-123',
        group_id='test-group',
        summary='User asked about authentication',
        episode_count=1,
        first_episode_date=utc_now(),
        last_episode_date=utc_now(),
        source_descriptions=[],
        created_at=utc_now(),
    )

    episode = EpisodicNode(
        name='Follow-up',
        group_id='test-group',
        content='[user]: What about OAuth specifically?',
        source=EpisodeType.message,
        source_description='test',
        valid_at=utc_now(),
        created_at=utc_now(),
    )

    result = await update_session_summary(mock_llm_client, session_node, episode, embedder=None)

    assert result.episode_count == 2
    # Without embedder, similarity=0.0, so it goes to detect_intent_change path
    mock_llm_client.generate_response.assert_called()


@pytest.mark.asyncio
async def test_metadata_updates(mock_llm_client, mock_embedder):
    """Test that episode metadata (count, dates) are updated correctly."""
    first_date = datetime(2024, 1, 1)
    last_date = datetime(2024, 1, 2)

    session_node = SessionNode(
        name='Test Session',
        session_id='test-123',
        group_id='test-group',
        summary='Existing summary',
        episode_count=5,
        first_episode_date=first_date,
        last_episode_date=last_date,
        source_descriptions=['source1'],
        created_at=utc_now(),
    )

    new_date = datetime(2024, 1, 3)
    episode = EpisodicNode(
        name='New message',
        group_id='test-group',
        content='[user]: Another question',
        source=EpisodeType.message,
        source_description='source2',
        valid_at=new_date,
        created_at=utc_now(),
    )

    result = await update_session_summary(mock_llm_client, session_node, episode, mock_embedder)

    assert result.episode_count == 6
    assert result.first_episode_date == first_date  # Unchanged
    assert result.last_episode_date == new_date  # Updated
    assert 'source2' in result.source_descriptions


@pytest.mark.asyncio
async def test_error_handling_preserves_summary(mock_llm_client, mock_embedder):
    """Test that LLM errors don't lose the summary."""
    session_node = SessionNode(
        name='Test Session',
        session_id='test-123',
        group_id='test-group',
        summary='Existing summary',
        episode_count=1,
        first_episode_date=utc_now(),
        last_episode_date=utc_now(),
        source_descriptions=[],
        created_at=utc_now(),
    )

    episode = EpisodicNode(
        name='New message',
        group_id='test-group',
        content='[user]: This will cause an error',
        source=EpisodeType.message,
        source_description='test',
        valid_at=utc_now(),
        created_at=utc_now(),
    )

    # Make LLM client raise exception
    mock_llm_client.generate_response = AsyncMock(side_effect=Exception('LLM error'))

    result = await update_session_summary(mock_llm_client, session_node, episode, mock_embedder)

    # Summary should be preserved despite error
    assert result.summary == 'Existing summary'
    assert result.episode_count == 2  # Metadata still updated


@pytest.mark.asyncio
async def test_embedding_error_fallback(mock_llm_client, mock_embedder):
    """Test that embedding errors don't break the flow."""
    session_node = SessionNode(
        name='Test Session',
        session_id='test-123',
        group_id='test-group',
        summary='Existing summary',
        episode_count=1,
        first_episode_date=utc_now(),
        last_episode_date=utc_now(),
        source_descriptions=[],
        created_at=utc_now(),
    )

    episode = EpisodicNode(
        name='New message',
        group_id='test-group',
        content='[user]: Test message',
        source=EpisodeType.message,
        source_description='test',
        valid_at=utc_now(),
        created_at=utc_now(),
    )

    # Make embedder raise exception
    mock_embedder.create = AsyncMock(side_effect=Exception('Embedding error'))

    result = await update_session_summary(mock_llm_client, session_node, episode, mock_embedder)

    # Should fall back to similarity=0.0 and continue
    assert result.episode_count == 2
    mock_llm_client.generate_response.assert_called()  # LLM still called


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
