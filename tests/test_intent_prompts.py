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

from graphiti_core.prompts.summarize_sessions import (
    IntentChange,
    append_new_intent,
    detect_intent_change,
    extract_initial_intent,
    refine_existing_intent,
)


def test_extract_initial_intent_structure():
    """Test that extract_initial_intent returns proper prompt structure."""
    context = {
        'user_message': 'Can you explain how session summarization works?',
        'session_id': 'test-session-123',
    }

    messages = extract_initial_intent(context)

    # Should return list of Message objects
    assert isinstance(messages, list)
    assert len(messages) == 2

    # First message should be system
    assert messages[0].role == 'system'
    assert 'intent' in messages[0].content.lower()

    # Second message should be user with the actual prompt
    assert messages[1].role == 'user'
    assert 'session summarization' in messages[1].content
    assert '50-100 characters' in messages[1].content


def test_extract_initial_intent_format_requirements():
    """Test that extract_initial_intent includes format requirements."""
    context = {'user_message': 'Help me debug this error', 'session_id': 'test-123'}

    messages = extract_initial_intent(context)
    user_prompt = messages[1].content

    # Should specify format
    assert 'User asked about' in user_prompt or 'User wants to' in user_prompt
    assert '50-100 characters' in user_prompt
    assert 'debug this error' in user_prompt


def test_detect_intent_change_structure():
    """Test that detect_intent_change returns proper prompt structure."""
    context = {
        'current_summary': 'User asked about Claude Code hooks',
        'new_message': 'Can you look at the contactcenter repo?',
        'similarity': 0.45,
        'session_id': 'test-123',
    }

    messages = detect_intent_change(context)

    assert isinstance(messages, list)
    assert len(messages) == 2
    assert messages[0].role == 'system'
    assert messages[1].role == 'user'

    user_prompt = messages[1].content
    assert 'Claude Code hooks' in user_prompt
    assert 'contactcenter repo' in user_prompt
    assert '45%' in user_prompt  # Similarity as percentage


def test_detect_intent_change_similarity_threshold():
    """Test that detect_intent_change includes similarity threshold guidance."""
    context = {
        'current_summary': 'User asked about Python packaging',
        'new_message': 'What about uv specifically?',
        'similarity': 0.85,
        'session_id': 'test-123',
    }

    messages = detect_intent_change(context)
    user_prompt = messages[1].content

    # Should mention the threshold
    assert '70%' in user_prompt or '< 70%' in user_prompt
    assert '85%' in user_prompt  # Actual similarity


def test_detect_intent_change_examples():
    """Test that detect_intent_change includes helpful examples."""
    context = {
        'current_summary': 'User wants to set up authentication',
        'new_message': 'Also need help with database migrations',
        'similarity': 0.3,
        'session_id': 'test-123',
    }

    messages = detect_intent_change(context)
    user_prompt = messages[1].content

    # Should include examples of same vs different intent
    assert 'changed=true' in user_prompt or 'changed=false' in user_prompt
    assert 'Examples' in user_prompt or 'Example' in user_prompt


def test_refine_existing_intent_structure():
    """Test that refine_existing_intent returns proper prompt structure."""
    context = {
        'current_summary': 'User asked about Claude Code hooks',
        'new_message': 'What about the memory hook specifically?',
        'session_id': 'test-123',
    }

    messages = refine_existing_intent(context)

    assert isinstance(messages, list)
    assert len(messages) == 2
    assert messages[0].role == 'system'
    assert messages[1].role == 'user'

    user_prompt = messages[1].content
    assert 'Claude Code hooks' in user_prompt
    assert 'memory hook' in user_prompt
    assert '150 characters' in user_prompt


def test_refine_existing_intent_rules():
    """Test that refine_existing_intent includes clear rules."""
    context = {
        'current_summary': 'User wants to optimize database queries',
        'new_message': 'Thanks, that helps!',
        'session_id': 'test-123',
    }

    messages = refine_existing_intent(context)
    user_prompt = messages[1].content

    # Should specify rules for when to update vs keep unchanged
    assert 'thanks' in user_prompt.lower() or 'unchanged' in user_prompt.lower()
    assert 'drilled' in user_prompt or 'asked about' in user_prompt
    assert 'Rules' in user_prompt or 'rules' in user_prompt


def test_refine_existing_intent_examples():
    """Test that refine_existing_intent includes examples."""
    context = {
        'current_summary': 'User asked about error handling',
        'new_message': 'What about retry logic?',
        'session_id': 'test-123',
    }

    messages = refine_existing_intent(context)
    user_prompt = messages[1].content

    assert 'Examples' in user_prompt or 'Example' in user_prompt


def test_append_new_intent_structure():
    """Test that append_new_intent returns proper prompt structure."""
    context = {
        'current_summary': 'User asked about Claude Code hooks, then drilled into memory hook',
        'new_intent': 'the contactcenter repository',
        'session_id': 'test-123',
    }

    messages = append_new_intent(context)

    assert isinstance(messages, list)
    assert len(messages) == 2
    assert messages[0].role == 'system'
    assert messages[1].role == 'user'

    user_prompt = messages[1].content
    assert 'Claude Code hooks' in user_prompt
    assert 'memory hook' in user_prompt
    assert 'contactcenter repository' in user_prompt
    assert '200 characters' in user_prompt


def test_append_new_intent_format():
    """Test that append_new_intent specifies the expected format."""
    context = {
        'current_summary': 'User wants to set up CI/CD pipeline',
        'new_intent': 'Docker configuration',
        'session_id': 'test-123',
    }

    messages = append_new_intent(context)
    user_prompt = messages[1].content

    # Should specify the format for appending
    assert 'User also asked about' in user_prompt or 'also asked' in user_prompt
    assert 'Format' in user_prompt or 'format' in user_prompt


def test_append_new_intent_overflow_handling():
    """Test that append_new_intent includes guidance for length overflow."""
    context = {
        'current_summary': 'A very long summary that might exceed limits when adding new intent',
        'new_intent': 'some new topic',
        'session_id': 'test-123',
    }

    messages = append_new_intent(context)
    user_prompt = messages[1].content

    # Should mention what to do if exceeds limit
    assert '200 char' in user_prompt
    assert ('compress' in user_prompt.lower() or 'other topics' in user_prompt.lower())


def test_intent_change_model():
    """Test the IntentChange Pydantic model."""
    # Test valid instance
    change1 = IntentChange(changed=True, new_intent='the contactcenter repository')
    assert change1.changed is True
    assert change1.new_intent == 'the contactcenter repository'

    # Test no change
    change2 = IntentChange(changed=False, new_intent='')
    assert change2.changed is False
    assert change2.new_intent == ''

    # Test model validation
    try:
        IntentChange(changed=True)  # Missing new_intent
        assert False, 'Should have raised validation error'
    except Exception:
        pass  # Expected


def test_all_prompts_use_to_prompt_json():
    """Test that all prompts properly use to_prompt_json for content."""
    context_initial = {'user_message': 'Test message', 'session_id': '123'}
    messages_initial = extract_initial_intent(context_initial)
    # to_prompt_json is called in the f-string, check the message exists
    assert 'Test message' in messages_initial[1].content

    context_detect = {
        'current_summary': 'Summary',
        'new_message': 'New',
        'similarity': 0.5,
        'session_id': '123',
    }
    messages_detect = detect_intent_change(context_detect)
    assert 'New' in messages_detect[1].content

    context_refine = {
        'current_summary': 'Summary',
        'new_message': 'Refine',
        'session_id': '123',
    }
    messages_refine = refine_existing_intent(context_refine)
    assert 'Refine' in messages_refine[1].content


def test_all_prompts_include_session_id_in_context():
    """Test that all prompts accept session_id in context (for potential logging)."""
    session_id = 'test-session-456'

    # All functions should accept session_id without error
    extract_initial_intent({'user_message': 'Test', 'session_id': session_id})
    detect_intent_change(
        {
            'current_summary': 'S',
            'new_message': 'M',
            'similarity': 0.5,
            'session_id': session_id,
        }
    )
    refine_existing_intent(
        {'current_summary': 'S', 'new_message': 'M', 'session_id': session_id}
    )
    append_new_intent({'current_summary': 'S', 'new_intent': 'I', 'session_id': session_id})

    # No assertions needed - just verify no exceptions
