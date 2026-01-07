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

from graphiti_core.graphiti import extract_message_content, extract_role_type


def test_extract_role_type_user():
    """Test extracting 'user' role type."""
    content = '[user]: Hello, how are you?'
    assert extract_role_type(content) == 'user'


def test_extract_role_type_assistant():
    """Test extracting 'assistant' role type."""
    content = '[assistant]: I am doing well, thank you!'
    assert extract_role_type(content) == 'assistant'


def test_extract_role_type_system():
    """Test extracting 'system' role type."""
    content = '[system]: System initialization complete'
    assert extract_role_type(content) == 'system'


def test_extract_role_type_no_prefix():
    """Test that None is returned when no role prefix exists."""
    content = 'This is a message without a role prefix'
    assert extract_role_type(content) is None


def test_extract_role_type_empty_string():
    """Test that None is returned for empty string."""
    content = ''
    assert extract_role_type(content) is None


def test_extract_role_type_malformed():
    """Test that None is returned for malformed prefix."""
    content = '[user] Missing colon'
    assert extract_role_type(content) is None


def test_extract_role_type_with_extra_spaces():
    """Test role extraction with extra spaces after colon."""
    content = '[user]:     Message with extra spaces'
    assert extract_role_type(content) == 'user'


def test_extract_message_content_user():
    """Test extracting message content from user message."""
    content = '[user]: Hello, how are you?'
    assert extract_message_content(content) == 'Hello, how are you?'


def test_extract_message_content_assistant():
    """Test extracting message content from assistant message."""
    content = '[assistant]: I am doing well, thank you!'
    assert extract_message_content(content) == 'I am doing well, thank you!'


def test_extract_message_content_system():
    """Test extracting message content from system message."""
    content = '[system]: System initialization complete'
    assert extract_message_content(content) == 'System initialization complete'


def test_extract_message_content_no_prefix():
    """Test that original content is returned when no role prefix exists."""
    content = 'This is a message without a role prefix'
    assert extract_message_content(content) == content


def test_extract_message_content_empty_string():
    """Test that empty string is returned for empty string."""
    content = ''
    assert extract_message_content(content) == ''


def test_extract_message_content_multiline():
    """Test extracting multiline message content."""
    content = """[user]: This is a multiline message
that spans multiple lines
and preserves formatting"""
    expected = """This is a multiline message
that spans multiple lines
and preserves formatting"""
    assert extract_message_content(content) == expected


def test_extract_message_content_with_brackets_in_message():
    """Test that brackets within the message content are preserved."""
    content = '[user]: I need [help] with this [problem]'
    assert extract_message_content(content) == 'I need [help] with this [problem]'


def test_extract_message_content_with_colon_in_message():
    """Test that colons within the message content are preserved."""
    content = '[user]: The time is 3:45 PM and the ratio is 1:2'
    assert extract_message_content(content) == 'The time is 3:45 PM and the ratio is 1:2'


def test_extract_message_content_empty_message():
    """Test extracting content when message body is single space."""
    content = '[user]: '
    # \s* matches zero chars, .+ matches the single space
    assert extract_message_content(content) == ' '


def test_extract_message_content_only_whitespace():
    """Test extracting content when message body is multiple spaces."""
    content = '[user]:    '  # 4 spaces after colon
    # \s* greedily matches 3 spaces, .+ matches remaining 1 space (minimum required)
    assert extract_message_content(content) == ' '


def test_extract_role_type_and_content_together():
    """Test that both functions work correctly on the same input."""
    content = '[assistant]: This is a test message'
    role_type = extract_role_type(content)
    message = extract_message_content(content)

    assert role_type == 'assistant'
    assert message == 'This is a test message'
    assert f'[{role_type}]: {message}' == content


def test_extract_role_type_case_sensitive():
    """Test that role type extraction is case sensitive."""
    content = '[User]: Hello'
    assert extract_role_type(content) == 'User'  # Preserves case


def test_real_world_example_user_question():
    """Test with real-world user question example."""
    content = '[user]: Can you explain how session summarization works in Graphiti?'
    assert extract_role_type(content) == 'user'
    assert extract_message_content(content) == 'Can you explain how session summarization works in Graphiti?'


def test_real_world_example_assistant_response():
    """Test with real-world assistant response example."""
    content = '[assistant]: Session summarization in Graphiti uses an incremental approach that updates summaries as new episodes are added to a session.'
    assert extract_role_type(content) == 'assistant'
    assert extract_message_content(content) == 'Session summarization in Graphiti uses an incremental approach that updates summaries as new episodes are added to a session.'


def test_real_world_example_system_message():
    """Test with real-world system message example."""
    content = '[system]: Session created with ID: 550e8400-e29b-41d4-a716-446655440000'
    assert extract_role_type(content) == 'system'
    assert extract_message_content(content) == 'Session created with ID: 550e8400-e29b-41d4-a716-446655440000'
