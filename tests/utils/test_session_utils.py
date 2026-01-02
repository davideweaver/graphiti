"""
Tests for session ID extraction utility.
"""

from graphiti_core.utils.session_utils import extract_session_id


def test_extract_session_id_standard_format():
    """Test extraction from standard Claude Code format."""
    desc = 'Claude Code: agents (session: 12345678-1234-1234-1234-123456789012)'
    session_id, cleaned = extract_session_id(desc)
    assert session_id == '12345678-1234-1234-1234-123456789012'
    assert cleaned == 'Claude Code: agents'
    assert '(session:' not in cleaned


def test_extract_session_id_case_insensitive():
    """Test that extraction is case-insensitive."""
    desc = 'Source (SESSION: 12345678-1234-1234-1234-123456789012)'
    session_id, cleaned = extract_session_id(desc)
    assert session_id == '12345678-1234-1234-1234-123456789012'
    # Verify session info is removed
    assert 'SESSION:' not in cleaned
    assert '(session:' not in cleaned.lower()


def test_extract_session_id_no_match():
    """Test that non-matching descriptions are returned unchanged."""
    desc = 'Just a description'
    session_id, cleaned = extract_session_id(desc)
    assert session_id is None
    assert cleaned == 'Just a description'


def test_extract_session_id_whitespace_handling():
    """Test extraction with various whitespace patterns."""
    # Extra spaces around colon
    desc = 'Source (session : 12345678-1234-1234-1234-123456789012 ) extra'
    session_id, cleaned = extract_session_id(desc)
    assert session_id == '12345678-1234-1234-1234-123456789012'
    assert 'extra' in cleaned
    assert '(session' not in cleaned


def test_extract_session_id_trailing_colon_removed():
    """Test that trailing colons after session removal are cleaned up."""
    desc = 'Source: (session: 12345678-1234-1234-1234-123456789012)'
    session_id, cleaned = extract_session_id(desc)
    assert session_id == '12345678-1234-1234-1234-123456789012'
    # Should remove trailing colon left after session removal
    assert cleaned == 'Source'
    assert not cleaned.endswith(':')


def test_extract_session_id_multiple_spaces_collapsed():
    """Test that multiple spaces are collapsed to single space."""
    desc = 'Source    text (session: 12345678-1234-1234-1234-123456789012)    extra'
    session_id, cleaned = extract_session_id(desc)
    assert session_id == '12345678-1234-1234-1234-123456789012'
    # Multiple spaces should be collapsed
    assert '    ' not in cleaned


def test_extract_session_id_real_world_example():
    """Test with real-world Claude Code session format."""
    desc = 'Claude Code: agents (session: bab16df7-3d0f-4ea1-94a6-1291d5f55a5f)'
    session_id, cleaned = extract_session_id(desc)
    assert session_id == 'bab16df7-3d0f-4ea1-94a6-1291d5f55a5f'
    assert cleaned == 'Claude Code: agents'
    assert len(session_id) == 36  # Standard UUID length


def test_extract_session_id_empty_description():
    """Test handling of empty description."""
    desc = ''
    session_id, cleaned = extract_session_id(desc)
    assert session_id is None
    assert cleaned == ''


def test_extract_session_id_only_session_info():
    """Test when description contains only session info."""
    desc = '(session: 12345678-1234-1234-1234-123456789012)'
    session_id, cleaned = extract_session_id(desc)
    assert session_id == '12345678-1234-1234-1234-123456789012'
    assert cleaned == ''


def test_extract_session_id_uuid_format_validation():
    """Test that only properly formatted UUIDs (36 chars) are extracted."""
    # Too short - should not match
    desc = 'Source (session: short-id)'
    session_id, cleaned = extract_session_id(desc)
    assert session_id is None
    assert cleaned == desc

    # Correct length - should match
    desc = 'Source (session: 12345678-1234-1234-1234-123456789012)'
    session_id, cleaned = extract_session_id(desc)
    assert session_id == '12345678-1234-1234-1234-123456789012'
    assert len(session_id) == 36
