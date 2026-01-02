"""
Utilities for extracting and handling session IDs from episode metadata.
"""

import re

# Regex pattern to match session UUIDs in source_description
# Format: (session: abc-123-def-456) or (SESSION: abc-123) with flexible spacing
# Allows optional spaces before and after the colon
SESSION_PATTERN = re.compile(r'\(session\s*:\s*([a-f0-9-]{36})\s*\)', re.IGNORECASE)


def extract_session_id(source_description: str) -> tuple[str | None, str]:
    """
    Extract session UUID from source_description and return cleaned description.

    Session IDs are expected to be embedded in parentheses with the format:
    "(session: <uuid>)" where <uuid> is a standard UUID string.

    Args:
        source_description: Original source description string that may contain
                          embedded session information

    Returns:
        Tuple of (session_id, cleaned_description)
        - session_id: Extracted UUID string or None if not found
        - cleaned_description: source_description with session info removed and cleaned

    Examples:
        >>> extract_session_id('Claude Code: agents (session: abc-123-def-456)')
        ("abc-123-def-456", "Claude Code: agents")

        >>> extract_session_id('Just a description')
        (None, "Just a description")

        >>> extract_session_id('Source (SESSION: ABC-123-DEF) extra')
        ("ABC-123-DEF", "Source extra")
    """
    match = SESSION_PATTERN.search(source_description)
    if match:
        session_id = match.group(1)
        # Remove the session portion from description
        cleaned = SESSION_PATTERN.sub('', source_description).strip()
        # Clean up extra whitespace and trailing colons/punctuation
        cleaned = re.sub(r':\s*$', '', cleaned).strip()
        # Collapse multiple spaces into single space
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return session_id, cleaned
    return None, source_description
