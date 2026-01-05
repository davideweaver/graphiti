"""
Utilities for extracting and handling project names from episode metadata.
"""


def extract_project_name(source_description: str) -> tuple[str | None, str]:
    """
    Extract project name from source_description using <Source>: <Project> pattern.

    Splits on first colon. Everything after first colon becomes project name (lowercased).
    Returns cleaned source description (part before colon).

    Args:
        source_description: Original source description string

    Returns:
        Tuple of (project_name, cleaned_description)
        - project_name: Extracted project name (lowercased) or None if not found
        - cleaned_description: source_description before first colon (or full string if no colon)

    Examples:
        >>> extract_project_name('Claude Code: agents')
        ("agents", "Claude Code")

        >>> extract_project_name('Source: multi: word: project')
        ("multi: word: project", "Source")

        >>> extract_project_name('No colon here')
        (None, "No colon here")

        >>> extract_project_name('Source:')
        (None, "Source")  # Empty project after colon

        >>> extract_project_name('Source:   ')
        (None, "Source")  # Whitespace-only project
    """
    if ':' not in source_description:
        return None, source_description

    parts = source_description.split(':', 1)  # Split on first colon only
    source = parts[0].strip()
    project = parts[1].strip() if len(parts) > 1 else ''

    # If project is empty or whitespace-only, return None
    if not project:
        return None, source

    # Lowercase the project name
    project_lower = project.lower()

    return project_lower, source
