"""
Tests for project extraction utilities.
"""

import pytest

from graphiti_core.utils.project_utils import extract_project_name


def test_extract_project_name_standard_format():
    """Test extraction from standard format."""
    project, source = extract_project_name('Claude Code: agents')
    assert project == 'agents'
    assert source == 'Claude Code'


def test_extract_project_name_multi_colon():
    """Test extraction with multiple colons (takes everything after first colon)."""
    project, source = extract_project_name('Source: multi: word: project')
    assert project == 'multi: word: project'
    assert source == 'Source'


def test_extract_project_name_no_colon():
    """Test extraction when no colon present."""
    project, source = extract_project_name('No colon here')
    assert project is None
    assert source == 'No colon here'


def test_extract_project_name_empty_project():
    """Test extraction with empty project after colon."""
    project, source = extract_project_name('Source:')
    assert project is None
    assert source == 'Source'


def test_extract_project_name_whitespace_project():
    """Test extraction with whitespace-only project."""
    project, source = extract_project_name('Source:   ')
    assert project is None
    assert source == 'Source'


def test_extract_project_name_lowercase():
    """Test that project name is lowercased."""
    project, source = extract_project_name('Source: MyProject')
    assert project == 'myproject'
    assert source == 'Source'


def test_extract_project_name_case_variations():
    """Test various case combinations."""
    project, source = extract_project_name('Source: MixedCase')
    assert project == 'mixedcase'
    assert source == 'Source'

    project, source = extract_project_name('Source: UPPERCASE')
    assert project == 'uppercase'
    assert source == 'Source'


def test_extract_project_name_whitespace_handling():
    """Test whitespace is properly stripped."""
    project, source = extract_project_name('  Source  :  project  ')
    assert project == 'project'
    assert source == 'Source'


def test_extract_project_name_special_characters():
    """Test project names with special characters."""
    project, source = extract_project_name('Source: my-project_v2')
    assert project == 'my-project_v2'
    assert source == 'Source'


def test_extract_project_name_empty_string():
    """Test extraction from empty string."""
    project, source = extract_project_name('')
    assert project is None
    assert source == ''
