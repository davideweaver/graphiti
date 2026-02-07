from datetime import datetime
from typing import Literal

from graphiti_core.utils.datetime_utils import utc_now
from pydantic import BaseModel, Field


class Result(BaseModel):
    message: str
    success: bool


class GroupInfo(BaseModel):
    group_id: str = Field(..., description='The group ID')
    entity_count: int = Field(0, description='Number of entities in this group')
    episode_count: int = Field(0, description='Number of episodes in this group')
    fact_count: int = Field(0, description='Number of facts (edges) in this group')


class GroupsListResponse(BaseModel):
    groups: list[GroupInfo] = Field(..., description='List of available groups')
    total: int = Field(..., description='Total number of groups')


class Message(BaseModel):
    content: str = Field(..., description='The content of the message')
    uuid: str | None = Field(default=None, description='The uuid of the message (optional)')
    name: str = Field(
        default='', description='The name of the episodic node for the message (optional)'
    )
    role_type: Literal['user', 'assistant', 'system'] = Field(
        ..., description='The role type of the message (user, assistant or system)'
    )
    role: str | None = Field(
        description='The custom role of the message to be used alongside role_type (user name, bot name, etc.)',
    )
    timestamp: datetime = Field(default_factory=utc_now, description='The timestamp of the message')
    source_description: str = Field(
        default='', description='The description of the source of the message'
    )
    session_id: str | None = Field(
        default=None, description='The session UUID for grouping related episodes'
    )
    project_name: str | None = Field(
        default=None, description='The project name (will be lowercased) for grouping related episodes'
    )
    project_path: str | None = Field(
        default=None, description='The file system path of the project'
    )
    programmatic: bool = Field(
        default=False,
        description='Whether this session is programmatically-generated (automated, imported, background) vs human-interactive',
    )
