from pydantic import BaseModel, Field

from graph_service.dto.common import Message


class AddMessagesRequest(BaseModel):
    group_id: str = Field(..., description='The group id of the messages to add')
    messages: list[Message] = Field(..., description='The messages to add')
    skip_extraction: bool = Field(
        default=False,
        description='If True, skip entity and fact extraction (creates episodes only)',
    )


class AddContentRequest(BaseModel):
    group_id: str = Field(..., description='The group id for the content')
    content: str = Field(..., description='The raw content to process into episode(s)')
    project_name: str | None = Field(
        default=None,
        description='The project name (null maps to "_general")',
    )
    source_name: str = Field(..., description='Name of the source (e.g., filename or "Manual Entry")')
    source_type: str = Field(..., description='Type of source: file, text, session, meeting, etc.')
    source_metadata: dict = Field(
        default_factory=dict,
        description='Additional metadata (filename, file_size, uploaded_at, etc.)',
    )


class AddEntityNodeRequest(BaseModel):
    uuid: str = Field(..., description='The uuid of the node to add')
    group_id: str = Field(..., description='The group id of the node to add')
    name: str = Field(..., description='The name of the node to add')
    summary: str = Field(default='', description='The summary of the node to add')


class UpdateEntityEdgeRequest(BaseModel):
    fact: str = Field(..., description='The updated fact text')
    group_id: str = Field(..., description='The group id of the edge')
