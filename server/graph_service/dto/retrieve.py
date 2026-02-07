from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from graph_service.dto.common import Message


class SearchQuery(BaseModel):
    group_id: str = Field(..., description='The group id for the memories to search')
    query: str
    max_facts: int = Field(default=10, description='The maximum number of facts to retrieve')
    start_date: datetime | None = Field(
        None, description='Filter facts with valid_at >= this datetime (ISO 8601)'
    )
    end_date: datetime | None = Field(
        None, description='Filter facts with valid_at <= this datetime (ISO 8601)'
    )
    center_node_uuid: str | None = Field(
        None, description='Filter search results to facts connected to this entity node UUID'
    )


class EntityNodeResponse(BaseModel):
    uuid: str
    name: str
    group_id: str
    summary: str
    labels: list[str]
    attributes: dict[str, Any]
    created_at: datetime

    class Config:
        json_encoders = {datetime: lambda v: v.astimezone(timezone.utc).isoformat()}


class EpisodeResponse(BaseModel):
    """Episode information for fact provenance."""
    uuid: str
    name: str
    content: str
    source_description: str
    session_id: str
    timestamp: datetime
    valid_at: datetime
    created_at: datetime
    group_id: str

    class Config:
        json_encoders = {datetime: lambda v: v.astimezone(timezone.utc).isoformat()}


class FactResult(BaseModel):
    uuid: str
    name: str
    fact: str
    valid_at: datetime | None
    invalid_at: datetime | None
    created_at: datetime
    expired_at: datetime | None
    similarity_score: float | None = Field(
        None, description='Similarity/reranker score for this fact (0.0-1.0, higher is more relevant)'
    )
    # Provenance fields (optional, only included when requested)
    source_node_uuid: str | None = Field(None, description='UUID of the source entity in this relationship')
    target_node_uuid: str | None = Field(None, description='UUID of the target entity in this relationship')
    source_entity: EntityNodeResponse | None = Field(None, description='Full details of the source entity')
    target_entity: EntityNodeResponse | None = Field(None, description='Full details of the target entity')
    episodes: list[EpisodeResponse] | None = Field(None, description='Episodes that created or mentioned this fact')

    class Config:
        json_encoders = {datetime: lambda v: v.astimezone(timezone.utc).isoformat()}


class SearchResults(BaseModel):
    facts: list[FactResult]


class GetMemoryRequest(BaseModel):
    group_id: str = Field(..., description='The group id of the memory to get')
    max_facts: int = Field(default=10, description='The maximum number of facts to retrieve')
    center_node_uuid: str | None = Field(
        ..., description='The uuid of the node to center the retrieval on'
    )
    messages: list[Message] = Field(
        ..., description='The messages to build the retrieval query from '
    )


class GetMemoryResponse(BaseModel):
    facts: list[FactResult] = Field(..., description='The facts that were retrieved from the graph')


class EntityListResponse(BaseModel):
    entities: list[EntityNodeResponse]
    total: int
    has_more: bool
    cursor: str | None = None


class SessionResponse(BaseModel):
    session_id: str
    uuid: str  # Session UUID from database (same as session_id for consistency)
    episode_count: int
    first_episode_date: datetime
    last_episode_date: datetime
    source_descriptions: list[str]
    summary: str | None = None
    project_name: str | None = None
    first_episode_preview: str | None = None
    programmatic: bool = False  # Whether this session is programmatic (automated) vs interactive

    class Config:
        json_encoders = {datetime: lambda v: v.astimezone(timezone.utc).isoformat()}


class SessionListResponse(BaseModel):
    sessions: list[SessionResponse]
    total: int
    has_more: bool
    cursor: str | None = None


class DaySessionCount(BaseModel):
    date: str
    count: int


class SessionStatsByDayResponse(BaseModel):
    stats: list[DaySessionCount]
    total_days: int


class ProjectResponse(BaseModel):
    """Response model for a single project."""

    uuid: str
    name: str
    created_at: datetime
    project_path: str | None = None
    episode_count: int
    session_count: int
    first_episode_date: datetime | None
    last_episode_date: datetime | None

    class Config:
        json_encoders = {datetime: lambda v: v.astimezone(timezone.utc).isoformat()}


class ProjectListResponse(BaseModel):
    """Response model for paginated list of projects."""

    projects: list[ProjectResponse]
    total: int
    has_more: bool
    cursor: str | None = None


class DayProjectActivity(BaseModel):
    """Project activity for a single day."""

    date: str  # ISO date format YYYY-MM-DD
    episode_count: int


class ProjectStatsByDayResponse(BaseModel):
    """Response model for project activity statistics by day."""

    stats: list[DayProjectActivity]
    total_days: int


class SourceExtractionResultsResponse(BaseModel):
    """Response model for source extraction results."""

    source: EntityNodeResponse  # The Source entity
    episodes: list[EpisodeResponse]  # Episodes created from the source
    facts: list[FactResult]  # Facts extracted from the source
    entities: list[EntityNodeResponse]  # Entities extracted from the source
    processing_complete: bool  # Whether extraction is finished
