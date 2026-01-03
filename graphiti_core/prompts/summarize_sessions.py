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

from typing import Any

from pydantic import BaseModel, Field

from .models import Message
from .prompt_helpers import to_prompt_json


class SessionSummary(BaseModel):
    summary: str = Field(
        ...,
        description='Brief session summary capturing main topics and outcomes. Under 250 characters',
    )


def create_initial_session_summary(context: dict[str, Any]) -> list[Message]:
    """Create an initial summary for a new session from its first episode(s).

    Context:
        - episode_content: Content of the first episode
        - source_description: Session context (e.g., "Claude Code: agents")
        - session_id: UUID of the session
    """
    return [
        Message(
            role='system',
            content='You are a helpful assistant that creates concise session summaries.',
        ),
        Message(
            role='user',
            content=f"""
        Create a brief summary for this conversation session based on the initial content.
        Focus on the main topic and purpose of the session.

        IMPORTANT: Keep the summary concise and factual. SUMMARIES MUST BE LESS THAN 250 CHARACTERS.

        Session Context: {context.get('source_description', 'Unknown')}

        Episode Content:
        {to_prompt_json(context['episode_content'])}

        Generate a summary that:
        - Identifies the primary topic or purpose
        - Is factual and direct
        - Stays under 250 characters
        """,
        ),
    ]


def summarize_session_incremental(context: dict[str, Any]) -> list[Message]:
    """Update an existing session summary with a new episode.

    Context:
        - previous_summary: Current session summary
        - new_episode_content: Content of the new episode to incorporate
        - episode_count: Total number of episodes in the session so far
        - session_id: UUID of the session
    """
    return [
        Message(
            role='system',
            content='You are a helpful assistant that updates session summaries as conversations progress.',
        ),
        Message(
            role='user',
            content=f"""
        Update the session summary to incorporate the new episode information.

        IMPORTANT: Keep the summary concise. SUMMARIES MUST BE LESS THAN 250 CHARACTERS.

        Previous Summary (based on {context['episode_count']} episodes):
        {context['previous_summary']}

        New Episode:
        {to_prompt_json(context['new_episode_content'])}

        Generate an updated summary that:
        - Captures the session's main topic and evolution
        - Incorporates key information from the new episode
        - Notes any decisions, outcomes, or important context
        - Stays under 250 characters
        - Maintains factual accuracy
        """,
        ),
    ]


versions = {
    'create_initial_session_summary': create_initial_session_summary,
    'summarize_session_incremental': summarize_session_incremental,
}
