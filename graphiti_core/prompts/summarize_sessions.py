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


def extract_initial_intent(context: dict[str, Any]) -> list[Message]:
    """Extract the primary intent from the first user message in a session.

    This creates a concise summary focused on what the user wants to accomplish.
    Only processes user messages - assistant and system messages are ignored.

    Context:
        - user_message: Content of the first user message (without role prefix)
        - session_id: UUID of the session

    Returns:
        Prompt messages that will generate a 50-100 character intent summary
    """
    return [
        Message(
            role='system',
            content='You are a concise assistant that extracts user intent from messages.',
        ),
        Message(
            role='user',
            content=f"""
        Extract the primary intent from this user message. Focus on what the user wants or is asking about.

        CRITICAL: Output must be 50-100 characters. Use format "User asked about [topic]" or "User wants to [goal]"

        User Message:
        {to_prompt_json(context['user_message'])}

        Generate a concise intent that:
        - Identifies what the user wants or is asking
        - Uses present tense ("asked about", "wants to")
        - Is 50-100 characters
        - Is factual and direct
        """,
        ),
    ]


class IntentChange(BaseModel):
    changed: bool = Field(..., description='True if the new message represents a different intent')
    new_intent: str = Field(
        ...,
        description='Brief description of the new intent (5-10 words), empty string if no change',
    )


def detect_intent_change(context: dict[str, Any]) -> list[Message]:
    """Detect if a new user message represents a different intent from the current summary.

    Uses the embedding similarity score to inform the decision, then uses LLM to
    extract a concise description of the new intent if it has changed.

    Context:
        - current_summary: Existing session summary
        - new_message: New user message content (without role prefix)
        - similarity: Cosine similarity score between summary and message (0-1)
        - session_id: UUID of the session

    Returns:
        Prompt messages that will generate an IntentChange response
    """
    similarity_pct = int(context['similarity'] * 100)
    return [
        Message(
            role='system',
            content='You are an assistant that detects intent changes in conversations.',
        ),
        Message(
            role='user',
            content=f"""
        Determine if the new user message represents a different topic or intent from the current session summary.

        Current Summary:
        {context['current_summary']}

        New User Message:
        {to_prompt_json(context['new_message'])}

        Semantic Similarity: {similarity_pct}% (< 70% suggests different intent)

        Respond with:
        - changed: true if this is a new/different topic, false if it's the same topic
        - new_intent: If changed=true, provide 5-10 word description like "the contactcenter repository" or "Python package management"
          If changed=false, leave as empty string

        Examples:
        - Same intent: "Tell me more about hooks" after summary "User asked about Claude Code hooks" -> changed=false
        - Different intent: "Look at the contactcenter repo" after summary "User asked about Claude Code hooks" -> changed=true, new_intent="the contactcenter repository"
        """,
        ),
    ]


def refine_existing_intent(context: dict[str, Any]) -> list[Message]:
    """Refine the existing summary when the user asks follow-up questions on the same topic.

    Only updates if there's meaningful progression (drilling deeper, adding details).
    Returns unchanged summary if the message is just acknowledgments or clarifications.

    Context:
        - current_summary: Existing session summary
        - new_message: New user message content (without role prefix)
        - session_id: UUID of the session

    Returns:
        Prompt messages that will generate a refined summary (under 150 characters)
    """
    return [
        Message(
            role='system',
            content='You are an assistant that tracks conversation progression.',
        ),
        Message(
            role='user',
            content=f"""
        Update the summary ONLY if the new message shows meaningful progression on the same topic.

        Current Summary:
        {context['current_summary']}

        New User Message:
        {to_prompt_json(context['new_message'])}

        Rules:
        - If message is just "thanks", "ok", "got it" -> return current summary UNCHANGED
        - If message drills deeper into the topic -> add ", then drilled into [subtopic]"
        - If message asks for clarification -> add ", then asked about [aspect]"
        - Keep under 150 characters total
        - Preserve the original intent at the start

        Examples:
        - "What about the memory hook?" after "User asked about Claude Code hooks" -> "User asked about Claude Code hooks, then drilled into memory hook"
        - "Thanks" after "User wants to debug an error" -> "User wants to debug an error" (unchanged)
        """,
        ),
    ]


def append_new_intent(context: dict[str, Any]) -> list[Message]:
    """Append a new intent to the summary when the user changes topics.

    Preserves the original intent and adds the new one.

    Context:
        - current_summary: Existing session summary
        - new_intent: Brief description of the new intent (from detect_intent_change)
        - session_id: UUID of the session

    Returns:
        Prompt messages that will generate an updated summary (under 200 characters)
    """
    return [
        Message(
            role='system',
            content='You are an assistant that tracks multiple conversation topics.',
        ),
        Message(
            role='user',
            content=f"""
        Add the new intent to the summary without overwriting the original.

        Current Summary:
        {context['current_summary']}

        New Intent to Add:
        {context['new_intent']}

        Generate updated summary:
        - Format: "[Current summary]. User also asked about [new intent]."
        - Keep under 200 characters total
        - If exceeds 200 chars, compress oldest parts or use "and other topics"
        - Preserve the original user intent

        Example:
        - Current: "User asked about Claude Code hooks, then drilled into memory hook"
        - New: "the contactcenter repository"
        - Result: "User asked about Claude Code hooks, then drilled into memory hook. User also asked about contactcenter repo."
        """,
        ),
    ]


versions = {
    'create_initial_session_summary': create_initial_session_summary,
    'summarize_session_incremental': summarize_session_incremental,
    'extract_initial_intent': extract_initial_intent,
    'detect_intent_change': detect_intent_change,
    'refine_existing_intent': refine_existing_intent,
    'append_new_intent': append_new_intent,
}
