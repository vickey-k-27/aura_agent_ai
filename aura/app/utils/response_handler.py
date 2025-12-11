"""
Response Handler - Clean abstraction for yielding Events and managing responses

This eliminates hard-coded messages and provides a single point for Event handling.
"""

import json
import logging
from typing import Optional, Dict, Any
from google.adk.events import Event
from google.genai import types

logger = logging.getLogger(__name__)


class ResponseHandler:
    """
    Handles Event creation and yielding with analytics and logging.

    Benefits:
    - Single responsibility: Event creation
    - Easy to test
    - Consistent logging
    - Analytics tracking built-in
    """

    def __init__(self, agent_name: str):
        self.agent_name = agent_name

    def create_event(
        self,
        message: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Event:
        """
        Create an Event from a message string.

        Args:
            message: The text to send to the user
            metadata: Optional metadata for analytics/logging

        Returns:
            Event ready to be yielded
        """
        if metadata:
            logger.info(
                f"[{self.agent_name}] Creating response: {metadata.get('action', 'unknown')} "
                f"| category={metadata.get('category', 'unknown')}"
            )

        content = types.Content(
            role="model",
            parts=[types.Part(text=message)]
        )

        return Event(author=self.agent_name, content=content)

    def create_final_state(
        self,
        decision: str,
        speech_text: str,
        should_escalate: bool = False,
        escalation_reason: Optional[str] = None,
        follow_up_prompt: Optional[str] = None
    ) -> str:
        """
        Create standardized final_response JSON for session state.

        This is separate from Event creation for clarity.
        """
        return json.dumps({
            "decision": decision,
            "speech_text": speech_text,
            "should_escalate": should_escalate,
            "escalation_reason": escalation_reason,
            "follow_up_prompt": follow_up_prompt,
        })