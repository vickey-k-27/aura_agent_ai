"""
Response Agent - LlmAgent for voice formatting and escalation decisions

ENHANCED: Now receives user_context for true personalization:
- Address user by name
- Reference their specific details
- Acknowledge conversation history
- Adjust tone based on sentiment trend
"""

import logging
from pathlib import Path
from typing import Callable
from google.adk.agents import LlmAgent
from app.config import settings

logger = logging.getLogger(__name__)


def _load_base_prompt() -> str:
    """Load response system prompt from file."""
    prompt_paths = [
        Path("app/prompts/response.txt"),
        Path("prompts/response.txt"),
        Path(__file__).parent.parent / "prompts" / "response.txt",
    ]
    
    for path in prompt_paths:
        if path.exists():
            logger.info(f"Loaded response prompt from: {path}")
            return path.read_text().strip()
    
    logger.warning("Could not find response.txt, using minimal prompt")
    return """You format answers for voice delivery with personalization.
    Output JSON with: decision (respond/escalate), speech_text, should_escalate, escalation_reason, follow_up_prompt"""


def _create_dynamic_instruction() -> Callable:
    """
    Create a dynamic instruction function that injects session state values.
    
    ADK LlmAgent supports callable instructions that receive the invocation context.
    Now includes user_context for personalization!
    """
    base_prompt = _load_base_prompt()
    
    def dynamic_instruction(ctx) -> str:
        """Build instruction with session state values including user context."""
        # Get values from session state
        guardrails_result = ctx.session.state.get("guardrails_result", "{}")
        rag_answer = ctx.session.state.get("rag_answer", "")
        original_query = ctx.session.state.get("original_query", "")
        user_context = ctx.session.state.get("user_context", "{}")
        
        # Replace placeholders in prompt
        instruction = base_prompt.replace("{guardrails_result}", str(guardrails_result))
        instruction = instruction.replace("{rag_answer}", str(rag_answer))
        instruction = instruction.replace("{original_query}", original_query)
        instruction = instruction.replace("{user_context}", str(user_context))
        
        return instruction
    
    return dynamic_instruction


def create_response_agent() -> LlmAgent:
    """
    Create the Response formatting agent.
    
    This agent:
    - Takes RAG answer + guardrails + USER CONTEXT
    - Personalizes response with user name, and details
    - Acknowledges conversation history ("I see you called before...")
    - Adjusts tone based on sentiment trend
    - Formats for natural voice delivery
    - Decides whether to escalate to human
    
    Uses a fast model for low latency.
    
    Returns:
        LlmAgent configured for personalized response formatting
    """
    agent = LlmAgent(
        name="response_agent",
        model=settings.MODEL_RESPONSE,
        description="Formats personalized answers for voice delivery and decides on escalation",
        instruction=_create_dynamic_instruction(),
        output_key="final_response",  # Saves to session state
    )
    
    logger.info(f"Response agent created with model: {settings.MODEL_RESPONSE}")
    return agent