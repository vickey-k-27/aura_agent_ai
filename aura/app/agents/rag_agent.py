"""
RAG Agent - LlmAgent for document search
"""

import logging
from pathlib import Path
from typing import Callable
from google.adk.agents import LlmAgent
from app.config import settings
from app.tools.search_policy import get_policy_search_tool

logger = logging.getLogger(__name__)


def _load_base_prompt() -> str:
    """Load RAG system prompt from file."""
    prompt_paths = [
        Path("app/prompts/rag.txt"),
        Path("prompts/rag.txt"),
        Path(__file__).parent.parent / "prompts" / "rag.txt",
    ]
    
    for path in prompt_paths:
        if path.exists():
            logger.info(f"Loaded RAG prompt from: {path}")
            return path.read_text().strip()
    
    logger.warning("Could not find rag.txt, using minimal prompt")
    return """You are a helpful assistant. Use the search tool to find answers in documents.
    Always cite your sources. Keep answers concise and voice-friendly."""


def _create_dynamic_instruction() -> Callable:
    """
    Create a dynamic instruction function that injects session state values.
    
    ADK LlmAgent supports callable instructions that receive the invocation context.
    """
    base_prompt = _load_base_prompt()
    
    def dynamic_instruction(ctx) -> str:
        """Build instruction with session state values."""
        # Get values from session state
        guardrails_result = ctx.session.state.get("guardrails_result", "{}")
        original_query = ctx.session.state.get("original_query", "")
        
        # Replace placeholders in prompt
        instruction = base_prompt.replace("{guardrails_result}", str(guardrails_result))
        instruction = instruction.replace("{original_query}", original_query)
        
        return instruction
    
    return dynamic_instruction


def create_rag_agent() -> LlmAgent:
    """
    Create the RAG (Retrieval Augmented Generation) agent.
    
    This agent:
    - Uses Vertex AI Search to find information
    - Provides grounded, cited answers
    - Keeps responses voice-friendly
    
    Uses a smart model for best RAG quality.
    
    Returns:
        LlmAgent configured with VertexAiSearchTool
    """
    # Get the search tool
    search_tool = get_policy_search_tool()
    
    agent = LlmAgent(
        name="rag_agent",
        model=settings.MODEL_RAG,
        description="Searches documents and provides grounded answers",
        instruction=_create_dynamic_instruction(),
        tools=[search_tool],
        output_key="rag_answer",  # Saves to session state
    )
    
    logger.info(f"RAG agent created with model: {settings.MODEL_RAG}")
    return agent
