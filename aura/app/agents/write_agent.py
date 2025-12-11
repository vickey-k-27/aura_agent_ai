# In your terminal, replace the write_agent.py content:
"""
Write Agent - Saves conversation data for analytics and agent handoff

This is a deterministic agent (not LLM-based) to avoid
infinite tool-calling loops. It directly executes the save/log operations
without needing LLM decisions.

This agent runs at the end of the conversation to:
1. Save conversation to long-term memory
2. Log telemetry to a data warehouse
3. Generate a summary for agent handoff
"""

import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, AsyncGenerator
from datetime import datetime, timezone
from typing_extensions import override

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.genai import types
from pydantic import BaseModel

from app.config import settings
from app.tools.conversation_memory import save_conversation, get_agent_handoff_context
from app.tools.telemetry import log_telemetry

logger = logging.getLogger(__name__)

# Maximum retries for GCP operations
MAX_RETRIES = 2


class WriteResult(BaseModel):
    """Output schema for write agent"""
    conversation_saved: bool
    telemetry_logged: bool
    conversation_summary: str
    topic_classification: str
    sentiment_detected: str
    resolution_status: str
    handoff_ready: bool
    handoff_context: Optional[dict] = None
    email_summary: Optional[str] = None
    relevant_links: Optional[List[str]] = None
    errors: Optional[List[str]] = None


class WriteAgent(BaseAgent):
    """
    Deterministic Write Agent - No LLM, just executes save operations.
    
    This avoids the infinite loop problem with LLM-based tool calling.
    """
    
    model_config = {"arbitrary_types_allowed": True}
    
    def __init__(self, name: str = "write_agent"):
        super().__init__(name=name, sub_agents=[])
    
    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """Execute write operations deterministically."""
        
        logger.info(f"[{self.name}] Starting deterministic write operations")
        
        errors = []
        telemetry_logged = False
        conversation_saved = False
        
        # Extract context from session state
        original_query = ctx.session.state.get("original_query", "")
        final_response = ctx.session.state.get("final_response", "")
        guardrails_result = ctx.session.state.get("guardrails_result", {})
        user_context = ctx.session.state.get("user_context", {})
        user_id = ctx.session.state.get("user_id", "")
        session_id = ctx.session.state.get("session_id", "unknown")
        
        # Parse guardrails result
        if isinstance(guardrails_result, str):
            try:
                guardrails_result = json.loads(guardrails_result)
            except:
                guardrails_result = {}
        
        # Parse user context
        if isinstance(user_context, str):
            try:
                user_context = json.loads(user_context)
            except:
                user_context = {}
        
        # Determine classification from guardrails
        category = guardrails_result.get("category", "general_inquiry")
        sentiment = guardrails_result.get("sentiment", "neutral")
        is_authenticated = guardrails_result.get("is_authenticated", False)
        
        # Determine if escalated
        was_escalated = False
        if isinstance(final_response, str):
            try:
                final_data = json.loads(final_response)
                was_escalated = final_data.get("should_escalate", False)
            except:
                pass
        
        # Map category to topic
        topic = self._map_category_to_topic(category)
        resolution_status = "escalated" if was_escalated else "resolved"
        
        # Generate simple summary
        summary = self._generate_summary(original_query, topic, resolution_status)
        
        # ============================================================
        # STEP 1: Log Telemetry (always, even for guests)
        # ============================================================
        for attempt in range(MAX_RETRIES):
            try:
                result = log_telemetry(
                    project_id=settings.GCP_PROJECT_ID,
                    session_id=session_id,
                    user_id=user_id if user_id else None,
                    is_authenticated=is_authenticated,
                    category=category,
                    sentiment=sentiment,
                    topic=topic,
                    action_taken="respond" if not was_escalated else "escalate",
                    was_escalated=was_escalated,
                    resolution_status=resolution_status,
                    flow_type="authenticated" if is_authenticated else "guest",
                    query_length=len(original_query),
                    response_length=len(str(final_response))
                )
                if result.get("success"):
                    telemetry_logged = True
                    logger.info(f"[{self.name}] Telemetry logged successfully")
                    break
                else:
                    errors.append(f"Telemetry attempt {attempt+1}: {result.get('error')}")
            except Exception as e:
                errors.append(f"Telemetry attempt {attempt+1}: {str(e)}")
                logger.warning(f"[{self.name}] Telemetry failed (attempt {attempt+1}): {e}")
        
        # ============================================================
        # STEP 2: Save Conversation (only for authenticated users)
        # ============================================================
        if user_id and is_authenticated:
            for attempt in range(MAX_RETRIES):
                try:
                    result = save_conversation(
                        project_id=settings.GCP_PROJECT_ID,
                        user_id=user_id,
                        session_id=session_id,
                        query=original_query[:500],
                        response=str(final_response)[:1000],
                        topic=topic,
                        sentiment=sentiment,
                        was_escalated=was_escalated,
                        resolution_status=resolution_status
                    )
                    if result.get("success"):
                        conversation_saved = True
                        logger.info(f"[{self.name}] Conversation saved for {user_id}")
                        break
                    else:
                        errors.append(f"Save attempt {attempt+1}: {result.get('error')}")
                except Exception as e:
                    errors.append(f"Save attempt {attempt+1}: {str(e)}")
                    logger.warning(f"[{self.name}] Save failed (attempt {attempt+1}): {e}")
        else:
            logger.info(f"[{self.name}] Skipping conversation save (guest user)")
        
        # ============================================================
        # STEP 3: Prepare handoff context (if escalated)
        # ============================================================
        handoff_context = None
        if was_escalated and user_id:
            try:
                result = get_agent_handoff_context(
                    project_id=settings.GCP_PROJECT_ID,
                    user_id=user_id,
                    current_session_summary=summary
                )
                if result.get("success"):
                    handoff_context = result.get("handoff_context")
                    logger.info(f"[{self.name}] Handoff context prepared")
            except Exception as e:
                errors.append(f"Handoff context: {str(e)}")
                logger.warning(f"[{self.name}] Handoff context failed: {e}")
        
        # ============================================================
        # Build final result
        # ============================================================
        write_result = WriteResult(
            conversation_saved=conversation_saved,
            telemetry_logged=telemetry_logged,
            conversation_summary=summary,
            topic_classification=topic,
            sentiment_detected=sentiment,
            resolution_status=resolution_status,
            handoff_ready=handoff_context is not None,
            handoff_context=handoff_context,
            errors=errors if errors else None
        )
        
        # Store result in session state
        ctx.session.state["write_result"] = write_result.model_dump_json()
        
        logger.info(
            f"[{self.name}] Complete: telemetry={telemetry_logged}, "
            f"saved={conversation_saved}, errors={len(errors)}"
        )
        
        # Yield completion event (no content to user)
        yield Event(
            author=self.name,
            actions=EventActions(state_delta={"write_result": write_result.model_dump_json()})
        )
    
    def _map_category_to_topic(self, category: str) -> str:
        """Map guardrails category to topic classification."""
        mapping = {
            "user_question": "user_question",
            "claims": "claims_inquiry",
            "billing": "billing_question",
            "account_change": "account_change",
            "complaint": "complaint",
            "greeting": "general_inquiry",
            "thanks": "general_inquiry",
            "goodbye": "general_inquiry",
            "out_of_scope": "general_inquiry",
        }
        return mapping.get(category, "general_inquiry")
    
    def _generate_summary(self, query: str, topic: str, resolution: str) -> str:
        """Generate a simple conversation summary."""
        topic_desc = {
            "user_question": "their account",
            "claims_inquiry": "claims",
            "billing_question": "billing",
            "account_change": "account changes",
            "complaint": "a complaint",
            "general_inquiry": "general information"
        }
        topic_text = topic_desc.get(topic, "their account")
        
        if resolution == "escalated":
            return f"User asked about {topic_text}. Transferred to human agent for assistance."
        else:
            return f"User asked about {topic_text}. Query was resolved."


def create_write_agent() -> WriteAgent:
    """
    Create the Write Agent for end-of-conversation processing.
    
    This is a DETERMINISTIC agent (not LLM-based) that directly
    executes save/log operations with retry logic and graceful fallback.
    
    Returns:
        WriteAgent configured for reliable background processing
    """
    agent = WriteAgent(name="write_agent")
    logger.info("Write Agent created (deterministic mode)")
    return agent
