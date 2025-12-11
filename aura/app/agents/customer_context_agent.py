"""
Context Agent - Enhanced with Long-Term Memory

This is a deterministic agent (not LLM-based) to avoid
infinite tool-calling loops. It directly fetches data without LLM decisions.

Fetches:
1. User data from a database
2. User tier limits from a database
3. Long-term conversation history
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional, List, AsyncGenerator
from typing_extensions import override

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from pydantic import BaseModel

from app.config import settings
from app.tools.fetch_customer import fetch_customer
from app.tools.fetch_limits import fetch_limits
from app.tools.conversation_memory import fetch_conversation_history

logger = logging.getLogger(__name__)

# Maximum retries for GCP operations
MAX_RETRIES = 2


class ContextResult(BaseModel):
    """Output schema for user context with conversation history"""
    # User found status
    user_found: bool
    
    # User profile
    user_id: Optional[str] = None
    user_name: Optional[str] = None
    user_tier: Optional[str] = None
    
    # User details
    item_1_sum: Optional[int] = None
    item_2_sum: Optional[int] = None
    standard_excess: Optional[int] = None
    extra_excess: Optional[int] = None
    add_ons: Optional[List[str]] = None
    special_conditions: Optional[str] = None
    tier_limits: Optional[dict] = None
    
    # Long-term conversation context
    has_previous_interactions: bool = False
    total_previous_calls: int = 0
    last_topic: Optional[str] = None
    last_interaction_date: Optional[str] = None
    unresolved_issues: Optional[List[str]] = None
    sentiment_trend: Optional[str] = None
    conversation_summary: Optional[str] = None
    
    # Combined summary for downstream agents
    summary: str


class ContextAgent(BaseAgent):
    """
    Deterministic Context Agent - No LLM, just fetches data.
    
    This avoids the infinite loop problem with LLM-based tool calling.
    """
    
    model_config = {"arbitrary_types_allowed": True}
    
    def __init__(self, name: str = "context_agent"):
        super().__init__(name=name, sub_agents=[])
    
    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """Fetch user context deterministically with comprehensive error handling."""
        
        try:
            logger.info(f"[{self.name}] Starting deterministic context fetch")
            
            # Get identifiers from session state
            user_id = ctx.session.state.get("user_id", "")
            
            # Initialize result
            result = ContextResult(
                user_found=False,
                summary="Unable to fetch user context"
            )
            
            # ============================================================
            # STEP 1: Fetch User Profile with validation
            # ============================================================
            user_data = None
            for attempt in range(MAX_RETRIES):
                try:
                    if user_id:
                        user_result = fetch_customer(
                            project_id=settings.GCP_PROJECT_ID,
                            customer_id=user_id
                        )
                    else:
                        logger.warning(f"[{self.name}] No user_id provided")
                        break
                    
                    if user_result and isinstance(user_result, dict) and user_result.get("success"):
                        user_data = user_result.get("customer", {})
                        user_id = user_result.get("customer_id") or user_id
                        # Update session state with resolved user_id
                        ctx.session.state["user_id"] = user_id
                        logger.info(f"[{self.name}] User found: {user_data.get('name')}")
                        break
                    else:
                        error_msg = user_result.get("error", "Unknown error") if isinstance(user_result, dict) else "Invalid result type"
                        logger.warning(f"[{self.name}] User lookup failed: {error_msg}")
                except Exception as e:
                    logger.warning(f"[{self.name}] User fetch attempt {attempt+1}/{MAX_RETRIES} failed: {e}")
                    if attempt < MAX_RETRIES - 1:
                        await asyncio.sleep(0.5 * (attempt + 1))
            
            if not user_data:
                # Return early if no user found
                result.summary = "User not found in database"
                self._safe_store_result(ctx, result)
                yield Event(
                    author=self.name,
                    actions=EventActions(state_delta={"customer_context": ctx.session.state.get("customer_context")})
                )
                return
            
            # ============================================================
            # STEP 2: Fetch Tier Limits with validation
            # ============================================================
            tier_limits = None
            user_tier = user_data.get("tier", "Standard")
            
            for attempt in range(MAX_RETRIES):
                try:
                    limits_result = fetch_limits(
                        project_id=settings.GCP_PROJECT_ID,
                        tier_name=user_tier
                    )
                    if limits_result and isinstance(limits_result, dict) and limits_result.get("success"):
                        tier_limits = limits_result.get("limits", {})
                        logger.info(f"[{self.name}] Tier limits found for {user_tier}")
                        break
                except Exception as e:
                    logger.warning(f"[{self.name}] Limits fetch attempt {attempt+1}/{MAX_RETRIES} failed: {e}")
                    if attempt < MAX_RETRIES - 1:
                        await asyncio.sleep(0.5 * (attempt + 1))
            
            # ============================================================
            # STEP 3: Fetch Conversation History with validation
            # ============================================================
            history_data = None
            if user_id:
                for attempt in range(MAX_RETRIES):
                    try:
                        history_result = fetch_conversation_history(
                            project_id=settings.GCP_PROJECT_ID,
                            customer_id=user_id,
                            limit=5
                        )
                        if history_result and isinstance(history_result, dict) and history_result.get("success"):
                            history_data = history_result
                            logger.info(f"[{self.name}] History fetched: {history_result.get('total_calls', 0)} calls")
                            break
                    except Exception as e:
                        logger.warning(f"[{self.name}] History fetch attempt {attempt+1}/{MAX_RETRIES} failed: {e}")
                        if attempt < MAX_RETRIES - 1:
                            await asyncio.sleep(0.5 * (attempt + 1))
            
            # ============================================================
            # STEP 4: Build Result
            # ============================================================
            result = ContextResult(
                user_found=True,
                user_id=user_id,
                user_name=user_data.get("name"),
                user_tier=user_tier,
                item_1_sum=user_data.get("item_1_sum"),
                item_2_sum=user_data.get("item_2_sum"),
                standard_excess=user_data.get("standard_excess"),
                extra_excess=user_data.get("extra_excess"),
                add_ons=user_data.get("add_ons"),
                special_conditions=user_data.get("special_conditions"),
                tier_limits=tier_limits,
                has_previous_interactions=history_data.get("has_history", False) if history_data else False,
                total_previous_calls=history_data.get("total_calls", 0) if history_data else 0,
                last_topic=history_data.get("last_topic") if history_data else None,
                last_interaction_date=history_data.get("last_interaction_date") if history_data else None,
                unresolved_issues=history_data.get("unresolved_issues") if history_data else None,
                sentiment_trend=history_data.get("sentiment_trend") if history_data else None,
                conversation_summary=history_data.get("summary") if history_data else None,
                summary=self._build_summary(user_data, tier_limits, history_data)
            )
            
            self._safe_store_result(ctx, result)
            
            logger.info(f"[{self.name}] Complete: user={result.user_name}, tier={result.user_tier}")
            
            # Yield completion event
            yield Event(
                author=self.name,
                actions=EventActions(state_delta={"customer_context": ctx.session.state.get("customer_context")})
            )
        
        except Exception as e:
            logger.error(f"[{self.name}] CRITICAL ERROR in context agent: {e}", exc_info=True)
            
            # Return safe fallback
            error_result = ContextResult(
                user_found=False,
                summary=f"Error fetching user context: {str(e)[:100]}"
            )
            
            self._safe_store_result(ctx, error_result)
            
            yield Event(
                author=self.name,
                actions=EventActions(state_delta={
                    "customer_context": ctx.session.state.get("customer_context"),
                    "context_error": str(e)[:200]
                })
            )
    
    def _safe_store_result(self, ctx: InvocationContext, result: ContextResult) -> None:
        """Safely store result with multiple fallback levels."""
        try:
            # Primary: Try JSON serialization
            ctx.session.state["customer_context"] = result.model_dump_json()
        except Exception as e1:
            logger.warning(f"[{self.name}] JSON serialization failed: {e1}, trying dict...")
            try:
                # Secondary: Try dict serialization
                ctx.session.state["customer_context"] = json.dumps(result.model_dump())
            except Exception as e2:
                logger.error(f"[{self.name}] Dict serialization failed: {e2}, using minimal fallback")
                # Ultimate fallback: Minimal dict
                ctx.session.state["customer_context"] = json.dumps({
                    "user_found": False,
                    "summary": "Critical error storing user context"
                })
    
    def _build_summary(self, user: dict, limits: dict, history: dict) -> str:
        """Build a comprehensive summary for downstream agents."""
        try:
            parts = []
            
            user = user or {}
            limits = limits or {}
            history = history or {}
            
            name = user.get("name", "User")
            tier = user.get("tier", "Standard")
            item_1 = user.get("item_1_sum", 0)
            extra_excess = user.get("extra_excess", 0)
            
            try:
                item_1 = int(item_1) if item_1 else 0
            except (ValueError, TypeError):
                item_1 = 0
            
            try:
                extra_excess = int(extra_excess) if extra_excess else 0
            except (ValueError, TypeError):
                extra_excess = 0
            
            # Basic info
            parts.append(f"{name} is a {tier} tier user")
            
            if item_1:
                parts.append(f"with ${item_1:,} item_1 cover")
            
            if extra_excess:
                parts.append(f"Extra excess: ${extra_excess:,}")
            
            # History info
            if history and history.get("has_history"):
                total_calls = history.get("total_calls", 0)
                last_topic = history.get("last_topic")
                sentiment = history.get("sentiment_trend")
                unresolved = history.get("unresolved_issues", [])
                
                if total_calls > 1:
                    parts.append(f"Returning user ({total_calls} previous calls)")
                
                if last_topic:
                    parts.append(f"Last discussed: {last_topic}")
                
                if unresolved:
                    parts.append(f"Unresolved issues: {', '.join(unresolved)}")
                
                if sentiment == "declining":
                    parts.append("Sentiment declining - handle with care")
            else:
                parts.append("This appears to be their first call")
            
            return ". ".join(parts) + "."
        
        except Exception as e:
            logger.error(f"[{self.name}] Error building summary: {e}", exc_info=True)
            try:
                name = user.get("name", "User") if user else "User"
                return f"{name} - context available but summary generation failed"
            except:
                return "User context available but summary generation failed"


def create_customer_context_agent() -> ContextAgent:
    """
    Create the Context Agent.
    
    This is a DETERMINISTIC agent (not LLM-based) that directly
    fetches user data with retry logic and comprehensive error handling.
    
    Returns:
        ContextAgent configured for reliable data fetching
    """
    agent = ContextAgent(name="context_agent")
    logger.info("Context Agent created (deterministic mode with full error handling)")
    return agent
