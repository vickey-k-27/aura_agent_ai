"""
Orchestrator Agent - Two-Flow Architecture

This agent orchestrates the conversation flow based on user authentication.

- Authenticated Users: Guardrails -> Customer Context -> RAG -> Response
- Guest Users: Guardrails -> RAG -> Response

This approach ensures that the Customer Context agent is only used when
necessary, improving performance and maintainability. The orchestrator is
responsible for routing, following the single responsibility principle.
"""

import json
import logging
from typing import AsyncGenerator, Union, Any
from typing_extensions import override
from pydantic import PrivateAttr

from google.adk.agents import Agent, BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event

from app.config import settings
from app.agents.guardrails_agent import create_guardrails_agent
from app.agents.customer_context_agent import create_customer_context_agent, CustomerContextAgent
from app.agents.rag_agent import create_rag_agent
from app.agents.response_agent import create_response_agent
from app.agents.write_agent import create_write_agent, WriteAgent
from app.utils.response_handler import ResponseHandler
from app.utils.response_templates import ResponseTemplates

logger = logging.getLogger(__name__)


class Orchestrator(BaseAgent):
    """
    Orchestrator with Long-Term Memory
    
    Flow 1: AUTHENTICATED
        Guardrails → Customer Context (with history) → RAG → Response → Write
    
    Flow 2: GUEST
        Guardrails → RAG → Response → Write
    
    The Write Agent runs at the end to save the conversation to long-term memory.
    
    NOTE: CustomerContextAgent and WriteAgent are deterministic to avoid
    infinite tool-calling loops.
    """

    model_config = {"arbitrary_types_allowed": True}
    
    # Declare private attributes for Pydantic
    _guardrails: Any = PrivateAttr(default=None)
    _customer_context: Any = PrivateAttr(default=None)
    _rag: Any = PrivateAttr(default=None)
    _response: Any = PrivateAttr(default=None)
    _write: Any = PrivateAttr(default=None)
    _response_handler: Any = PrivateAttr(default=None)

    def __init__(
        self,
        name: str = "orchestrator",
        guardrails_agent: Agent = None,
        customer_context_agent: BaseAgent = None,
        rag_agent: Agent = None,
        response_agent: Agent = None,
        write_agent: BaseAgent = None,
    ):
        """Initialize orchestrator with all agents including the write agent."""
        # Create agents BEFORE calling super().__init__
        _guardrails = guardrails_agent or create_guardrails_agent()
        _customer_context = customer_context_agent or create_customer_context_agent()
        _rag = rag_agent or create_rag_agent()
        _response = response_agent or create_response_agent()
        _write = write_agent or create_write_agent()

        # Create sub_agents list for BaseAgent
        sub_agents_list = [_guardrails, _customer_context, _rag, _response, _write]

        # Initialize BaseAgent
        super().__init__(
            name=name,
            sub_agents=sub_agents_list,
        )
        
        # Now set private attributes AFTER super().__init__
        self._guardrails = _guardrails
        self._customer_context = _customer_context
        self._rag = _rag
        self._response = _response
        self._write = _write
        self._response_handler = ResponseHandler(agent_name=name)

        logger.info(f"Orchestrator initialized with deterministic agents")
    
    # Properties to access agents (avoids Pydantic field serialization issues)
    @property
    def guardrails_agent(self):
        return self._guardrails
    
    @property
    def customer_context_agent(self):
        return self._customer_context
    
    @property
    def rag_agent(self):
        return self._rag
    
    @property
    def response_agent(self):
        return self._response
    
    @property
    def write_agent(self):
        return self._write
    
    @property
    def response_handler(self):
        return self._response_handler

    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """Orchestrates the conversation flow based on authentication status."""

        logger.info(f"[{self.name}] Starting orchestration")

        # Extract Dialogflow CX session parameters
        cx_params = self._extract_cx_parameters(ctx)
        if cx_params:
            logger.info(f"[{self.name}] CX Parameters: {cx_params}")
            ctx.session.state["cx_parameters"] = cx_params

        # Store original query
        original_query = self._extract_user_query(ctx)
        ctx.session.state["original_query"] = original_query
        logger.info(f"[{self.name}] Query: {original_query[:100]}...")

        # Run Guardrails Agent
        logger.info(f"[{self.name}] → Running Guardrails Agent")
        async for event in self.guardrails_agent.run_async(ctx):
            yield event

        guardrails_result = self._parse_guardrails_result(ctx)
        action = guardrails_result.get("action", "allow")
        category = guardrails_result.get("category", "other")
        is_authenticated = guardrails_result.get("is_authenticated", False)

        logger.info(
            f"[{self.name}] Guardrails: action={action}, category={category}, "
            f"authenticated={is_authenticated}"
        )

        # Route based on guardrails
        if action == "block":
            async for event in self._handle_blocked_route(ctx, category, guardrails_result):
                yield event
            return

        if action == "escalate":
            async for event in self._handle_escalation(ctx, category, guardrails_result):
                yield event
            return

        if category in ["greeting", "thanks", "goodbye"]:
            async for event in self._handle_simple_response(ctx, category):
                yield event
            return

        # Decide between authenticated and guest flows
        if is_authenticated:
            logger.info(f"[{self.name}] ═══ AUTHENTICATED FLOW ═══")
            async for event in self._authenticated_flow(ctx, guardrails_result):
                yield event
        else:
            logger.info(f"[{self.name}] ═══ GUEST FLOW ═══")
            async for event in self._guest_flow(ctx):
                yield event

    async def _authenticated_flow(
        self,
        ctx: InvocationContext,
        guardrails_result: dict
    ) -> AsyncGenerator[Event, None]:
        """
        Authenticated customer flow with long-term memory.
        Steps: Customer Context -> RAG -> Personalized Response -> Write
        """

        # Extract customer identifiers from guardrails
        customer_id = guardrails_result.get("customer_id")
        caller_name = guardrails_result.get("caller_name")

        logger.info(
            f"[{self.name}] Customer identifiers: "
            f"id={customer_id}, name={caller_name}"
        )

        # Store for customer context agent to use
        ctx.session.state["customer_id"] = customer_id
        ctx.session.state["caller_name"] = caller_name

        # Run Customer Context Agent
        logger.info(f"[{self.name}] → Running Customer Context Agent")
        async for event in self.customer_context_agent.run_async(ctx):
            yield event

        customer_context = ctx.session.state.get("customer_context", {})
        
        if isinstance(customer_context, str):
            try:
                import json
                customer_context = json.loads(customer_context)
            except:
                customer_context = {}
        
        customer_found = customer_context.get("customer_found", False)

        if not customer_found:
            logger.warning(
                f"[{self.name}] Customer not found. Falling back to guest flow."
            )
            # Fall back to guest flow if lookup fails
            async for event in self._guest_flow(ctx):
                yield event
            return

        customer_name = customer_context.get("customer_name")
        has_history = customer_context.get("has_previous_interactions", False)
        
        logger.info(
            f"[{self.name}] Customer found: {customer_name}, "
            f"has_history={has_history}"
        )

        # Run RAG Agent
        logger.info(f"[{self.name}] → Running RAG Agent")
        async for event in self.rag_agent.run_async(ctx):
            yield event

        # Run Response Agent (with personalization)
        logger.info(f"[{self.name}] → Running Response Agent (Personalized)")
        ctx.session.state["personalize"] = True
        async for event in self.response_agent.run_async(ctx):
            yield event

        # Run Write Agent (save to long-term memory)
        logger.info(f"[{self.name}] → Running Write Agent (save conversation)")
        async for event in self.write_agent.run_async(ctx):
            # Write agent events are internal, don't yield to user
            pass

        logger.info(f"[{self.name}] Authenticated flow complete")

    async def _guest_flow(
        self,
        ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """
        Guest user flow.
        Steps: RAG -> Generic Response -> Write
        """

        logger.info(f"[{self.name}] Guest user - no customer context needed")

        # Set guest mode flag
        ctx.session.state["customer_context"] = {
            "is_authenticated": False,
            "customer_found": False,
            "has_previous_interactions": False,
            "summary": "Guest user - provide generic information"
        }

        # Run RAG Agent
        logger.info(f"[{self.name}] → Running RAG Agent")
        async for event in self.rag_agent.run_async(ctx):
            yield event

        # Run Response Agent (generic, no personalization)
        logger.info(f"[{self.name}] → Running Response Agent (Generic)")
        ctx.session.state["personalize"] = False
        async for event in self.response_agent.run_async(ctx):
            yield event

        # Run Write Agent (for analytics)
        logger.info(f"[{self.name}] → Running Write Agent (telemetry only)")
        async for event in self.write_agent.run_async(ctx):
            pass  # Internal, don't yield

        logger.info(f"[{self.name}] Guest flow complete")

    def _extract_cx_parameters(self, ctx: InvocationContext) -> dict:
        """Extract customer parameters from Dialogflow CX session"""
        cx_params = {}
        state = ctx.session.state

        param_keys = [
            "customer_id",
            "caller_name",
            "phone_number",
            "ani",
            "dnis"
        ]

        for key in param_keys:
            if key in state and state[key]:
                cx_params[key] = state[key]

        return cx_params

    async def _handle_blocked_route(
        self, ctx: InvocationContext, category: str, guardrails_result: dict
    ) -> AsyncGenerator[Event, None]:
        """Handle blocked queries"""
        logger.info(f"[{self.name}] → BLOCKED: {guardrails_result.get('reason')}")

        template = ResponseTemplates.get_blocked_message(category)
        message = template.get_message()

        event = self.response_handler.create_event(
            message=message,
            metadata={"action": "block", "category": category}
        )
        yield event

        ctx.session.state["final_response"] = self.response_handler.create_final_state(
            decision="respond",
            speech_text=message,
            should_escalate=False,
            follow_up_prompt=template.follow_up
        )

    async def _handle_escalation(
        self, ctx: InvocationContext, category: str, guardrails_result: dict
    ) -> AsyncGenerator[Event, None]:
        """Handle escalations"""
        logger.info(f"[{self.name}] → ESCALATION: {guardrails_result.get('reason')}")

        sentiment = guardrails_result.get("sentiment", "neutral")
        template = ResponseTemplates.get_escalation_message(category, sentiment)
        message = template.get_message()

        event = self.response_handler.create_event(
            message=message,
            metadata={"action": "escalate", "category": category}
        )
        yield event

        ctx.session.state["final_response"] = self.response_handler.create_final_state(
            decision="escalate",
            speech_text=message,
            should_escalate=True,
            escalation_reason=guardrails_result.get("reason")
        )

    async def _handle_simple_response(
        self, ctx: InvocationContext, category: str
    ) -> AsyncGenerator[Event, None]:
        """Handle greetings/thanks/goodbye"""
        logger.info(f"[{self.name}] → Simple response: {category}")

        template = ResponseTemplates.get_greeting_message(category)
        message = template.get_message(use_variation=True)

        event = self.response_handler.create_event(
            message=message,
            metadata={"action": "simple_response", "category": category}
        )
        yield event

        ctx.session.state["final_response"] = self.response_handler.create_final_state(
            decision="respond",
            speech_text=message,
            should_escalate=False,
            follow_up_prompt=template.follow_up if category == "greeting" else None
        )

    def _extract_user_query(self, ctx: InvocationContext) -> str:
        """Extract user query from context"""
        try:
            if ctx.user_content and ctx.user_content.parts:
                return ctx.user_content.parts[0].text or ""
        except Exception as e:
            logger.warning(f"Could not extract user query: {e}")
        return ""

    def _parse_guardrails_result(self, ctx: InvocationContext) -> dict:
        """Parse guardrails result from session state"""
        raw_result = ctx.session.state.get("guardrails_result", "{}")

        try:
            if isinstance(raw_result, dict):
                return raw_result
            result = json.loads(str(raw_result))
            return result
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse guardrails result: {e}")
            return {
                "action": "escalate",
                "category": "out_of_scope",
                "is_authenticated": False
            }


# Create the root_agent instance
root_agent = Orchestrator()