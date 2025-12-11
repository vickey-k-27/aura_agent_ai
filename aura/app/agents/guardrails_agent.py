"""
Enhanced Guardrails Agent with Dialogflow CX Integration
Detects user identity from multiple sources

"""

import logging
from pathlib import Path
from google.adk.agents import LlmAgent
from pydantic import BaseModel
from typing import Optional, List

from app.config import settings

logger = logging.getLogger(__name__)


class GuardrailsResult(BaseModel):
    """Enhanced guardrails output with identity detection"""
    category: str
    action: str
    sentiment: str
    confidence: float
    reason: str
    detected_issues: List[str]
    escalation_priority: str

    # NEW: Identity detection fields
    is_authenticated: bool
    user_id: Optional[str] = None
    caller_name: Optional[str] = None
    phone_number: Optional[str] = None


def load_guardrails_prompt() -> str:
    """Load guardrails prompt with identity detection"""
    prompt_paths = [
        Path(__file__).parent.parent / "prompts" / "guardrails.txt",
        Path("app/prompts/guardrails.txt"),
    ]
    
    for path in prompt_paths:
        if path.exists():
            logger.info(f"Loaded guardrails prompt from: {path}")
            return path.read_text().strip()
    
    logger.warning("guardrails.txt not found, using default")
    return get_default_guardrails_prompt()


def get_default_guardrails_prompt() -> str:
    return """You are a Guardrails Agent with identity detection.

Your job:
1. Classify the query intent
2. Detect user identity
3. Determine routing

## Identity Detection

Check MULTIPLE sources for user identity:

Source 1: Session parameters (from Dialogflow CX)
- Look for: user_id, caller_name, phone_number
- These come from CTI or CX session state

Source 2: Query text
- "My user ID is 12345"
- "I'm John Smith"  
- "My account", "My details" (personal pronouns)

Source 3: Context
- Previous turns in conversation
- User already authenticated

If ANY source has identity data:
- is_authenticated: true
- Extract: user_id, caller_name, phone_number

If NO identity data:
- is_authenticated: false
- All identity fields: null
ALWAYS set is_authenticated: true when user ID is found!

## Output JSON

{
  "category": "user_question",
  "action": "allow",
  "sentiment": "neutral",
  "confidence": 0.95,
  "reason": "User asking about their account",
  "detected_issues": [],
  "escalation_priority": "normal",
  "is_authenticated": true,
  "user_id": "CUST-001",
  "caller_name": "John Smith",
  "phone_number": "+447700900001"
}

Prioritize session parameters over text detection."""

# In guardrails_agent.py - Replace create_dynamic_instruction()
def create_dynamic_instruction(ctx):
    """Inject ACTUAL session state into LLM prompt."""
    session_state = ctx.session.state or {}
    
    user_id = session_state.get("user_id")
    caller_name = session_state.get("caller_name")
    
    base_prompt = load_guardrails_prompt()
    
    instruction = base_prompt + f"""
    
CURRENT SESSION STATE (PRIORITY #1):
user_id: {user_id or 'None'}
caller_name: {caller_name or 'None'}

CRITICAL: If ANY of above fields have values → is_authenticated: true
Example: user_id: "12345" → is_authenticated: true"""
    
    return instruction



def create_guardrails_agent() -> LlmAgent:
    """Create enhanced guardrails agent with identity detection"""

    prompt = load_guardrails_prompt()

    agent = LlmAgent(
        name="guardrails_identity_agent",
        model=settings.MODEL_GUARDRAILS,
        instruction=create_dynamic_instruction,
        output_schema=GuardrailsResult,
        output_key="guardrails_result"
    )

    logger.info(f"Guardrails Agent created with model: {settings.MODEL_GUARDRAILS}")
    return agent