"""
Guardrails Models - Pydantic models for classification results
"""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class GuardrailAction(str, Enum):
    """Action to take based on guardrails classification."""
    ALLOW = "allow"       # Safe to process with RAG
    ESCALATE = "escalate" # Route to human agent
    BLOCK = "block"       # Reject/redirect


class GuardrailCategory(str, Enum):
    """Query category classification."""
    USER_QUESTION = "user_question"   # General user queries
    CLAIMS_INQUIRY = "claims_inquiry"     # Claims-related
    COMPLAINT = "complaint"               # User complaint
    URGENT_LEGAL = "urgent_legal"         # Legal/regulatory threats
    OUT_OF_SCOPE = "out_of_scope"         # Not account related
    SECURITY_RISK = "security_risk"       # Malicious input
    GREETING = "greeting"                 # Hello, hi, etc.
    THANKS = "thanks"                     # Thank you
    GOODBYE = "goodbye"                   # Bye, end conversation


class SentimentLabel(str, Enum):
    """User sentiment."""
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    FRUSTRATED = "frustrated"


class GuardrailsResult(BaseModel):
    """Result from guardrails classification."""
    
    category: GuardrailCategory = Field(
        description="Classification of the query type"
    )
    action: GuardrailAction = Field(
        description="Recommended action to take"
    )
    sentiment: SentimentLabel = Field(
        default=SentimentLabel.NEUTRAL,
        description="Detected user sentiment"
    )
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Classification confidence score"
    )
    reason: str = Field(
        description="Brief explanation of classification"
    )
    detected_issues: List[str] = Field(
        default_factory=list,
        description="List of detected risk indicators"
    )
    escalation_priority: str = Field(
        default="normal",
        description="Priority level: normal, high, urgent"
    )
    
    class Config:
        use_enum_values = True


class RAGResult(BaseModel):
    """Result from RAG agent."""
    
    answer: str = Field(
        description="The grounded answer from documents"
    )
    confidence: str = Field(
        default="MEDIUM",
        description="Confidence level: HIGH, MEDIUM, LOW, NONE"
    )
    citations: List[str] = Field(
        default_factory=list,
        description="Source citations from documents"
    )
    has_answer: bool = Field(
        default=True,
        description="Whether a relevant answer was found"
    )


class ResponseDecision(str, Enum):
    """Decision for final response."""
    RESPOND = "respond"           # Give the answer
    ESCALATE = "escalate"         # Transfer to human
    CLARIFY = "clarify"           # Ask clarifying question


class FinalResponse(BaseModel):
    """Final response to send to user."""
    
    decision: ResponseDecision = Field(
        description="What action to take"
    )
    speech_text: str = Field(
        description="Voice-friendly response text"
    )
    should_escalate: bool = Field(
        default=False,
        description="Whether to transfer to human"
    )
    escalation_reason: Optional[str] = Field(
        default=None,
        description="Reason for escalation if applicable"
    )
    follow_up_prompt: Optional[str] = Field(
        default=None,
        description="Suggested follow-up question"
    )
    
    class Config:
        use_enum_values = True