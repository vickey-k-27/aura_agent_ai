"""
Response Templates - Configuration for all system messages

Makes it easy to:
- Update messages without code changes
- Add A/B testing variations
- Support multiple languages
- Maintain consistent tone
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
import random


@dataclass
class ResponseTemplate:
    """Single response template with variations."""
    messages: List[str]  # Multiple variations for natural feel
    follow_up: Optional[str] = None
    tone: str = "neutral"

    def get_message(self, use_variation: bool = True) -> str:
        """Get a message, optionally rotating through variations."""
        if use_variation and len(self.messages) > 1:
            return random.choice(self.messages)
        return self.messages[0]


class ResponseTemplates:
    """
    Central repository for all response templates.

    TODO: Move to YAML/JSON config file or database for production
    """

    # ============================================================
    # BLOCKED RESPONSES
    # ============================================================
    BLOCKED = {
        "out_of_scope": ResponseTemplate(
            messages=[
                "I'm your assistant, so I can help with questions about your account, services, and support. Is there something I can help with?",
                "I specialize in account questions. I'd be happy to help with your account details, or support process. What would you like to know?",
            ],
            follow_up="How can I assist you today?",
            tone="friendly_redirect"
        ),
        "security_risk": ResponseTemplate(
            messages=[
                "I'm sorry, I wasn't able to process that request. How can I help you today?",
            ],
            tone="neutral"
        ),
        "default": ResponseTemplate(
            messages=[
                "I'm not able to help with that particular request, but I'm happy to answer questions about your account. What would you like to know?",
            ],
            tone="polite"
        )
    }

    # ============================================================
    # GREETING RESPONSES
    # ============================================================
    GREETING = ResponseTemplate(
        messages=[
            "Hello! I'm your assistant. I'm here to help with questions about your account, services, and support.",
            "Hi there! Welcome. I'm here to help you with any questions about your account.",
            "Good day! I'm your assistant, ready to help with your account questions.",
        ],
        follow_up="How can I help you today?"
    )

    THANKS = ResponseTemplate(
        messages=[
            "You're welcome! Is there anything else I can help you with regarding your account?",
            "My pleasure! Feel free to ask if you have any other questions about your account.",
        ]
    )

    GOODBYE = ResponseTemplate(
        messages=[
            "Thank you for calling. Take care, and don't hesitate to call back if you have any more questions. Goodbye!",
            "Thanks for reaching out. Have a great day, and feel free to reach out anytime. Goodbye!",
        ]
    )

    # ============================================================
    # ESCALATION RESPONSES
    # ============================================================
    ESCALATION = {
        "urgent_legal": ResponseTemplate(
            messages=[
                "I understand this is an important matter. Let me connect you with a senior member of our team who can properly address your concerns and ensure they're handled with the attention they deserve. Please hold for just a moment.",
            ],
            tone="empathetic_professional"
        ),
        "frustrated": ResponseTemplate(
            messages=[
                "I can hear this has been frustrating, and I'm sorry for that experience. Let me connect you with a specialist who can give this their full attention and work to resolve this for you. Please hold.",
                "I understand your frustration, and I want to make sure we get this resolved properly. Let me transfer you to a colleague who can give this the attention it deserves. One moment please.",
            ],
            tone="empathetic"
        ),
        "default": ResponseTemplate(
            messages=[
                "This is an important query that deserves specialist attention. Let me transfer you to a colleague who can help. Please hold for just a moment.",
            ],
            tone="professional"
        )
    }

    @classmethod
    def get_blocked_message(cls, category: str) -> ResponseTemplate:
        """Get blocked response template for a category."""
        return cls.BLOCKED.get(category, cls.BLOCKED["default"])

    @classmethod
    def get_greeting_message(cls, category: str) -> ResponseTemplate:
        """Get greeting/thanks/goodbye response."""
        if category == "greeting":
            return cls.GREETING
        elif category == "thanks":
            return cls.THANKS
        elif category == "goodbye":
            return cls.GOODBYE
        return cls.GREETING

    @classmethod
    def get_escalation_message(cls, category: str, sentiment: str) -> ResponseTemplate:
        """Get escalation response based on category and sentiment."""
        if category == "urgent_legal":
            return cls.ESCALATION["urgent_legal"]
        elif sentiment == "frustrated":
            return cls.ESCALATION["frustrated"]
        return cls.ESCALATION["default"]