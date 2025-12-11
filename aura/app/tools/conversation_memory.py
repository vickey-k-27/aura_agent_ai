"""
Conversation Memory Tool - Long-term memory for "pick up where we left off"

Stores and retrieves conversation history per user from a database.
Collection: conversation_history/{user_id}

This enables:
- "Welcome back, I see you called about X last time"
- Tracking sentiment trends
- Identifying unresolved issues
- Agent handoff context
"""

from google.cloud import firestore
from datetime import datetime, timezone
import logging
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

# Global Firestore client
_firestore_client = None

COLLECTION_NAME = "conversation_history"
MAX_STORED_INTERACTIONS = 20  # Keep last 20 interactions per customer


def _get_firestore_client(project_id: str) -> firestore.Client:
    """Get or create Firestore client."""
    global _firestore_client
    if _firestore_client is None:
        _firestore_client = firestore.Client(project=project_id)
        logger.info(f"Initialized Firestore client for project: {project_id}")
    return _firestore_client


def fetch_conversation_history(
    project_id: str,
    user_id: str,
    limit: int = 10
) -> dict:
    """
    Fetch conversation history for a user.
    
    Use this tool to get previous interactions with a user.
    This enables personalized "welcome back" experiences and
    continuity across calls.
    
    Args:
        project_id: GCP project ID
        user_id: User ID (e.g., "USER-001")
        limit: Max number of past interactions to retrieve (default 10)
    
    Returns:
        dict with conversation history and analytics:
        - has_history: bool
        - total_calls: int
        - last_interactions: list of recent conversations
        - last_topic: str (most recent topic discussed)
        - unresolved_issues: list of pending matters
        - sentiment_trend: "improving", "stable", "declining"
        - last_interaction_date: ISO timestamp
    
    Example:
        fetch_conversation_history(
            project_id="my-project",
            user_id="USER-001",
            limit=5
        )
    """
    try:
        db = _get_firestore_client(project_id)
        doc_ref = db.collection(COLLECTION_NAME).document(user_id)
        doc = doc_ref.get()
        
        if not doc.exists:
            logger.info(f"No conversation history for user: {user_id}")
            return {
                "success": True,
                "has_history": False,
                "total_calls": 0,
                "last_interactions": [],
                "last_topic": None,
                "unresolved_issues": [],
                "sentiment_trend": "unknown",
                "last_interaction_date": None,
                "summary": f"No previous interaction history found for {user_id}. This appears to be a new user or their first call."
            }
        
        data = doc.to_dict()
        interactions = data.get("interactions", [])
        
        # Get last N interactions (most recent first)
        recent = interactions[-limit:] if len(interactions) > limit else interactions
        recent.reverse()  # Most recent first
        
        # Calculate sentiment trend
        sentiment_trend = _calculate_sentiment_trend(interactions)
        
        # Get unresolved issues
        unresolved = data.get("unresolved_issues", [])
        
        # Get last topic
        last_topic = recent[0].get("topic") if recent else None
        last_date = recent[0].get("timestamp") if recent else None
        
        logger.info(
            f"Found {len(interactions)} total interactions for {user_id}. "
            f"Sentiment trend: {sentiment_trend}"
        )
        
        # Build summary
        summary_parts = []
        if len(interactions) > 0:
            summary_parts.append(f"User has {len(interactions)} previous interaction(s).")
        if last_topic:
            summary_parts.append(f"Last discussed: {last_topic}.")
        if unresolved:
            summary_parts.append(f"Unresolved issues: {', '.join(unresolved)}.")
        if sentiment_trend == "declining":
            summary_parts.append("Sentiment has been declining - handle with care.")
        
        return {
            "success": True,
            "has_history": True,
            "total_calls": len(interactions),
            "last_interactions": recent,
            "last_topic": last_topic,
            "unresolved_issues": unresolved,
            "sentiment_trend": sentiment_trend,
            "last_interaction_date": last_date,
            "summary": " ".join(summary_parts) if summary_parts else "Returning user."
        }
        
    except Exception as e:
        logger.error(f"Error fetching conversation history: {e}", exc_info=True)
        return {
            "success": False,
            "has_history": False,
            "error": f"Failed to fetch history: {str(e)}"
        }


def save_conversation(
    project_id: str,
    user_id: str,
    session_id: str,
    query: str,
    response: str,
    topic: str,
    sentiment: str,
    was_escalated: bool = False,
    escalation_reason: Optional[str] = None,
    resolution_status: str = "resolved",
    metadata: Optional[Dict[str, Any]] = None
) -> dict:
    """
    Save a conversation interaction to long-term memory.
    
    Call this at the END of each conversation turn to build history.
    
    Args:
        project_id: GCP project ID
        user_id: User ID
        session_id: Current session ID (for grouping turns in same call)
        query: User's question/statement
        response: Agent's response
        topic: Topic category (e.g., "billing", "support", "sales")
        sentiment: Detected sentiment ("positive", "neutral", "negative", "frustrated")
        was_escalated: Whether this was escalated to human
        escalation_reason: Why it was escalated (if applicable)
        resolution_status: "resolved", "pending", "escalated"
        metadata: Additional data to store (analytics, citations, etc.)
    
    Returns:
        dict with success status
    
    Example:
        save_conversation(
            project_id="my-project",
            user_id="USER-001",
            session_id="session-123",
            query="How do I reset my password?",
            response="You can reset your password by clicking...",
            topic="password_reset",
            sentiment="neutral",
            resolution_status="resolved"
        )
    """
    try:
        db = _get_firestore_client(project_id)
        doc_ref = db.collection(COLLECTION_NAME).document(user_id)
        
        # Create interaction record
        interaction = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "query": query[:500],  # Truncate long queries
            "response": response[:1000],  # Truncate long responses
            "topic": topic,
            "sentiment": sentiment,
            "was_escalated": was_escalated,
            "escalation_reason": escalation_reason,
            "resolution_status": resolution_status,
        }
        
        if metadata:
            interaction["metadata"] = metadata
        
        # Get existing doc or create new
        doc = doc_ref.get()
        
        if doc.exists:
            data = doc.to_dict()
            interactions = data.get("interactions", [])
            interactions.append(interaction)
            
            # Trim to max size
            if len(interactions) > MAX_STORED_INTERACTIONS:
                interactions = interactions[-MAX_STORED_INTERACTIONS:]
            
            # Update unresolved issues
            unresolved = data.get("unresolved_issues", [])
            if resolution_status == "pending" and topic not in unresolved:
                unresolved.append(topic)
            elif resolution_status == "resolved" and topic in unresolved:
                unresolved.remove(topic)
            
            doc_ref.update({
                "interactions": interactions,
                "unresolved_issues": unresolved,
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "total_interactions": len(interactions),
            })
        else:
            # Create new document
            unresolved = [topic] if resolution_status == "pending" else []
            doc_ref.set({
                "user_id": user_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "interactions": [interaction],
                "unresolved_issues": unresolved,
                "total_interactions": 1,
            })
        
        logger.info(f"Saved conversation for user {user_id}, topic: {topic}")
        
        return {
            "success": True,
            "message": f"Conversation saved for {user_id}"
        }
        
    except Exception as e:
        logger.error(f"Error saving conversation: {e}", exc_info=True)
        return {
            "success": False,
            "error": f"Failed to save conversation: {str(e)}"
        }


def _calculate_sentiment_trend(interactions: List[dict]) -> str:
    """Calculate sentiment trend from recent interactions."""
    if len(interactions) < 2:
        return "unknown"
    
    # Get last 5 sentiments
    recent_sentiments = [
        i.get("sentiment", "neutral") 
        for i in interactions[-5:]
    ]
    
    # Score sentiments
    scores = {
        "positive": 2,
        "neutral": 1,
        "negative": -1,
        "frustrated": -2
    }
    
    # Calculate trend
    recent_scores = [scores.get(s, 0) for s in recent_sentiments]
    
    if len(recent_scores) >= 3:
        first_half = sum(recent_scores[:len(recent_scores)//2])
        second_half = sum(recent_scores[len(recent_scores)//2:])
        
        if second_half > first_half + 1:
            return "improving"
        elif second_half < first_half - 1:
            return "declining"
    
    return "stable"


def get_agent_handoff_context(
    project_id: str,
    user_id: str,
    current_session_summary: Optional[str] = None
) -> dict:
    """
    Get comprehensive context for human agent handoff.
    
    Call this when escalating to human agent to provide them with
    full context: user info, conversation history, current issue.
    
    Args:
        project_id: GCP project ID
        user_id: User ID
        current_session_summary: Summary of current conversation
    
    Returns:
        dict with all context needed for agent handoff screen
    """
    try:
        # Fetch full history
        history = fetch_conversation_history(project_id, user_id, limit=10)
        
        # Fetch user data
        from app.tools.fetch_customer import fetch_customer
        user = fetch_customer(project_id, customer_id=user_id)
        
        handoff = {
            "user_id": user_id,
            "user_name": user.get("name") if user.get("success") else "Unknown",
            "user_tier": user.get("tier") if user.get("success") else "Unknown",
            
            # Conversation context
            "total_previous_calls": history.get("total_calls", 0),
            "sentiment_trend": history.get("sentiment_trend", "unknown"),
            "unresolved_issues": history.get("unresolved_issues", []),
            "last_topic": history.get("last_topic"),
            
            # Current session
            "current_session_summary": current_session_summary,
            
            # Recent history for agent to review
            "recent_interactions": history.get("last_interactions", [])[:5],
            
            # Recommendations for agent
            "agent_notes": _generate_agent_notes(history, user)
        }
        
        return {
            "success": True,
            "handoff_context": handoff
        }
        
    except Exception as e:
        logger.error(f"Error getting handoff context: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }


def _generate_agent_notes(history: dict, user: dict) -> List[str]:
    """Generate helpful notes for human agent."""
    notes = []
    
    if history.get("sentiment_trend") == "declining":
        notes.append("⚠️ User sentiment declining - approach with empathy")
    
    if history.get("total_calls", 0) > 3:
        notes.append(f"📞 Frequent caller ({history.get('total_calls')} calls) - consider VIP handling")
    
    if history.get("unresolved_issues"):
        notes.append(f"🔴 Has unresolved issues: {', '.join(history.get('unresolved_issues'))}")
    
    if user.get("tier") == "Premium":
        notes.append("⭐ Premium tier user")
    
    return notes