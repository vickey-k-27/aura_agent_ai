"""
Telemetry Tool - Logs conversation metrics to BigQuery

Stores:
- Conversation timing
- Classification accuracy
- Sentiment data
- Escalation patterns
- Resolution rates

Table: {project}.adk_agent_analytics.conversation_telemetry
"""

from google.cloud import bigquery
from datetime import datetime, timezone
import logging
from typing import Optional, Dict, Any
import json

logger = logging.getLogger(__name__)

# Global BigQuery client
_bigquery_client = None

DATASET_ID = "adk_agent_analytics"
TABLE_ID = "conversation_telemetry"


def _get_bigquery_client(project_id: str) -> bigquery.Client:
    """Get or create BigQuery client."""
    global _bigquery_client
    if _bigquery_client is None:
        _bigquery_client = bigquery.Client(project=project_id)
        logger.info(f"Initialized BigQuery client for project: {project_id}")
    return _bigquery_client


def _ensure_table_exists(client: bigquery.Client, project_id: str):
    """Create telemetry table if it doesn't exist."""
    table_ref = f"{project_id}.{DATASET_ID}.{TABLE_ID}"
    
    schema = [
        bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("session_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("user_id", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("is_authenticated", "BOOLEAN", mode="REQUIRED"),
        
        # Classification
        bigquery.SchemaField("category", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("sentiment", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("topic", "STRING", mode="NULLABLE"),
        
        # Outcome
        bigquery.SchemaField("action_taken", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("was_escalated", "BOOLEAN", mode="REQUIRED"),
        bigquery.SchemaField("escalation_reason", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("resolution_status", "STRING", mode="NULLABLE"),
        
        # Timing (milliseconds)
        bigquery.SchemaField("total_latency_ms", "INTEGER", mode="NULLABLE"),
        bigquery.SchemaField("guardrails_latency_ms", "INTEGER", mode="NULLABLE"),
        bigquery.SchemaField("rag_latency_ms", "INTEGER", mode="NULLABLE"),
        bigquery.SchemaField("response_latency_ms", "INTEGER", mode="NULLABLE"),
        
        # Quality
        bigquery.SchemaField("confidence_score", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("rag_sources_count", "INTEGER", mode="NULLABLE"),
        
        # Context
        bigquery.SchemaField("query_length", "INTEGER", mode="NULLABLE"),
        bigquery.SchemaField("response_length", "INTEGER", mode="NULLABLE"),
        bigquery.SchemaField("flow_type", "STRING", mode="NULLABLE"),  # "authenticated" or "guest"
        
        # Metadata (JSON string)
        bigquery.SchemaField("metadata", "STRING", mode="NULLABLE"),
    ]
    
    try:
        table = bigquery.Table(table_ref, schema=schema)
        table = client.create_table(table, exists_ok=True)
        logger.info(f"Telemetry table ready: {table_ref}")
    except Exception as e:
        logger.warning(f"Could not create table (may already exist): {e}")


def log_telemetry(
    project_id: str,
    session_id: str,
    user_id: Optional[str] = None,
    is_authenticated: bool = False,
    category: Optional[str] = None,
    sentiment: Optional[str] = None,
    topic: Optional[str] = None,
    action_taken: Optional[str] = None,
    was_escalated: bool = False,
    escalation_reason: Optional[str] = None,
    resolution_status: Optional[str] = None,
    total_latency_ms: Optional[int] = None,
    guardrails_latency_ms: Optional[int] = None,
    rag_latency_ms: Optional[int] = None,
    response_latency_ms: Optional[int] = None,
    confidence_score: Optional[float] = None,
    rag_sources_count: Optional[int] = None,
    query_length: Optional[int] = None,
    response_length: Optional[int] = None,
    flow_type: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> dict:
    """
    Log conversation telemetry to BigQuery for analytics.
    
    Use this tool at the end of each conversation to track:
    - Performance metrics (latency)
    - Quality metrics (confidence, sources)
    - Outcome metrics (escalation, resolution)
    
    Args:
        project_id: GCP project ID
        session_id: Unique session identifier
        user_id: User ID (if authenticated)
        is_authenticated: Whether user was authenticated
        category: Query category from guardrails
        sentiment: Detected sentiment
        topic: Topic classification
        action_taken: What action was taken (respond, escalate, block)
        was_escalated: Whether escalated to human
        escalation_reason: Why escalated
        resolution_status: resolved/pending/escalated
        total_latency_ms: Total response time
        guardrails_latency_ms: Guardrails agent time
        rag_latency_ms: RAG agent time
        response_latency_ms: Response agent time
        confidence_score: Classification confidence
        rag_sources_count: Number of sources cited
        query_length: Length of user query
        response_length: Length of response
        flow_type: "authenticated" or "guest"
        metadata: Additional JSON metadata
    
    Returns:
        dict with success status
    
    Example:
        log_telemetry(
            project_id="my-project",
            session_id="sess-123",
            user_id="USER-001",
            is_authenticated=True,
            category="user_question",
            sentiment="neutral",
            topic="password_reset",
            was_escalated=False,
            resolution_status="resolved",
            total_latency_ms=1250
        )
    """
    try:
        client = _get_bigquery_client(project_id)
        
        # Ensure table exists
        _ensure_table_exists(client, project_id)
        
        table_ref = f"{project_id}.{DATASET_ID}.{TABLE_ID}"
        
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "user_id": user_id,
            "is_authenticated": is_authenticated,
            "category": category,
            "sentiment": sentiment,
            "topic": topic,
            "action_taken": action_taken,
            "was_escalated": was_escalated,
            "escalation_reason": escalation_reason,
            "resolution_status": resolution_status,
            "total_latency_ms": total_latency_ms,
            "guardrails_latency_ms": guardrails_latency_ms,
            "rag_latency_ms": rag_latency_ms,
            "response_latency_ms": response_latency_ms,
            "confidence_score": confidence_score,
            "rag_sources_count": rag_sources_count,
            "query_length": query_length,
            "response_length": response_length,
            "flow_type": flow_type,
            "metadata": json.dumps(metadata) if metadata else None,
        }
        
        errors = client.insert_rows_json(table_ref, [row])
        
        if errors:
            logger.error(f"BigQuery insert errors: {errors}")
            return {
                "success": False,
                "error": f"Insert failed: {errors}"
            }
        
        logger.info(f"Telemetry logged for session: {session_id}")
        return {
            "success": True,
            "message": f"Telemetry logged for session {session_id}"
        }
        
    except Exception as e:
        logger.error(f"Error logging telemetry: {e}", exc_info=True)
        return {
            "success": False,
            "error": f"Failed to log telemetry: {str(e)}"
        }


def get_user_analytics(
    project_id: str,
    user_id: str,
    days: int = 30
) -> dict:
    """
    Get analytics summary for a specific user.
    
    Use this to understand user patterns:
    - How often they call
    - Common topics
    - Sentiment trends
    - Escalation history
    
    Args:
        project_id: GCP project ID
        user_id: User ID
        days: Number of days to analyze (default 30)
    
    Returns:
        dict with user analytics
    """
    try:
        client = _get_bigquery_client(project_id)
        
        query = f"""
        SELECT
            COUNT(*) as total_conversations,
            COUNTIF(was_escalated) as escalation_count,
            AVG(total_latency_ms) as avg_latency_ms,
            APPROX_TOP_COUNT(topic, 3) as top_topics,
            APPROX_TOP_COUNT(sentiment, 3) as sentiment_distribution,
            MAX(timestamp) as last_interaction
        FROM `{project_id}.{DATASET_ID}.{TABLE_ID}`
        WHERE user_id = @user_id
          AND timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @days DAY)
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
                bigquery.ScalarQueryParameter("days", "INT64", days),
            ]
        )
        
        results = client.query(query, job_config=job_config).result()
        rows = [dict(row) for row in results]
        
        if rows:
            return {
                "success": True,
                "user_id": user_id,
                "period_days": days,
                "analytics": rows[0]
            }
        else:
            return {
                "success": True,
                "user_id": user_id,
                "analytics": None,
                "message": "No data found for this user"
            }
        
    except Exception as e:
        logger.error(f"Error getting user analytics: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }