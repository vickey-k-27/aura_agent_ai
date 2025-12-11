"""
User Lookup Tool - Corrected for ADK
Simple function tool that fetches user data from a database

ADK automatically converts functions to tools - no base class needed!
Reference: https://google.github.io/adk-docs/tools/
"""

from google.cloud import firestore
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Initialize Firestore client globally
_firestore_client = None

def _get_firestore_client(project_id: str) -> firestore.Client:
    """Get or create Firestore client"""
    global _firestore_client
    if _firestore_client is None:
        _firestore_client = firestore.Client(project=project_id)
        logger.info(f"Initialized Firestore client for project: {project_id}")
    return _firestore_client


def fetch_user(
    project_id: str,
    user_id: Optional[str] = None
) -> dict:
    """
    Fetch user profile and details from a database.

    Use this tool when you need to get user information for personalization.
    You can lookup by user_id.

    Args:
        project_id: GCP project ID (usually from settings)
        user_id: User ID like "USER-001" (optional)

    Returns:
        dict: User data with success status or error message

    Example:
        fetch_user(project_id="my-project", user_id="USER-001")
    """
    try:
        db = _get_firestore_client(project_id)

        # Lookup by user_id
        if user_id:
            logger.info(f"Fetching user by ID: {user_id}")
            doc = db.collection('users').document(user_id).get()

            if doc.exists:
                data = doc.to_dict()
                logger.info(f"Found user: {data.get('name')}")
                return {
                    "success": True,
                    "user": data,
                    "user_id": user_id,
                    "name": data.get('name'),
                    "tier": data.get('tier')
                }
            else:
                logger.warning(f"User not found: {user_id}")
                return {
                    "success": False,
                    "error": f"User {user_id} not found in database"
                }

        return {
            "success": False,
            "error": "Must provide user_id"
        }

    except Exception as e:
        logger.error(f"Error fetching user: {e}", exc_info=True)
        return {
            "success": False,
            "error": f"Database error: {str(e)}"
        }