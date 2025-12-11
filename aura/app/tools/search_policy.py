"""
Search Tool - Wraps VertexAiSearchTool for ADK
"""

import logging
from google.adk.tools import VertexAiSearchTool
from app.config import settings

logger = logging.getLogger(__name__)


def build_datastore_path() -> str:
    """
    Build the fully-qualified Vertex AI Search datastore path.
    
    Format: projects/{project}/locations/{location}/collections/default_collection/dataStores/{datastore_id}
    
    NOTE: Vertex AI Search datastores are typically in 'global' location,
    not regional locations like us-central1.
    """
    if not settings.GCP_PROJECT_ID:
        raise ValueError("GCP_PROJECT_ID is not set in environment.")
    if not settings.VERTEX_SEARCH_DATASTORE_ID:
        raise ValueError("VERTEX_SEARCH_DATASTORE_ID is not set in environment.")

    # Use VERTEX_SEARCH_LOCATION if set, otherwise default to 'us'
    # Vertex AI Search datastores can be in 'global', 'us', or 'eu'
    search_location = getattr(settings, 'VERTEX_SEARCH_LOCATION', None) or 'us'

    datastore_path = (
        f"projects/{settings.GCP_PROJECT_ID}/"
        f"locations/{search_location}/"
        "collections/default_collection/"
        f"dataStores/{settings.VERTEX_SEARCH_DATASTORE_ID}"
    )

    logger.info(f"Vertex AI Search datastore path: {datastore_path}")
    return datastore_path


def get_search_tool() -> VertexAiSearchTool:
    """
    Factory function to create the Search tool.
    
    This tool allows the RAG agent to search documents
    using Vertex AI Search with grounding.
    
    Returns:
        VertexAiSearchTool configured for the datastore
    """
    datastore_path = build_datastore_path()
    
    tool = VertexAiSearchTool(data_store_id=datastore_path)
    
    logger.info("Search tool created successfully")
    return tool