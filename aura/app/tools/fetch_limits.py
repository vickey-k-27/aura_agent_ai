"""
Limits Lookup Tool - Corrected for ADK
Simple function tool that fetches tier limits from a database

ADK automatically converts functions to tools!
"""

from google.cloud import bigquery
import logging
from typing import Optional, List

logger = logging.getLogger(__name__)

# Initialize BigQuery client globally
_bigquery_client = None

def _get_bigquery_client(project_id: str) -> bigquery.Client:
    """Get or create BigQuery client"""
    global _bigquery_client
    if _bigquery_client is None:
        _bigquery_client = bigquery.Client(project=project_id)
        logger.info(f"Initialized BigQuery client for project: {project_id}")
    return _bigquery_client


def fetch_limits(
    project_id: str,
    tier_name: Optional[str] = None
) -> dict:
    """
    Get limits for tiers from a database.

    Use this tool to answer questions about limits, excesses,
    and differences between tiers.

    Args:
        project_id: GCP project ID
        tier_name: Tier to fetch - "Basic", "Standard", or "Premium" (optional)
                   If not provided, returns all tiers for comparison

    Returns:
        dict: Limits data or comparison of multiple tiers

    Example:
        fetch_limits(project_id="my-project", tier_name="Premium")
        fetch_limits(project_id="my-project")  # Returns all tiers
    """
    try:
        client = _get_bigquery_client(project_id)

        # Single tier query
        if tier_name:
            logger.info(f"Fetching limits for tier: {tier_name}")

            query = f"""
            SELECT *
            FROM `{project_id}.database.limits`
            WHERE tier_name = @tier_name
            """

            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("tier_name", "STRING", tier_name)
                ]
            )

            results = client.query(query, job_config=job_config).result()
            rows = [dict(row) for row in results]

            if rows:
                logger.info(f"Found limits for {tier_name}")
                return {
                    "success": True,
                    "tier": tier_name,
                    "limits": rows[0]
                }
            else:
                return {
                    "success": False,
                    "error": f"Tier '{tier_name}' not found. Valid tiers: Basic, Standard, Premium"
                }

        # Get all tiers (for comparison)
        else:
            logger.info("Fetching all tiers for comparison")

            query = f"""
            SELECT 
                tier_name,
                tier_description,
                annual_fee_from,
                item_1_max,
                item_2_max,
                single_item_limit,
                excess_standard,
                excess_extra,
                extra_limit,
                other_limit
            FROM `{project_id}.database.limits`
            ORDER BY annual_fee_from
            """

            results = client.query(query).result()
            rows = [dict(row) for row in results]

            if rows:
                logger.info(f"Comparison found {len(rows)} tiers")
                return {
                    "success": True,
                    "all_tiers": rows,
                    "tier_count": len(rows)
                }
            else:
                return {
                    "success": False,
                    "error": "No tiers found in database"
                }

    except Exception as e:
        logger.error(f"Error fetching limits: {e}", exc_info=True)
        return {
            "success": False,
            "error": f"Database error: {str(e)}"
        }