"""
Application Configuration
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from functools import lru_cache

# Load .env file from the project root
from dotenv import load_dotenv
_env_path = Path(__file__).parent.parent / ".env"
load_dotenv(_env_path)


class Settings(BaseSettings):
    """Application settings loaded from environment."""
    
    # GCP Settings - support both naming conventions
    GCP_PROJECT_ID: str = ""
    GCP_LOCATION: str = "global"  # For Vertex AI Search datastore
    GOOGLE_CLOUD_PROJECT: str = ""
    GOOGLE_CLOUD_LOCATION: str = "us-central1"  # For Gemini
    
    # Vertex AI Search
    VERTEX_SEARCH_DATASTORE_ID: str = ""
    
    # Model Settings
    MODEL_GUARDRAILS: str = "gemini-2.0-flash"  # Fast for classification
    MODEL_RAG: str = "gemini-2.0-flash"         # Smart for RAG
    MODEL_RESPONSE: str = "gemini-2.0-flash"    # Fast for formatting
    MODEL_NAME: str = "gemini-2.0-flash"        # Default model
    
    # ADK Settings
    GOOGLE_GENAI_USE_VERTEXAI: str = "TRUE"
    
    # App Settings
    APP_NAME: str = "adk-agent"
    LOG_LEVEL: str = "INFO"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Fallback: use GOOGLE_CLOUD_PROJECT if GCP_PROJECT_ID not set
        if not self.GCP_PROJECT_ID and self.GOOGLE_CLOUD_PROJECT:
            object.__setattr__(self, 'GCP_PROJECT_ID', self.GOOGLE_CLOUD_PROJECT)
        # Fallback: use GOOGLE_CLOUD_LOCATION if GCP_LOCATION not set (for non-global)
        if self.GCP_LOCATION == "global" and self.GOOGLE_CLOUD_LOCATION:
            # Keep global for Vertex Search, but this allows override
            pass
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()