"""
Configuration settings for the European Car Troubleshooting Chatbot API
Specialized for European markets with focus on Lithuania and Baltic region
"""

from typing import List

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # Application Settings
    app_name: str = "Car Troubleshooting Chatbot API"
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=True, env="DEBUG")
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")

    # OpenAI Settings
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")

    @field_validator("openai_api_key")
    @classmethod
    def strip_api_key(cls, v):
        """Strip whitespace from the API key."""
        if isinstance(v, str):
            return v.strip()
        return v

    openai_model: str = Field(default="gpt-4-1106-preview", env="OPENAI_MODEL")
    openai_embedding_model: str = Field(
        default="text-embedding-ada-002", env="OPENAI_EMBEDDING_MODEL"
    )
    openai_temperature: float = Field(default=0.1, env="OPENAI_TEMPERATURE")

    # Vector Store Settings
    vector_store_type: str = Field(default="chroma", env="VECTOR_STORE_TYPE")  # chroma, faiss
    vector_store_path: str = Field(default="./data/vector_store", env="VECTOR_STORE_PATH")
    embedding_dimension: int = Field(default=1536, env="EMBEDDING_DIMENSION")

    # RAG Settings
    chunk_size: int = Field(default=1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, env="CHUNK_OVERLAP")
    retrieval_k: int = Field(default=5, env="RETRIEVAL_K")
    similarity_threshold: float = Field(default=0.6, env="SIMILARITY_THRESHOLD")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # FORCE override to ensure 0.6 is used
        if self.similarity_threshold != 0.6:
            self.similarity_threshold = 0.6

    # Rate Limiting
    rate_limit_requests: int = Field(default=30, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=900, env="RATE_LIMIT_WINDOW")  # 15 minutes

    # CORS Settings
    allowed_origins: List[str] = Field(
        default=[
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "https://localhost:3000",
            "http://localhost:3001",
        ],
        env="ALLOWED_ORIGINS",
    )

    allowed_hosts: List[str] = Field(
        default=["localhost", "127.0.0.1", "0.0.0.0"], env="ALLOWED_HOSTS"
    )

    # Logging Settings
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="logs/app.log", env="LOG_FILE")
    log_rotation: str = Field(default="10 MB", env="LOG_ROTATION")
    log_retention: str = Field(default="1 week", env="LOG_RETENTION")

    # Security Settings
    secret_key: str = Field(default="your-secret-key-change-in-production", env="SECRET_KEY")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")

    # Function Calling Settings
    enable_function_calling: bool = Field(default=True, env="ENABLE_FUNCTION_CALLING")
    function_timeout: int = Field(default=30, env="FUNCTION_TIMEOUT")  # seconds

    # Streaming Settings
    enable_streaming: bool = Field(default=True, env="ENABLE_STREAMING")
    stream_timeout: int = Field(default=60, env="STREAM_TIMEOUT")  # seconds

    # Web Search Settings
    enable_web_search: bool = Field(default=True, env="ENABLE_WEB_SEARCH")
    web_search_timeout: int = Field(default=30, env="WEB_SEARCH_TIMEOUT")  # seconds
    web_search_max_results: int = Field(default=3, env="WEB_SEARCH_MAX_RESULTS")
    web_search_min_kb_results: int = Field(default=2, env="WEB_SEARCH_MIN_KB_RESULTS")

    # Knowledge-First Learning Settings
    enable_intelligent_learning: bool = Field(default=True, env="ENABLE_INTELLIGENT_LEARNING")
    learned_content_threshold: float = Field(
        default=0.5, env="LEARNED_CONTENT_THRESHOLD"
    )  # Lower threshold for learned content
    technical_query_min_results: int = Field(
        default=2, env="TECHNICAL_QUERY_MIN_RESULTS"
    )  # Min results for technical queries
    general_query_min_results: int = Field(
        default=1, env="GENERAL_QUERY_MIN_RESULTS"
    )  # Min results for general queries
    prioritize_learned_content: bool = Field(
        default=True, env="PRIORITIZE_LEARNED_CONTENT"
    )  # Always prioritize learned content
    max_learned_docs_per_query: int = Field(
        default=3, env="MAX_LEARNED_DOCS_PER_QUERY"
    )  # Max learned docs to store per query

    # Knowledge Relevancy Tuning
    high_confidence_threshold: float = Field(
        default=0.75, env="HIGH_CONFIDENCE_THRESHOLD"
    )  # High confidence similarity
    medium_confidence_threshold: float = Field(
        default=0.6, env="MEDIUM_CONFIDENCE_THRESHOLD"
    )  # Medium confidence similarity
    low_confidence_threshold: float = Field(
        default=0.4, env="LOW_CONFIDENCE_THRESHOLD"
    )  # Low confidence similarity

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
    }


# Global settings instance
_settings = None


def get_settings() -> Settings:
    """Get application settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Reset the settings singleton (for testing or reloading)."""
    global _settings
    _settings = None


# For testing
def get_test_settings() -> Settings:
    """Get test-specific settings."""
    return Settings(
        environment="testing",
        debug=True,
        openai_api_key="test-key",
        vector_store_path="./test_data/vector_store",
        allowed_origins=["http://testserver"],
    )
