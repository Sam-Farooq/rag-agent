"""Settings. Everything is env-driven so the same image runs locally and on EKS."""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="RAG_", extra="ignore")

    # Generation. Claude is the default because the grader prompt in evals/ is
    # calibrated against it; OpenAI stays wired up as the fallback provider.
    primary_model: str = "claude-sonnet-4-5"
    fallback_model: str = "gpt-4o"
    temperature: float = 0.0
    max_tokens: int = 1024

    # Retrieval
    qdrant_url: str = "http://localhost:6333"
    collection: str = "regulations"
    embed_model: str = "BAAI/bge-base-en-v1.5"
    rerank_model: str = "BAAI/bge-reranker-base"
    # Pull wide, re-rank, keep few. 24 -> 5 was chosen in evals/, see README.
    candidate_k: int = 24
    final_k: int = 5
    min_rerank_score: float = 0.15

    # Self-correction loop
    max_retrieval_retries: int = 2
    min_grounding_score: float = 0.6

    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 3600

    langfuse_host: str = "https://cloud.langfuse.com"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
