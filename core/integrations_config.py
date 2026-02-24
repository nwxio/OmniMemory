from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class DatabaseSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OMNIMIND_")

    db_type: str = "sqlite"

    # SQLite
    db_path: str = "./memory.db"

    # PostgreSQL
    postgres_host: str = "localhost"
    postgres_port: int = 5442
    postgres_db: str = "memory"
    postgres_user: str = "postgres"
    postgres_password: str = ""

    # Redis
    redis_enabled: bool = False
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None


class EmbeddingsSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OMNIMIND_EMBEDDINGS_")

    provider: str = "fastembed"

    # FastEmbed (default)
    fastembed_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    # OpenAI
    openai_api_key: str = ""
    openai_embedding_model: str = "text-embedding-3-small"

    # Cohere
    cohere_api_key: str = ""
    cohere_model: str = "embed-multilingual-v3.0"


db_settings = DatabaseSettings()
embeddings_settings = EmbeddingsSettings()
