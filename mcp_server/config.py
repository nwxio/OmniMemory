"""MCP server configuration."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class MCPSettings(BaseSettings):
    """MCP server settings."""

    model_config = SettingsConfigDict(env_prefix="MEMORY_")

    # Server
    server_name: str = "omnimind-memory"
    server_version: str = "0.1.0"

    # Storage
    db_path: str = "./memory.db"

    # TTL settings (days)
    lessons_ttl_days: int = 90
    episodes_ttl_days: int = 60
    preferences_ttl_days: int = 180

    # Vector settings
    vector_enabled: bool = True
    vector_model: str = "BAAI/bge-small-en-v1.5"
    vector_dimensions: int = 384

    # Graph settings
    graph_enabled: bool = True
    graph_max_file_size: int = 1_000_000  # 1MB

    # Embeddings
    embeddings_provider: str = "fastembed"  # fastembed, openai, cohere


mcp_settings = MCPSettings()
