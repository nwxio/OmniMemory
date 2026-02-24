from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OMNIMIND_LLM_")

    provider: str = "ollama"
    base_url: str = "http://localhost:11434"
    api_key: str = ""
    model: str = "llama3.2"

    auto_consolidate: bool = True
    cache_enabled: bool = True
    cache_ttl_hours: int = 168
    cache_path: str = "./llm_cache.db"
    timeout_seconds: int = 60
    max_retries: int = 2
    fallback_enabled: bool = True

    deepseek_endpoint: str = "https://api.deepseek.com/v1"
    openrouter_endpoint: str = "https://openrouter.ai/api/v1"
    openai_endpoint: str = "https://api.openai.com/v1"


llm_settings = LLMSettings()
