from pydantic_settings import BaseSettings, SettingsConfigDict


class SearchSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OMNIMIND_SEARCH_")

    bm25_enabled: bool = True
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    bm25_weight: float = 0.3

    fts_enabled: bool = True
    fts_weight: float = 0.5

    vector_enabled: bool = True
    vector_weight: float = 0.2

    query_expansion_enabled: bool = False
    query_expansion_use_llm: bool = False
    query_expansion_max_terms: int = 5

    rerank_enabled: bool = False
    rerank_use_llm: bool = True
    rerank_top_k: int = 10

    hybrid_recency_enabled: bool = True
    hybrid_recency_half_life_days: float = 30.0
    hybrid_recency_max_bonus: float = 0.12

    hybrid_use_mmr: bool = True
    hybrid_mmr_lambda: float = 0.70
    hybrid_dedupe_enabled: bool = True
    hybrid_dedupe_threshold: float = 0.92


search_settings = SearchSettings()
