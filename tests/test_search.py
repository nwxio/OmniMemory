class TestSearchConfig:
    def test_default_settings(self):
        from core.search_config import SearchSettings

        settings = SearchSettings()

        assert settings.bm25_enabled is True
        assert settings.bm25_weight == 0.3
        assert settings.fts_weight == 0.5
        assert settings.vector_weight == 0.2
        assert settings.query_expansion_enabled is False
        assert settings.rerank_enabled is False
        assert settings.hybrid_use_mmr is True
        assert settings.hybrid_dedupe_enabled is True


class TestBM25:
    def test_bm25_creation(self):
        from core.search.bm25 import BM25Ranker

        ranker = BM25Ranker(k1=1.5, b=0.75)
        assert ranker.bm25.k1 == 1.5
        assert ranker.bm25.b == 0.75

    def test_bm25_tokenize(self):
        from core.search.bm25 import BM25

        bm25 = BM25()
        tokens = bm25._tokenize("это тестовый текст для тестирования")

        assert isinstance(tokens, list)
        assert len(tokens) > 0

    def test_bm25_ranking(self):
        from core.search.bm25 import BM25Ranker

        ranker = BM25Ranker()

        docs = [
            {"text": "питон язык программирования"},
            {"text": "джава скрипт веб"},
            {"text": "питон для машинного обучения"},
        ]

        results = ranker.rank("питон", docs, k=2)

        assert len(results) <= 2
        assert results[0]["bm25_score"] >= results[1]["bm25_score"]

    def test_bm25_cyrillic(self):
        from core.search.bm25 import BM25Ranker

        ranker = BM25Ranker()

        docs = [
            {"text": "машинное обучение нейросети"},
            {"text": "веб разработка на питоне"},
            {"text": "глубокое обучение трансформеры"},
        ]

        results = ranker.rank("обучение", docs, k=3)

        assert len(results) >= 1
        assert results[0].get("bm25_score", 0) > 0


class TestQueryExpansion:
    def test_expander_creation(self):
        from core.search.query_expansion import QueryExpander

        expander = QueryExpander()
        assert expander.ru_synonyms is not None
        assert expander.en_synonyms is not None

    def test_expand_synonyms(self):
        from core.search.query_expansion import QueryExpander

        expander = QueryExpander()
        expansions = expander.expand("файл")

        assert isinstance(expansions, list)

    def test_expand_file_extensions(self):
        from core.search.query_expansion import QueryExpander

        expander = QueryExpander()
        expansions = expander.expand("test.py")

        assert "python" in expansions

    def test_expand_query(self):
        from core.search.query_expansion import QueryExpander

        expander = QueryExpander()
        expanded = expander.expand_query("файл код")

        assert isinstance(expanded, str)
        assert "файл" in expanded or "код" in expanded


class TestHybridSearch:
    def test_hybrid_search_creation(self):
        from core.search.hybrid import HybridSearch

        search = HybridSearch()
        assert search.bm25 is not None
        assert search.query_expander is not None

    def test_hybrid_search_empty_query(self):
        import asyncio
        from core.search.hybrid import HybridSearch

        async def run():
            search = HybridSearch()
            return await search.search("")

        results = asyncio.run(run())
        assert results == []

    def test_hybrid_search_basic(self):
        import asyncio
        from core.search.hybrid import HybridSearch

        async def run():
            search = HybridSearch()
            fts_results = [
                {"path": "/test.py", "text": "def test(): pass", "score": 1.0, "meta": {}},
            ]
            return await search.search("test", fts_results=fts_results)

        results = asyncio.run(run())
        assert len(results) > 0
        assert results[0].get("score") is not None

    def test_norm_scores(self):
        from core.search.hybrid import _norm_scores

        scores = [1.0, 5.0, 10.0]
        normalized = _norm_scores(scores)

        assert normalized[1.0] == 0.0
        assert normalized[10.0] == 1.0
        assert 0 < normalized[5.0] < 1

    def test_jaccard(self):
        from core.search.hybrid import _jaccard

        set_a = {"a", "b", "c"}
        set_b = {"b", "c", "d"}

        similarity = _jaccard(set_a, set_b)

        assert 0 < similarity < 1

    def test_mmr_diversity(self):
        from core.search.hybrid import HybridSearch

        search = HybridSearch()

        candidates = [
            {"text": "питон язык программирования", "score": 1.0, "path": "1"},
            {"text": "питон змея животное", "score": 0.9, "path": "2"},
            {"text": "джава кофе напиток", "score": 0.8, "path": "3"},
        ]

        diversified = search._apply_mmr(candidates, limit=3)

        assert len(diversified) <= 3


class TestSearchIntegration:
    def test_search_module_import(self):
        from core import search

        assert hasattr(search, "hybrid_search")
        assert hasattr(search, "search_settings")

    def test_search_settings_in_search_module(self):
        from core.search import search_settings

        assert search_settings.bm25_enabled is True
        assert search_settings.bm25_weight == 0.3
