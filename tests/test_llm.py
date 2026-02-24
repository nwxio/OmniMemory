import os


class TestLLMConfig:
    def test_default_config(self):
        from core.llm_config import LLMSettings

        settings = LLMSettings()

        assert settings.provider == "ollama"
        assert settings.base_url == "http://localhost:11434"
        assert settings.model == "llama3.2"
        assert settings.auto_consolidate is True
        assert settings.cache_enabled is True
        assert settings.fallback_enabled is True

    def test_env_override(self):
        os.environ["OMNIMIND_LLM_PROVIDER"] = "openai"
        os.environ["OMNIMIND_LLM_API_KEY"] = "test-key"
        os.environ["OMNIMIND_LLM_MODEL"] = "gpt-4"

        # Reload module
        import importlib
        import core.llm_config

        importlib.reload(core.llm_config)
        from core.llm_config import LLMSettings

        settings = LLMSettings()

        assert settings.provider == "openai"
        assert settings.api_key == "test-key"
        assert settings.model == "gpt-4"

        # Cleanup
        del os.environ["OMNIMIND_LLM_PROVIDER"]
        del os.environ["OMNIMIND_LLM_API_KEY"]
        del os.environ["OMNIMIND_LLM_MODEL"]

        # Reload again
        importlib.reload(core.llm_config)


class TestFallbackConsolidator:
    def test_extract_keywords(self):
        from core.llm.fallback import FallbackConsolidator

        consol = FallbackConsolidator()

        text = "this is a test text to verify keyword extraction test test"
        keywords = consol.extract_keywords(text, top_n=3)

        assert isinstance(keywords, list)
        assert len(keywords) <= 3

    def test_extract_preferences(self):
        from core.llm.fallback import FallbackConsolidator

        consol = FallbackConsolidator()

        text = "i want to use python for work i like machine learning"
        prefs = consol.extract_preferences(text)

        assert isinstance(prefs, list)
        assert len(prefs) > 0

    def test_extract_preferences_ru(self):
        from core.llm.fallback import FallbackConsolidator

        consol = FallbackConsolidator()

        text = "я хочу использовать питон для работы"
        prefs = consol.extract_preferences(text)

        assert isinstance(prefs, list)
        assert len(prefs) > 0

    def test_extract_preferences_uk(self):
        from core.llm.fallback import FallbackConsolidator

        consol = FallbackConsolidator()

        text = "мені потрібно використовувати python для роботи"
        prefs = consol.extract_preferences(text)

        assert isinstance(prefs, list)
        assert len(prefs) > 0

    def test_summarize(self):
        from core.llm.fallback import FallbackConsolidator

        consol = FallbackConsolidator()

        text = "First sentence. Second sentence. Third sentence."
        summary = consol.summarize(text, max_sentences=2)

        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_consolidate_episodes(self):
        from core.llm.fallback import FallbackConsolidator

        consol = FallbackConsolidator()

        episodes = [
            {"title": "Task 1", "summary": "Completed the first task"},
            {"title": "Task 2", "summary": "Completed the second task"},
        ]

        result = consol.consolidate_episodes(episodes)

        assert result["ok"] is True
        assert "lessons" in result
        assert "preferences" in result


class TestOllamaProvider:
    def test_ollama_provider_creation(self):
        from core.llm.providers.ollama import OllamaProvider

        provider = OllamaProvider(
            base_url="http://localhost:11434",
            model="llama3.2",
            timeout=30,
        )

        assert provider.base_url == "http://localhost:11434"
        assert provider.model == "llama3.2"
        assert provider.is_available() is True

    def test_ollama_endpoint(self):
        from core.llm.providers.ollama import OllamaProvider

        provider = OllamaProvider(
            base_url="http://localhost:11434",
            model="llama3.2",
        )

        endpoint = provider._get_endpoint()
        assert "/api/chat" in endpoint


class TestOpenAICompatibleProvider:
    def test_openai_provider_creation(self):
        from core.llm.providers.openai_compatible import OpenAICompatibleProvider

        provider = OpenAICompatibleProvider(
            base_url="https://api.openai.com/v1",
            api_key="sk-test",
            model="gpt-4",
            timeout=30,
        )

        assert provider.base_url == "https://api.openai.com/v1"
        assert provider.api_key == "sk-test"
        assert provider.model == "gpt-4"
        assert provider.is_available() is True

    def test_deepseek_endpoint(self):
        from core.llm_config import LLMSettings

        settings = LLMSettings()

        assert settings.deepseek_endpoint == "https://api.deepseek.com/v1"

    def test_openrouter_endpoint(self):
        from core.llm_config import LLMSettings

        settings = LLMSettings()

        assert settings.openrouter_endpoint == "https://openrouter.ai/api/v1"

    def test_openai_headers(self):
        from core.llm.providers.openai_compatible import OpenAICompatibleProvider

        provider = OpenAICompatibleProvider(
            base_url="https://api.openai.com/v1",
            api_key="sk-test",
            model="gpt-4",
        )

        headers = provider._get_headers()
        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer sk-test"


class TestLLMCache:
    def test_cache_init(self):
        from core.llm.cache import LLMCache

        cache = LLMCache(db_path=":memory:")

        assert cache.db_path == ":memory:"

    def test_cache_disabled_setting(self):
        from core.llm_config import LLMSettings

        settings = LLMSettings()

        # Default enabled
        assert settings.cache_enabled is True


class TestLLMClient:
    def test_client_creation(self):
        from core.llm.client import LLMClient

        client = LLMClient()

        assert client.settings is not None
        assert client.cache is not None
        assert client.fallback is not None

    def test_provider_selection_ollama(self):
        from core.llm.client import LLMClient

        client = LLMClient()
        provider = client.provider

        assert provider is not None

    def test_is_available(self):
        from core.llm.client import LLMClient

        client = LLMClient()

        # Fallback enabled, so should always be available
        assert client.is_available() is True


class TestMemoryIntegration:
    def test_consolidate_imports(self):
        from core.memory import MemoryStore
        from core.llm_config import llm_settings

        store = MemoryStore()

        assert hasattr(store, "consolidate")

        # Check settings are loaded
        assert llm_settings is not None
        assert llm_settings.provider == "ollama"

    def test_memory_has_llm_settings(self):
        from core.memory import llm_settings

        assert llm_settings is not None
        assert hasattr(llm_settings, "provider")
        assert hasattr(llm_settings, "auto_consolidate")
        assert hasattr(llm_settings, "cache_enabled")
        assert hasattr(llm_settings, "fallback_enabled")
