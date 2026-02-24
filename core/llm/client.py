import asyncio
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from ..llm_config import llm_settings
from .base import LLMProvider, LLMResponse
from .providers import OllamaProvider, OpenAICompatibleProvider
from .cache import LLMCache
from .fallback import FallbackConsolidator


@dataclass
class CircuitBreakerState:
    failures: int = 0
    open_until_ts: float = 0.0


class LLMClient:
    """LLM client with provider routing, cache, retries, and fallback."""

    def __init__(self) -> None:
        self.settings = llm_settings
        self.cache = LLMCache()
        self.fallback = FallbackConsolidator()
        self._provider: Optional[LLMProvider] = None
        self._provider_initialized = False
        self._breaker = CircuitBreakerState()

    def _get_provider_endpoint(self) -> str:
        provider = self.settings.provider.lower()
        if provider == "openai":
            return self.settings.openai_endpoint
        if provider == "deepseek":
            return self.settings.deepseek_endpoint
        if provider == "openrouter":
            return self.settings.openrouter_endpoint
        return self.settings.base_url

    def _create_provider(self) -> LLMProvider:
        provider = self.settings.provider.lower()
        if provider == "ollama":
            return OllamaProvider(
                base_url=self.settings.base_url,
                model=self.settings.model,
                timeout=self.settings.timeout_seconds,
            )
        return OpenAICompatibleProvider(
            base_url=self._get_provider_endpoint(),
            api_key=self.settings.api_key,
            model=self.settings.model,
            timeout=self.settings.timeout_seconds,
        )

    @property
    def provider(self) -> LLMProvider:
        if not self._provider_initialized:
            self._provider = self._create_provider()
            self._provider_initialized = True
        if self._provider is None:
            raise RuntimeError("LLM provider is not initialized")
        return self._provider

    def _is_breaker_open(self) -> bool:
        return time.time() < float(self._breaker.open_until_ts)

    def _record_failure(self) -> None:
        self._breaker.failures += 1
        threshold = max(1, int(getattr(self.settings, "max_retries", 2) or 2))
        if self._breaker.failures >= threshold:
            self._breaker.open_until_ts = time.time() + 15.0

    def _record_success(self) -> None:
        self._breaker.failures = 0
        self._breaker.open_until_ts = 0.0

    def is_available(self) -> bool:
        if not llm_settings.fallback_enabled:
            return self.provider.is_available()
        return True

    async def complete(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 1000,
        temperature: float = 0.7,
        use_cache: bool = True,
    ) -> str:
        if use_cache:
            cached = await self.cache.get(prompt, system)
            if cached:
                return cached

        if self._is_breaker_open():
            if llm_settings.fallback_enabled:
                return self.fallback.summarize(prompt)
            raise RuntimeError("LLM circuit breaker is open")

        try:
            response = await self._retry_with_backoff(
                self.provider.complete,
                prompt=prompt,
                system=system,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            if use_cache and response.content:
                await self.cache.set(prompt, system, response.content)
            self._record_success()
            return response.content
        except Exception:
            self._record_failure()
            if llm_settings.fallback_enabled:
                return self.fallback.summarize(prompt)
            raise

    async def _retry_with_backoff(
        self,
        func: Callable[..., Awaitable[LLMResponse]],
        *args: Any,
        **kwargs: Any,
    ) -> LLMResponse:
        max_retries = self.settings.max_retries
        last_exception: Optional[Exception] = None

        for attempt in range(max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < max_retries:
                    await asyncio.sleep(2 ** attempt)

        if last_exception is not None:
            raise last_exception
        raise RuntimeError("LLM retry failed without exception")

    async def check_availability(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "provider": self.settings.provider,
            "breaker_open": self._is_breaker_open(),
            "breaker_failures": self._breaker.failures,
            "cache_enabled": bool(getattr(self.settings, "cache_enabled", True)),
        }
        try:
            out["provider_configured"] = bool(self.provider.is_available())
        except Exception as e:
            out["provider_configured"] = False
            out["error"] = str(e)
        return out

    async def consolidate(
        self,
        episodes: list[dict[str, Any]],
        session_id: str = "",
    ) -> dict[str, Any]:
        if not episodes:
            return {"ok": True, "lessons": [], "preferences": []}

        combined_text = "\n".join(
            [f"{e.get('title', '')}: {e.get('summary', '')}" for e in episodes]
        )

        prompt = f"""Проанализируй эпизоды работы и извлеки уроки (lessons) и предпочтения (preferences).

Эпизоды:
{combined_text}

Верни результат в JSON формате:
{{
    "lessons": [
        {{"key": "уникальный_ключ", "value": "описание урока", "meta": {{}}}}
    ],
    "preferences": [
        {{"key": "уникальный_ключ", "value": "описание предпочтения"}}
    ],
    "summary": "краткое резюме"
}}

Верни только JSON без дополнительного текста."""

        system = (
            "Ты AI ассистент, который анализирует рабочие сессии и "
            "извлекает полезные уроки и предпочтения пользователя."
        )

        try:
            result = await self.complete(
                prompt=prompt,
                system=system,
                max_tokens=2000,
                temperature=0.3,
            )

            import json

            try:
                parsed = json.loads(result)
                parsed["ok"] = True
                parsed["llm"] = True
                if session_id:
                    parsed["session_id"] = session_id
                return parsed
            except json.JSONDecodeError:
                return self.fallback.consolidate_episodes(episodes)

        except Exception:
            if llm_settings.fallback_enabled:
                return self.fallback.consolidate_episodes(episodes)
            raise

    async def summarize_episodes(self, episodes: list[dict[str, Any]]) -> str:
        if not episodes:
            return ""

        combined_text = "\n".join([f"- {e.get('title', '')}: {e.get('summary', '')}" for e in episodes])
        prompt = f"""Сделай краткое резюме этих эпизодов (2-3 предложения):

{combined_text}

Резюме:"""

        try:
            return await self.complete(prompt=prompt, max_tokens=200, temperature=0.3)
        except Exception:
            return self.fallback.summarize(combined_text)

    async def close(self) -> None:
        await self.cache.close()


llm_client = LLMClient()
