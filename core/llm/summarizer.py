from typing import List, Dict, Any

from ..llm_config import llm_settings
from .client import llm_client
from .fallback import FallbackConsolidator


class Summarizer:
    """Auto-summarization for episodes and documents."""

    def __init__(self):
        self.fallback = FallbackConsolidator()

    async def summarize(
        self,
        text: str,
        max_length: int = 500,
        style: str = "brief",
    ) -> str:
        """Summarize text using LLM.

        Args:
            text: Text to summarize
            max_length: Maximum length of summary
            style: Style of summary (brief, detailed, bullet_points)
        """
        if not text:
            return ""

        if not llm_settings.auto_consolidate:
            return self.fallback.summarize(text, max_sentences=3)

        if style == "bullet_points":
            prompt = f"""Сделай краткую выжимку из текста в виде маркированного списка (5-7 пунктов):

{text}

Маркированный список:"""
            system = "Ты ассистент, который создает структурированные заметки."
        elif style == "detailed":
            prompt = f"""Сделай подробное резюме текста (2-3 параграфа):

{text}

Резюме:"""
            system = "Ты ассистент, который создает подробные резюме."
        else:
            prompt = f"""Сделай краткое резюме текста в 2-3 предложениях:

{text}

Резюме:"""
            system = "Ты ассистент, который создает краткие резюме."

        try:
            result = await llm_client.complete(
                prompt=prompt,
                system=system,
                max_tokens=max_length // 4,
                temperature=0.3,
                use_cache=True,
            )
            return result.strip()
        except Exception:
            return self.fallback.summarize(text, max_sentences=3)

    async def summarize_episodes(
        self,
        episodes: List[Dict[str, Any]],
        max_episodes: int = 10,
    ) -> str:
        """Summarize multiple episodes into one summary."""
        if not episodes:
            return ""

        limited = episodes[:max_episodes]

        episodes_text = "\n\n".join(
            [
                f"Эпизод {i + 1}: {e.get('title', '')}\n{e.get('summary', '')}"
                for i, e in enumerate(limited)
            ]
        )

        prompt = f"""Сделай краткое резюме этих эпизодов работы (3-4 предложения):

{episodes_text}

Резюме:"""

        try:
            result = await llm_client.complete(
                prompt=prompt,
                system="Ты ассистент, который суммаризирует рабочие эпизоды.",
                max_tokens=300,
                temperature=0.3,
                use_cache=True,
            )
            return result.strip()
        except Exception:
            combined = "\n".join([e.get("summary", "") for e in limited])
            return self.fallback.summarize(combined, max_sentences=3)

    async def extract_key_points(
        self,
        text: str,
        max_points: int = 7,
    ) -> List[str]:
        """Extract key points from text as bullet list."""
        if not text:
            return []

        prompt = f"""Извлеки ключевые пункты из текста (не более {max_points}):

{text}

Ключевые пункты (маркированный список):"""

        try:
            result = await llm_client.complete(
                prompt=prompt,
                system="Ты ассистент, который извлекает ключевые пункты.",
                max_tokens=500,
                temperature=0.3,
                use_cache=True,
            )

            points = []
            for line in result.strip().split("\n"):
                line = line.strip()
                if line and (line.startswith("-") or line.startswith("*") or line[0].isdigit()):
                    clean = line.lstrip(" -*0123456789. ").strip()
                    if clean:
                        points.append(clean)

            return points[:max_points]

        except Exception:
            keywords = self.fallback.extract_keywords(text, top_n=max_points)
            return keywords

    async def compare(
        self,
        texts: List[Dict[str, str]],
        focus: str = "differences",
    ) -> str:
        """Compare multiple texts/documents.

        Args:
            texts: List of dicts with 'title' and 'text' keys
            focus: What to focus on (differences, similarities, both)
        """
        if not texts:
            return ""

        docs_text = "\n\n".join(
            [
                f"Документ {i + 1} ({t.get('title', 'Без названия')}):\n{t.get('text', '')}"
                for i, t in enumerate(texts)
            ]
        )

        if focus == "differences":
            prompt = f"""Сравни документы и выдели ключевые различия:

{docs_text}

Ключевые различия:"""
        elif focus == "similarities":
            prompt = f"""Сравни документы и выдели ключевые сходства:

{docs_text}

Ключевые сходства:"""
        else:
            prompt = f"""Сравни документы (сходства и различия):

{docs_text}

Сравнение:"""

        try:
            result = await llm_client.complete(
                prompt=prompt,
                system="Ты ассистент, который сравнивает документы.",
                max_tokens=800,
                temperature=0.3,
                use_cache=True,
            )
            return result.strip()
        except Exception:
            return "Ошибка при сравнении документов."


summarizer = Summarizer()
