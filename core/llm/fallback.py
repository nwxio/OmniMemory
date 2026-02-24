import re
from collections import Counter


PREFERENCE_PATTERNS = [
    (r"я\s+(хочу|хотел|хотела|желаю)\s+(.*?)(?:\.|$)", "want"),
    (r"мне\s+(нужно|необходимо)\s+(.*?)(?:\.|$)", "need"),
    (r"мне\s+(нравится|люблю)\s+(.*?)(?:\.|$)", "like"),
    (r"я\s+(не\s+)?(хочу|люблю|ненавижу)\s+(.*?)(?:\.|$)", "preference"),
    (r"всегда\s+(.*?)(?:\.|$)", "always"),
    (r"никогда\s+(.*?)(?:\.|$)", "never"),
    (r"предпочитаю\s+(.*?)(?:\.|$)", "prefer"),
    (r"лучше\s+(.*?)(?:\.|$)", "prefer"),
    (r"использую\s+(.*?)(?:\.|$)", "use"),
]


class FallbackConsolidator:
    """Простой консолидатор без LLM - использует эвристики."""

    def extract_keywords(self, text: str, top_n: int = 10) -> list[str]:
        """Извлекает ключевые слова через TF-IDF-like подход."""
        words = re.findall(r"[а-яёa-zA-Z]{4,}", text.lower())

        stop_words = {
            "это",
            "что",
            "как",
            "где",
            "когда",
            "почему",
            "потому",
            "therefore",
            "this",
            "that",
            "what",
            "how",
            "where",
            "было",
            "будет",
            "есть",
            "быть",
            "иметь",
            "сделать",
            "was",
            "will",
            "have",
            "has",
            "been",
            "being",
        }

        filtered = [w for w in words if w not in stop_words]
        counter = Counter(filtered)
        return [word for word, _ in counter.most_common(top_n)]

    def extract_preferences(self, text: str) -> list[dict]:
        """Извлекает простые паттерны preferences из текста."""
        preferences = []

        for pattern, ptype in PREFERENCE_PATTERNS:
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                value = match.group(2).strip() if match.lastindex >= 2 else match.group(1).strip()
                if value and len(value) > 2:
                    preferences.append(
                        {
                            "type": ptype,
                            "value": value,
                            "original": match.group(0),
                        }
                    )

        return preferences[:10]

    def summarize(self, text: str, max_sentences: int = 3) -> str:
        """Простое summary - первые предложения."""
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return ""

        return ". ".join(sentences[:max_sentences]) + "."

    def consolidate_episodes(self, episodes: list[dict]) -> dict:
        """Консолидирует эпизоды в lessons/preferences через простую логику."""
        if not episodes:
            return {"ok": True, "lessons": [], "preferences": []}

        combined_text = "\n".join(
            [f"{e.get('title', '')}: {e.get('summary', '')}" for e in episodes]
        )

        keywords = self.extract_keywords(combined_text, top_n=10)
        preferences = self.extract_preferences(combined_text)
        summary = self.summarize(combined_text)

        lessons = []
        if keywords:
            lessons.append(
                {
                    "key": f"topic_{keywords[0]}",
                    "value": f"Обсуждались темы: {', '.join(keywords[:5])}",
                    "meta": {"keywords": keywords, "type": "auto_fallback"},
                }
            )

        if summary:
            lessons.append(
                {
                    "key": "session_summary",
                    "value": summary,
                    "meta": {"type": "summary"},
                }
            )

        prefs_output = []
        for p in preferences[:5]:
            prefs_output.append(
                {
                    "key": f"pref_{p['type']}_{len(prefs_output)}",
                    "value": p["value"],
                    "source": "auto_fallback",
                }
            )

        return {
            "ok": True,
            "lessons": lessons,
            "preferences": prefs_output,
            "summary": summary,
            "keywords": keywords,
            "fallback": True,
        }
