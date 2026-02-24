import re
from collections import Counter


PREFERENCE_PATTERNS = [
    # EN
    (r"\bi\s+(?:want|would\s+like)\s+(?P<value>.*?)(?:[.!?]|$)", "want"),
    (r"\bi\s+(?:need|require)\s+(?P<value>.*?)(?:[.!?]|$)", "need"),
    (r"\bi\s+(?:like|love)\s+(?P<value>.*?)(?:[.!?]|$)", "like"),
    (
        r"\bi\s+(?:do\s+not\s+|don't\s+)?(?:want|like|hate)\s+(?P<value>.*?)(?:[.!?]|$)",
        "preference",
    ),
    (r"\balways\s+(?P<value>.*?)(?:[.!?]|$)", "always"),
    (r"\bnever\s+(?P<value>.*?)(?:[.!?]|$)", "never"),
    (r"\bprefer\s+(?P<value>.*?)(?:[.!?]|$)", "prefer"),
    (r"\bbetter\s+(?P<value>.*?)(?:[.!?]|$)", "prefer"),
    (r"\buse\s+(?P<value>.*?)(?:[.!?]|$)", "use"),
    # RU
    (r"\bя\s+(?:хочу|хотел|хотела|желаю)\s+(?P<value>.*?)(?:[.!?]|$)", "want"),
    (r"\bмне\s+(?:нужно|необходимо)\s+(?P<value>.*?)(?:[.!?]|$)", "need"),
    (r"\bмне\s+(?:нравится|люблю)\s+(?P<value>.*?)(?:[.!?]|$)", "like"),
    (r"\bя\s+(?:не\s+)?(?:хочу|люблю|ненавижу)\s+(?P<value>.*?)(?:[.!?]|$)", "preference"),
    (r"\bвсегда\s+(?P<value>.*?)(?:[.!?]|$)", "always"),
    (r"\bникогда\s+(?P<value>.*?)(?:[.!?]|$)", "never"),
    (r"\bпредпочитаю\s+(?P<value>.*?)(?:[.!?]|$)", "prefer"),
    (r"\bлучше\s+(?P<value>.*?)(?:[.!?]|$)", "prefer"),
    (r"\bиспользую\s+(?P<value>.*?)(?:[.!?]|$)", "use"),
    # UK
    (r"\bя\s+(?:хочу|хотів|хотіла|бажаю)\s+(?P<value>.*?)(?:[.!?]|$)", "want"),
    (r"\bмені\s+(?:потрібно|необхідно)\s+(?P<value>.*?)(?:[.!?]|$)", "need"),
    (r"\bмені\s+(?:подобається|люблю)\s+(?P<value>.*?)(?:[.!?]|$)", "like"),
    (r"\bя\s+(?:не\s+)?(?:хочу|люблю|ненавиджу)\s+(?P<value>.*?)(?:[.!?]|$)", "preference"),
    (r"\bзавжди\s+(?P<value>.*?)(?:[.!?]|$)", "always"),
    (r"\bніколи\s+(?P<value>.*?)(?:[.!?]|$)", "never"),
    (r"\bвіддаю\s+перевагу\s+(?P<value>.*?)(?:[.!?]|$)", "prefer"),
    (r"\bкраще\s+(?P<value>.*?)(?:[.!?]|$)", "prefer"),
    (r"\bвикористовую\s+(?P<value>.*?)(?:[.!?]|$)", "use"),
]


class FallbackConsolidator:
    """Simple non-LLM consolidator using heuristics."""

    def extract_keywords(self, text: str, top_n: int = 10) -> list[str]:
        """Extract keywords using a light TF-IDF-like approach."""
        words = re.findall(r"[^\W\d_]{4,}", text.lower(), flags=re.UNICODE)

        stop_words = {
            # EN
            "this",
            "that",
            "what",
            "how",
            "where",
            "when",
            "why",
            "because",
            "therefore",
            "was",
            "will",
            "have",
            "has",
            "been",
            "being",
            "make",
            "done",
            # RU
            "это",
            "что",
            "как",
            "где",
            "когда",
            "почему",
            "потому",
            "было",
            "будет",
            "есть",
            "быть",
            "иметь",
            "сделать",
            # UK
            "це",
            "як",
            "де",
            "коли",
            "чому",
            "тому",
            "було",
            "буде",
            "бути",
            "мати",
            "зробити",
        }

        filtered = [w for w in words if w not in stop_words]
        counter = Counter(filtered)
        return [word for word, _ in counter.most_common(top_n)]

    def extract_preferences(self, text: str) -> list[dict]:
        """Extract simple preference patterns from text."""
        preferences = []

        for pattern, ptype in PREFERENCE_PATTERNS:
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                value = (match.groupdict().get("value") or "").strip()
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
        """Simple summary - keep the first sentences."""
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return ""

        return ". ".join(sentences[:max_sentences]) + "."

    def consolidate_episodes(self, episodes: list[dict]) -> dict:
        """Consolidate episodes into lessons/preferences using simple logic."""
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
                    "value": f"Discussed topics: {', '.join(keywords[:5])}",
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
