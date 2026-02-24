from typing import List, Dict, Any, Optional
import json

from ..llm_config import llm_settings
from .client import llm_client
from .fallback import FallbackConsolidator


class EnhancedConsolidator:
    """Enhanced consolidation using LLM."""

    def __init__(self):
        self.fallback = FallbackConsolidator()

    async def consolidate(
        self,
        episodes: List[Dict[str, Any]],
        session_id: str = "",
        max_lessons: int = 10,
    ) -> Dict[str, Any]:
        """Consolidate episodes into lessons and preferences using LLM.

        Args:
            episodes: List of episodes to consolidate
            session_id: Session ID
            max_lessons: Maximum number of lessons to generate
        """
        if not episodes:
            return {"ok": True, "lessons": [], "preferences": []}

        episodes_text = self._format_episodes(episodes)

        prompt = f"""Analyze work episodes and extract lessons and preferences.

Episodes:
{episodes_text}

Instructions:
1. Extract no more than {max_lessons} key lessons
2. Identify user preferences
3. Highlight decisions made
4. Identify topics/concepts

Return JSON:
{{
    "lessons": [
        {{"key": "unique_key", "value": "lesson description", "meta": {{"type": "lesson", "topics": []}}}}
    ],
    "preferences": [
        {{"key": "pref_name", "value": "preference description", "source": "auto"}}
    ],
    "decisions": ["list of decisions"],
    "topics": ["main topics"],
    "summary": "short summary of the work"
}}

Return ONLY JSON without additional text."""

        system = """You are an AI assistant for analyzing work sessions.
Your task is to extract useful lessons and preferences from work episodes.
Return a structured JSON response."""

        try:
            result = await llm_client.complete(
                prompt=prompt,
                system=system,
                max_tokens=2000,
                temperature=0.3,
                use_cache=True,
            )

            parsed = self._parse_json_response(result)

            if parsed:
                parsed["ok"] = True
                parsed["llm"] = True
                return parsed

            return self.fallback.consolidate_episodes(episodes)

        except Exception:
            if llm_settings.fallback_enabled:
                return self.fallback.consolidate_episodes(episodes)
            raise

    def _format_episodes(self, episodes: List[Dict[str, Any]]) -> str:
        """Format episodes for LLM prompt."""
        lines = []
        for i, ep in enumerate(episodes):
            title = ep.get("title", f"Episode {i + 1}")
            summary = ep.get("summary", "")
            tags = ep.get("tags", [])

            lines.append(f"## {title}")
            if tags:
                lines.append(f"Tags: {', '.join(tags)}")
            lines.append(summary)
            lines.append("")

        return "\n".join(lines)

    def _parse_json_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse JSON from LLM response."""
        try:
            response = response.strip()

            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]

            response = response.strip()

            return json.loads(response)

        except (json.JSONDecodeError, IndexError):
            return None

    async def extract_entities(
        self,
        text: str,
    ) -> Dict[str, List[str]]:
        """Extract entities from text (names, dates, concepts)."""
        if not text:
            return {"names": [], "dates": [], "concepts": [], "actions": []}

        if not llm_settings.auto_consolidate:
            return self._simple_extract_entities(text)

        prompt = f"""Extract entities from the text:

{text[:8000]}

Return JSON:
{{
    "names": ["proper names"],
    "dates": ["dates and time"],
    "concepts": ["concepts"],
    "actions": ["actions and operations"]
}}

JSON:"""

        try:
            result = await llm_client.complete(
                prompt=prompt,
                system="You are an assistant for entity extraction.",
                max_tokens=500,
                temperature=0.2,
                use_cache=True,
            )

            entities = self._parse_json_response(result)
            if entities:
                return entities

        except Exception:
            pass

        return self._simple_extract_entities(text)

    def _simple_extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Simple entity extraction without LLM."""
        import re

        names = []
        dates = []
        concepts = []
        actions = []

        date_pattern = r"\d{1,2}[\.\-/]\d{1,2}[\.\-/]\d{2,4}|\d{4}[\.\-/]\d{1,2}[\.\-/]\d{1,2}"
        dates = re.findall(date_pattern, text)

        words = text.split()
        for word in words:
            if word and word[0].isupper() and len(word) > 3:
                names.append(word)

        action_words = ["created", "updated", "delete", "added", "configured", "started", "done"]
        for action in action_words:
            if action in text.lower():
                actions.append(action)

        return {
            "names": list(set(names))[:10],
            "dates": list(set(dates))[:10],
            "concepts": concepts,
            "actions": list(set(actions))[:10],
        }

    async def analyze_trends(
        self,
        episodes: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Analyze trends across multiple episodes."""
        if not episodes:
            return {"trends": [], "patterns": [], "insights": []}

        episodes_text = self._format_episodes(episodes)

        prompt = f"""Analyze trends and patterns in these episodes:

{episodes_text}

Identify:
1. Recurring topics/patterns
2. Trends (what repeats frequently)
3. Key insights

Return JSON:
{{
    "trends": ["trend 1", "trend 2"],
    "patterns": ["pattern 1", "pattern 2"],
    "insights": ["insight 1", "insight 2"]
}}

JSON:"""

        try:
            result = await llm_client.complete(
                prompt=prompt,
                system="You are an assistant for trend analysis.",
                max_tokens=800,
                temperature=0.3,
                use_cache=True,
            )

            analysis = self._parse_json_response(result)
            if analysis:
                return analysis

        except Exception:
            pass

        return {"trends": [], "patterns": [], "insights": []}


enhanced_consolidator = EnhancedConsolidator()
