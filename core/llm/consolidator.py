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
        
        prompt = f"""Проанализируй рабочие эпизоды и извлеки уроки (lessons) и предпочтения (preferences).

Эпизоды:
{episodes_text}

Инструкции:
1. Извлеки не более {max_lessons} ключевых уроков
2. Определи предпочтения пользователя
3. Выдели принятые решения
4. Определи темы/концепты

Верни JSON:
{{
    "lessons": [
        {{"key": "уникальный_ключ", "value": "описание урока", "meta": {{"type": "lesson", "topics": []}}}}
    ],
    "preferences": [
        {{"key": "pref_название", "value": "описание предпочтения", "source": "auto"}}
    ],
    "decisions": ["список принятых решений"],
    "topics": ["основные темы"],
    "summary": "краткое резюме всей работы"
}}

Верни ТОЛЬКО JSON без дополнительного текста."""

        system = """Ты AI ассистент для анализа рабочих сессий.
Твоя задача - извлекать полезные уроки и предпочтения из рабочих эпизодов.
Верни структурированный JSON ответ."""

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
            title = ep.get("title", f"Эпизод {i+1}")
            summary = ep.get("summary", "")
            tags = ep.get("tags", [])
            
            lines.append(f"## {title}")
            if tags:
                lines.append(f"Теги: {', '.join(tags)}")
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
        
        prompt = f"""Извлеки сущности из текста:

{text[:8000]}

Верни JSON:
{{
    "names": ["имена собственные"],
    "dates": ["даты и время"],
    "concepts": ["понятия и концепции"],
    "actions": ["действия и операции"]
}}

JSON:"""
        
        try:
            result = await llm_client.complete(
                prompt=prompt,
                system="Ты ассистент для извлечения сущностей.",
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
        
        action_words = ["создал", "изменил", "удалить", "добавил", "настроил", "запустил", "сделал"]
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
        
        prompt = f"""Проанализируй тренды и паттерны в этих эпизодах:

{episodes_text}

Определи:
1. Повторяющиеся темы/паттерны
2. Тренды (что часто повторяется)
3. Ключевые инсайты

Верни JSON:
{{
    "trends": ["тренд 1", "тренд 2"],
    "patterns": ["паттерн 1", "паттерн 2"],
    "insights": ["инсайт 1", "инсайт 2"]
}}

JSON:"""
        
        try:
            result = await llm_client.complete(
                prompt=prompt,
                system="Ты ассистент для анализа трендов.",
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
