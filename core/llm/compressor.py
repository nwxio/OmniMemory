from typing import List, Dict, Any
import json

from ..llm_config import llm_settings
from .client import llm_client


class ContextCompressor:
    """Compress context/messages for LLM context window."""

    def __init__(self):
        pass

    async def compress_messages(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 4000,
    ) -> List[Dict[str, str]]:
        """Compress messages to fit within token limit.

        Args:
            messages: List of messages with 'role' and 'content'
            max_tokens: Maximum tokens to keep
        """
        if not messages:
            return []

        prompt = self._messages_to_text(messages)

        if not llm_settings.auto_consolidate:
            return self._simple_compress(messages, max_tokens)

        prompt_text = f"""Compress this message while preserving key information and meaning:

{prompt[:10000]}

Compressed version (keep key facts and decisions):"""

        try:
            compressed = await llm_client.complete(
                prompt=prompt_text,
                system="You are an assistant that compresses context while preserving key information.",
                max_tokens=max_tokens // 4,
                temperature=0.3,
                use_cache=False,
            )

            return [{"role": "user", "content": compressed}]

        except Exception:
            return self._simple_compress(messages, max_tokens)

    def _messages_to_text(self, messages: List[Dict[str, str]]) -> str:
        """Convert messages to text format."""
        lines = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")
        return "\n\n".join(lines)

    def _simple_compress(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
    ) -> List[Dict[str, str]]:
        """Simple compression - keep first and last messages."""
        if not messages:
            return []

        if len(messages) <= 2:
            return messages

        first = messages[0]
        last = messages[-1]

        return [
            first,
            {"role": "system", "content": "[context compressed]"},
            last,
        ]

    async def extract_important(
        self,
        messages: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """Extract important information from messages.

        Returns:
            dict with keys: decisions, facts, tasks, entities
        """
        if not messages:
            return {"decisions": [], "facts": [], "tasks": [], "entities": []}

        if not llm_settings.auto_consolidate:
            return self._simple_extract(messages)

        prompt = f"""Extract key information from this conversation:

{self._messages_to_text(messages)}

Return JSON:
{{
    "decisions": ["list of decisions made"],
    "facts": ["important facts"],
    "tasks": ["mentioned tasks"],
    "entities": ["entities (names, products, concepts)"]
}}

JSON:"""

        try:
            result = await llm_client.complete(
                prompt=prompt,
                system="You are an assistant that extracts structured information.",
                max_tokens=1000,
                temperature=0.3,
                use_cache=True,
            )

            data = json.loads(result)
            return data

        except Exception:
            return self._simple_extract(messages)

    def _simple_extract(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Simple extraction without LLM."""
        decisions = []
        facts = []
        tasks = []
        entities = set()

        keywords_decisions = ["decided", "decision", "approved", "will", "need to do"]
        keywords_tasks = ["task", "do", "todo", "need", "next", "follow-up"]

        for msg in messages:
            content = msg.get("content", "").lower()

            if any(k in content for k in keywords_decisions):
                decisions.append(content[:100])

            if any(k in content for k in keywords_tasks):
                tasks.append(content[:100])

            words = content.split()
            for word in words:
                if word and word[0].isupper() and len(word) > 3:
                    entities.add(word)

        return {
            "decisions": decisions[:5],
            "facts": facts[:5],
            "tasks": tasks[:5],
            "entities": list(entities)[:10],
        }

    async def compress_document(
        self,
        text: str,
        max_tokens: int = 2000,
        focus: str = "main",
    ) -> str:
        """Compress a single document.

        Args:
            text: Document text
            max_tokens: Target token count
            focus: What to focus on (main, details, technical)
        """
        if not text:
            return ""

        if not llm_settings.auto_consolidate:
            return text[: max_tokens * 4]

        if focus == "technical":
            prompt = f"""Extract technical information from this text (code, API, configuration):

{text[:15000]}

Technical details:"""
        elif focus == "details":
            prompt = f"""Keep all important details from this text:

{text[:15000]}

Details:"""
        else:
            prompt = f"""Write a concise summary of this text while preserving key information:

{text[:15000]}

Summary:"""

        try:
            result = await llm_client.complete(
                prompt=prompt,
                system="You are an assistant that compresses documents.",
                max_tokens=max_tokens,
                temperature=0.3,
                use_cache=True,
            )
            return result.strip()
        except Exception:
            return text[: max_tokens * 4]


context_compressor = ContextCompressor()
