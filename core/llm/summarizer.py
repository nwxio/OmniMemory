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
            prompt = f"""Create a concise bullet-point digest of this text (5-7 bullets):

{text}

Bullet list:"""
            system = "You are an assistant that writes structured notes."
        elif style == "detailed":
            prompt = f"""Write a detailed summary of this text (2-3 paragraphs):

{text}

Summary:"""
            system = "You are an assistant that writes detailed summaries."
        else:
            prompt = f"""Write a brief summary of this text in 2-3 sentences:

{text}

Summary:"""
            system = "You are an assistant that writes concise summaries."

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
                f"Episode {i + 1}: {e.get('title', '')}\n{e.get('summary', '')}"
                for i, e in enumerate(limited)
            ]
        )

        prompt = f"""Write a short summary of these work episodes (3-4 sentences):

{episodes_text}

Summary:"""

        try:
            result = await llm_client.complete(
                prompt=prompt,
                system="You are an assistant that summarizes work episodes.",
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

        prompt = f"""Extract key points from this text (no more than {max_points}):

{text}

Key points (bullet list):"""

        try:
            result = await llm_client.complete(
                prompt=prompt,
                system="You are an assistant that extracts key points.",
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
                f"Document {i + 1} ({t.get('title', 'Untitled')}):\n{t.get('text', '')}"
                for i, t in enumerate(texts)
            ]
        )

        if focus == "differences":
            prompt = f"""Compare these documents and highlight key differences:

{docs_text}

Key differences:"""
        elif focus == "similarities":
            prompt = f"""Compare these documents and highlight key similarities:

{docs_text}

Key similarities:"""
        else:
            prompt = f"""Compare these documents (similarities and differences):

{docs_text}

Comparison:"""

        try:
            result = await llm_client.complete(
                prompt=prompt,
                system="You are an assistant that compares documents.",
                max_tokens=800,
                temperature=0.3,
                use_cache=True,
            )
            return result.strip()
        except Exception:
            return "Error while comparing documents."


summarizer = Summarizer()
