from typing import List, Dict, Any

from ..llm_config import llm_settings


class LLMReranker:
    """Re-rank search results using LLM."""
    
    def __init__(self):
        self.enabled = llm_settings.fallback_enabled
    
    async def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Re-rank documents using LLM."""
        if not documents or not query:
            return documents[:top_k]
        
        if not self.enabled or not llm_settings.auto_consolidate:
            return documents[:top_k]
        
        try:
            from ..llm.client import llm_client
            
            docs_text = "\n\n".join([
                f"[{i+1}] {doc.get('text', '')[:500]}"
                for i, doc in enumerate(documents[:20])
            ])
            
            prompt = f"""Оцени релевантность каждого документа к запросу.
Запрос: {query}

Документы:
{docs_text}

Верни JSON массив с оценками релевантности (0-10) для каждого документа в формате:
[{{"index": 0, "score": 8.5}}, {{"index": 1, "score": 3.2}}, ...]

Верни только JSON массив без дополнительного текста."""
            
            system = "Ты эксперт по оценке релевантности документов. Верни только JSON."
            
            result = await llm_client.complete(
                prompt=prompt,
                system=system,
                max_tokens=500,
                temperature=0.1,
                use_cache=True,
            )
            
            import json
            try:
                scores = json.loads(result)
                
                score_map = {item["index"]: item["score"] for item in scores}
                
                for i, doc in enumerate(documents):
                    doc["llm_score"] = score_map.get(i, 0.0)
                
                documents.sort(key=lambda x: x.get("llm_score", 0), reverse=True)
                
            except (json.JSONDecodeError, KeyError, TypeError):
                pass
            
            return documents[:top_k]
        
        except Exception:
            return documents[:top_k]
    
    async def score_single(
        self,
        query: str,
        document: Dict[str, Any],
    ) -> float:
        """Score a single document against query."""
        if not self.enabled:
            return document.get("score", 0.0)
        
        try:
            from ..llm.client import llm_client
            
            prompt = f"""Оцени релевантность документа к запросу от 0 до 10.
Запрос: {query}
Документ: {document.get('text', '')[:1000]}

Верни только число (например: 7.5)"""
            
            result = await llm_client.complete(
                prompt=prompt,
                max_tokens=10,
                temperature=0.1,
                use_cache=True,
            )
            
            try:
                score = float(result.strip())
                return min(max(score, 0.0), 10.0)
            except ValueError:
                return document.get("score", 0.0)
        
        except Exception:
            return document.get("score", 0.0)


llm_reranker = LLMReranker()
