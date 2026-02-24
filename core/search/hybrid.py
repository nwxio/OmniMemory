import re
import math
from typing import List, Dict, Any, Optional

from .bm25 import BM25Ranker
from .query_expansion import query_expander
from .reranker import llm_reranker
from ..search_config import search_settings


def _norm_scores(vals: List[float]) -> Dict[float, float]:
    """Normalize scores to 0..1 range."""
    if not vals:
        return {}
    mn = min(vals)
    mx = max(vals)
    if mx == mn:
        return {v: 1.0 for v in vals}
    return {v: (v - mn) / (mx - mn) for v in vals}


def _tokset(s: str) -> set:
    """Tokenize text for similarity comparison."""
    text = (s or "")[:800].lower()
    tokens = re.findall(r"[^\W_]{2,}", text, flags=re.UNICODE)
    if len(tokens) > 120:
        tokens = tokens[:120]
    return set(tokens)


def _jaccard(a: set, b: set) -> float:
    """Calculate Jaccard similarity."""
    if not a or not b:
        return 0.0
    inter = len(a & b)
    uni = len(a | b)
    if uni <= 0:
        return 0.0
    return inter / uni


class HybridSearch:
    """Hybrid search combining FTS, BM25, and vector search."""
    
    def __init__(self):
        self.bm25 = BM25Ranker(
            k1=search_settings.bm25_k1,
            b=search_settings.bm25_b,
        )
        self.query_expander = query_expander
        self.reranker = llm_reranker
    
    async def search(
        self,
        query: str,
        fts_results: Optional[List[Dict[str, Any]]] = None,
        vector_results: Optional[List[Dict[str, Any]]] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search combining multiple sources."""
        if not query:
            return []
        
        expanded_query = query
        if search_settings.query_expansion_enabled:
            expanded = self.query_expander.expand_query(
                query,
                max_terms=search_settings.query_expansion_max_terms,
            )
            if expanded:
                expanded_query = expanded
        
        candidates: List[Dict[str, Any]] = []
        
        fts_hits = fts_results or []
        for h in fts_hits:
            candidates.append({
                "path": h.get("path", ""),
                "text": h.get("text", "") or h.get("snippet", ""),
                "fts_score": h.get("score", 0),
                "vec_score": None,
                "bm25_score": None,
                "src": "fts",
                "meta": h.get("meta", {}),
            })
        
        vec_hits = vector_results or []
        for h in vec_hits:
            candidates.append({
                "path": h.get("path", ""),
                "text": h.get("text", "") or h.get("snippet", ""),
                "fts_score": None,
                "vec_score": h.get("score", 0),
                "bm25_score": None,
                "src": "vector",
                "meta": h.get("meta", {}),
            })
        
        if search_settings.bm25_enabled and candidates:
            docs_for_bm25 = [
                {"path": c.get("path", ""), "text": c.get("text", "")} for c in candidates
            ]
            bm25_scores = self.bm25.score_all(expanded_query, docs_for_bm25)
            for i, c in enumerate(candidates):
                c["bm25_score"] = float(bm25_scores[i]) if i < len(bm25_scores) else 0.0
        
        fts_scores = [c["fts_score"] for c in candidates if c.get("fts_score") is not None]
        vec_scores = [c["vec_score"] for c in candidates if c.get("vec_score") is not None]
        bm25_scores = [c["bm25_score"] for c in candidates if c.get("bm25_score") is not None]
        
        fts_norm = _norm_scores(fts_scores)
        vec_norm = _norm_scores(vec_scores)
        bm25_norm = _norm_scores(bm25_scores)
        
        tokens = re.findall(r"[^\W_]{3,}", query.lower(), flags=re.UNICODE)
        if len(tokens) > 40:
            tokens = tokens[:40]
        tokset = set(tokens)
        
        for c in candidates:
            fts_s = c.get("fts_score")
            vec_s = c.get("vec_score")
            bm25_s = c.get("bm25_score")
            
            fn = fts_norm.get(fts_s, 0.0) if fts_s is not None else 0.0
            vn = vec_norm.get(vec_s, 0.0) if vec_s is not None else 0.0
            bn = bm25_norm.get(bm25_s, 0.0) if bm25_s is not None else 0.0
            
            bonus = 0.0
            path = c.get("path", "")
            if path and tokset:
                pl = path.lower()
                if any(t in pl for t in tokset):
                    bonus += 0.15
            
            score = (
                search_settings.fts_weight * fn +
                search_settings.vector_weight * vn +
                search_settings.bm25_weight * bn +
                bonus
            )
            c["score"] = score
            c["normalized"] = {
                "fts": fn,
                "vector": vn,
                "bm25": bn,
                "bonus": bonus,
            }
        
        candidates.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        if search_settings.hybrid_use_mmr:
            candidates = self._apply_mmr(candidates, limit)
        
        if search_settings.hybrid_dedupe_enabled:
            candidates = self._dedupe(candidates, limit)
        
        if search_settings.rerank_enabled and search_settings.rerank_use_llm:
            candidates = await self.reranker.rerank(query, candidates, top_k=limit)
        
        return candidates[:limit]
    
    def _apply_mmr(
        self,
        candidates: List[Dict[str, Any]],
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Apply MMR (Maximal Marginal Relevance) for diversity."""
        mmr_lambda = search_settings.hybrid_mmr_lambda
        
        if mmr_lambda >= 0.999:
            return candidates[:limit]
        
        for c in candidates:
            c["_tok"] = _tokset(c.get("text", "") + " " + c.get("path", ""))
        
        selected: List[Dict[str, Any]] = []
        selected_toks: List[set] = []
        
        if candidates:
            selected.append(candidates[0])
            selected_toks.append(candidates[0].get("_tok", set()))
        
        while len(selected) < limit:
            best = None
            best_val = -math.inf
            
            for c in candidates:
                if c in selected:
                    continue
                
                rel = c.get("score", 0)
                toks = c.get("_tok", set())
                max_sim = 0.0
                
                for st in selected_toks:
                    sim = _jaccard(toks, st)
                    if sim > max_sim:
                        max_sim = sim
                
                val = mmr_lambda * rel - (1.0 - mmr_lambda) * max_sim
                
                if val > best_val:
                    best_val = val
                    best = c
            
            if best is None:
                break
            
            selected.append(best)
            selected_toks.append(best.get("_tok", set()))
        
        return selected
    
    def _dedupe(
        self,
        candidates: List[Dict[str, Any]],
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Remove near-duplicate results."""
        threshold = search_settings.hybrid_dedupe_threshold
        
        import hashlib
        
        def _norm_text(s: str) -> str:
            s = (s or "").strip().lower()
            s = re.sub(r"\s+", " ", s)
            return s[:1200]
        
        def _sig(s: str) -> str:
            return hashlib.sha1(_norm_text(s).encode()).hexdigest()
        
        result: List[Dict[str, Any]] = []
        seen_hashes: set = set()
        seen_toksets: List[set] = []
        
        for c in candidates:
            if len(result) >= limit:
                break
            
            text = c.get("text", "")
            h = _sig(text)
            
            if h in seen_hashes:
                continue
            
            is_dup = False
            cur_tokset = _tokset(text)
            for seen_set in seen_toksets:
                if _jaccard(cur_tokset, seen_set) >= threshold:
                    is_dup = True
                    break
            
            if not is_dup:
                result.append(c)
                seen_hashes.add(h)
                seen_toksets.append(cur_tokset)
        
        return result


hybrid_search = HybridSearch()
