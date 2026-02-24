import re
import math
from collections import Counter
from typing import List, Dict, Any


class BM25:
    """BM25 ranking algorithm implementation."""
    
    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.k1 = k1
        self.b = b
        self.corpus: List[Dict[str, Any]] = []
        self.avgdl: float = 0.0
        self.doc_freqs: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.doc_len: List[int] = []
        self.doc_tokens: List[List[str]] = []
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text - supports Cyrillic and Latin."""
        text = text.lower()
        tokens = re.findall(r"[^\W_]{2,}", text, flags=re.UNICODE)
        return tokens
    
    def _preprocess(self, documents: List[Dict[str, Any]]) -> None:
        """Preprocess corpus for BM25."""
        self.corpus = documents
        self.doc_tokens = []
        self.doc_len = []
        
        for doc in documents:
            text = doc.get("text", "") or doc.get("snippet", "") or ""
            tokens = self._tokenize(text)
            self.doc_tokens.append(tokens)
            self.doc_len.append(len(tokens))
        
        self.avgdl = sum(self.doc_len) / max(len(self.doc_len), 1)
        
        df: Counter[str] = Counter()
        for tokens in self.doc_tokens:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                df[token] += 1
        
        self.doc_freqs = dict(df)
        
        N = len(self.corpus)
        for term, freq in self.doc_freqs.items():
            self.idf[term] = math.log((N - freq + 0.5) / (freq + 0.5) + 1)
    
    def score(self, query: str, doc_index: int) -> float:
        """Calculate BM25 score for a query against a document."""
        if not self.corpus or doc_index >= len(self.corpus):
            return 0.0
        
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return 0.0
        
        doc_len = self.doc_len[doc_index]
        doc_tokens = self.doc_tokens[doc_index]
        doc_tf = Counter(doc_tokens)
        
        score = 0.0
        for term in query_tokens:
            if term not in self.idf:
                continue
            
            tf = doc_tf.get(term, 0)
            idf = self.idf[term]
            
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / max(self.avgdl, 1))
            
            score += idf * (numerator / max(denominator, 0.001))
        
        return score
    
    def get_scores(self, query: str) -> List[float]:
        """Get BM25 scores for all documents."""
        return [self.score(query, i) for i in range(len(self.corpus))]
    
    def get_top_k(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Get top-k documents for a query."""
        scores = self.get_scores(query)
        
        doc_scores = []
        for i, score in enumerate(scores):
            if score > 0:
                doc = dict(self.corpus[i])
                doc["bm25_score"] = score
                doc_scores.append(doc)
        
        doc_scores.sort(key=lambda x: x["bm25_score"], reverse=True)
        return doc_scores[:k]


class BM25Ranker:
    """BM25 ranker that works with search results."""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.bm25 = BM25(k1=k1, b=b)
    
    def rank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Rank documents using BM25."""
        if not documents or not query:
            return documents[:k]
        
        self.bm25._preprocess(documents)
        results = self.bm25.get_top_k(query, k=k)
        
        for doc in results:
            doc["source"] = "bm25"
        
        return results

    def score_all(self, query: str, documents: List[Dict[str, Any]]) -> List[float]:
        """Return BM25 scores aligned with input documents order."""
        if not documents or not query:
            return [0.0] * len(documents)
        self.bm25._preprocess(documents)
        return [float(v) for v in self.bm25.get_scores(query)]


bm25_ranker = BM25Ranker()
