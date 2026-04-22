from __future__ import annotations

from typing import List, Tuple

import numpy as np

try:
    from rank_bm25 import BM25Okapi
except Exception as e:
    raise ImportError("rank_bm25 is required for BM25 support. Install with `pip install rank-bm25`.") from e


class BM25Index:
    """Lightweight BM25 wrapper for a corpus of texts.

    Usage:
        bm = BM25Index([doc.page_content for doc in corpus])
        top = bm.top_n(query, 50)  # list of (doc_idx, score)
    """

    def __init__(self, corpus_texts: List[str]):
        self.corpus_texts = list(corpus_texts)
        self.tokenized_corpus = [self._tokenize(t) for t in self.corpus_texts]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        # Minimal whitespace tokenizer; replace with a better tokenizer if needed
        return text.split()

    def get_scores(self, query: str) -> np.ndarray:
        tokenized = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized)
        return np.array(scores, dtype=float)

    def top_n(self, query: str, n: int) -> List[Tuple[int, float]]:
        scores = self.get_scores(query)
        if scores.size == 0:
            return []
        idx = np.argsort(scores)[::-1][:n]
        return [(int(i), float(scores[i])) for i in idx]
