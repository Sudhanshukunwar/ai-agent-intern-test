"""A small, dependency-free BM25 implementation.

This replaces the third-party ``rank_bm25`` package, which pulls in numpy
as a transitive dependency. numpy ships prebuilt wheels for essentially
every mainstream platform, so needing this at all is itself a sign of an
unusual local pip/network environment -- but since the whole point of this
project is to run cleanly from a clean clone, it's more robust to just not
depend on a native/compiled package chain for a few dozen short documents.
This is the standard BM25 (Okapi) formula, unvectorized -- fine at this
corpus size (a handful of documents), not intended to scale to a large
corpus.
"""
from __future__ import annotations

import math
from collections import Counter


class BM25:
    def __init__(self, tokenized_corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = tokenized_corpus
        self.doc_freqs: list[Counter] = [Counter(doc) for doc in tokenized_corpus]
        self.doc_lengths = [len(doc) for doc in tokenized_corpus]
        self.avg_doc_length = (sum(self.doc_lengths) / len(self.doc_lengths)) if tokenized_corpus else 0.0
        self.n_docs = len(tokenized_corpus)

        df: Counter = Counter()
        for doc in tokenized_corpus:
            for term in set(doc):
                df[term] += 1
        self.idf: dict[str, float] = {}
        for term, freq in df.items():
            # Standard BM25 idf with the +1 smoothing term so idf stays
            # non-negative for common terms.
            self.idf[term] = math.log(1 + (self.n_docs - freq + 0.5) / (freq + 0.5))

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores = [0.0] * self.n_docs
        for term in query_tokens:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i in range(self.n_docs):
                freq = self.doc_freqs[i].get(term, 0)
                if freq == 0:
                    continue
                denom = freq + self.k1 * (1 - self.b + self.b * self.doc_lengths[i] / (self.avg_doc_length or 1))
                scores[i] += idf * (freq * (self.k1 + 1)) / denom
        return scores
