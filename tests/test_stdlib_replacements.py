from app.bm25 import BM25
from app.simple_yaml import parse_front_matter


class TestBM25:
    def test_ranks_exact_term_match_highest(self):
        corpus = [
            ["apple", "banana", "cherry"],
            ["dog", "cat", "bird"],
            ["apple", "apple", "orange"],
        ]
        bm25 = BM25(corpus)
        scores = bm25.get_scores(["apple"])
        assert scores[2] > scores[0] > scores[1]

    def test_unknown_query_term_scores_zero_everywhere(self):
        corpus = [["apple", "banana"], ["cherry", "date"]]
        bm25 = BM25(corpus)
        scores = bm25.get_scores(["nonexistent"])
        assert scores == [0.0, 0.0]

    def test_empty_corpus_does_not_crash(self):
        bm25 = BM25([])
        assert bm25.get_scores(["anything"]) == []

    def test_longer_matching_document_scored_relative_to_avg_length(self):
        # BM25's length normalization means a short doc with the same term
        # frequency as a long doc should score at least as high.
        corpus = [["apple"] * 1, ["apple"] * 1 + ["filler"] * 50]
        bm25 = BM25(corpus)
        scores = bm25.get_scores(["apple"])
        assert scores[0] >= scores[1]


class TestFrontMatterParser:
    def test_parses_simple_key_values(self):
        text = """
document_id: RET-2026-01
title: Returns Policy
status: active
policy_authority: official
"""
        meta = parse_front_matter(text)
        assert meta["document_id"] == "RET-2026-01"
        assert meta["title"] == "Returns Policy"
        assert meta["status"] == "active"
        assert meta["policy_authority"] == "official"

    def test_strips_quotes(self):
        meta = parse_front_matter('title: "Quoted Title"')
        assert meta["title"] == "Quoted Title"

    def test_empty_and_null_values(self):
        meta = parse_front_matter("supersedes:\nsuperseded_by: null")
        assert meta["supersedes"] is None
        assert meta["superseded_by"] is None

    def test_ignores_blank_lines_and_comments(self):
        text = """
# a comment
document_id: RET-2026-01

title: Returns Policy
"""
        meta = parse_front_matter(text)
        assert meta == {"document_id": "RET-2026-01", "title": "Returns Policy"}

    def test_matches_real_document_front_matter(self):
        from app.retrieval import KnowledgeBase

        kb = KnowledgeBase()
        current = [c for c in kb.chunks if c.filename == "01-returns-policy-current.md"][0]
        assert current.status == "active"
        assert current.policy_authority == "official"
