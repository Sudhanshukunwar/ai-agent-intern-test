import pytest

from app.retrieval import KnowledgeBase


@pytest.fixture(scope="module")
def kb():
    return KnowledgeBase()


class TestLoading:
    def test_loads_all_documents(self, kb):
        filenames = {c.filename for c in kb.chunks}
        assert len(filenames) == 14

    def test_front_matter_metadata_captured(self, kb):
        current = [c for c in kb.chunks if c.filename == "01-returns-policy-current.md"]
        assert current
        assert current[0].status == "active"
        assert current[0].policy_authority == "official"

    def test_chunks_have_headings(self, kb):
        current = [c for c in kb.chunks if c.filename == "01-returns-policy-current.md"]
        headings = {c.heading for c in current}
        assert "Standard return window" in headings
        assert "Return shipping and refunds" in headings


class TestPrecedenceRanking:
    def test_current_returns_policy_outranks_legacy(self, kb):
        results = kb.search("how many days do I have to return an item")
        top_filenames = [r.chunk.filename for r in results[:3]]
        assert "01-returns-policy-current.md" in top_filenames
        current_rank = top_filenames.index("01-returns-policy-current.md")
        if "02-returns-policy-legacy.md" in top_filenames:
            legacy_rank = top_filenames.index("02-returns-policy-legacy.md")
            assert current_rank < legacy_rank

    def test_draft_migration_doc_is_deprioritized_but_findable(self, kb):
        # It should still be retrievable (the agent needs to be able to see
        # and dismiss it), just not ranked above the real policy.
        results = kb.search("60 days return everything including gift cards migration")
        filenames = [r.chunk.filename for r in results]
        assert "14-internal-content-migration-notes.md" in filenames

    def test_both_breeze_tumbler_docs_retrievable_for_conflict_question(self, kb):
        results = kb.search("can I put the breeze tumbler in the dishwasher")
        filenames = {r.chunk.filename for r in results}
        assert "11-product-care.md" in filenames
        assert "12-breeze-tumbler-product-card.md" in filenames

    def test_canada_multiturn_query_hits_international_doc(self, kb):
        results = kb.search("do you ship internationally what about canada how long does it take")
        filenames = [r.chunk.filename for r in results]
        assert "06-international-shipping.md" in filenames


class TestEmptyQuery:
    def test_empty_query_returns_nothing(self, kb):
        assert kb.search("") == []
        assert kb.search("   ") == []


class TestConfidenceFlag:
    def test_out_of_scope_question_tail_results_are_low_confidence(self, kb):
        # Regression test for bug diary #3: "are your fabrics/adhesives
        # vegan?" is not answered anywhere in the knowledge base, but BM25
        # still returns plausible-looking chunks purely on shared
        # vocabulary. Before the fix there was no confidence signal at all.
        # Note: a pure keyword scorer can't perfectly separate "shares rare
        # vocabulary" from "actually answers the question" -- a chunk that
        # happens to contain a rare query term (e.g. "fabric") can still
        # score above the floor. The floor reliably catches the weaker tail
        # of the result set, which is what it's for; it is a heuristic aid
        # for the model, not a substitute for the model checking whether a
        # passage actually supports the claim (see system prompt and the
        # groundedness/abstention rules, which are the real backstop here).
        results = kb.search("are all fabrics and adhesives in your bags vegan")
        assert results
        assert results[-1].confidence == "low"
        assert any(r.confidence == "low" for r in results)

    def test_strong_direct_match_is_normal_confidence(self, kb):
        results = kb.search("how long does a regular customer have to return an unused backpack")
        top = results[0]
        assert top.chunk.filename == "01-returns-policy-current.md"
        assert top.confidence == "normal"

    def test_confidence_floor_is_respected(self, kb):
        results = kb.search("are all fabrics and adhesives in your bags vegan")
        for r in results:
            expected = "low" if r.score < 6.0 else "normal"
            assert r.confidence == expected
