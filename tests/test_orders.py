import pytest

from app.orders import OrderStore


@pytest.fixture(scope="module")
def store():
    return OrderStore()


class TestNormalization:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("ORD-1007", "ORD-1007"),
            ("ord-1007", "ORD-1007"),
            ("  ord-1007  ", "ORD-1007"),
            ("Ord-1007", "ORD-1007"),
            ("ord_1007", "ORD-1007"),
            ("ord 1007", "ORD-1007"),
            ("ORD1007", "ORD-1007"),
            ("ord--1007", "ORD-1007"),
        ],
    )
    def test_normalizes_harmless_variations(self, raw, expected):
        assert OrderStore.normalize_order_id(raw) == expected

    def test_does_not_guess_a_different_id(self):
        # "ORD-107" should stay "ORD-107", not silently become "ORD-1007".
        assert OrderStore.normalize_order_id("ORD-107") == "ORD-107"

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("ORD-1007?", "ORD-1007"),
            ("ORD-1007.", "ORD-1007"),
            ("ORD-1007!", "ORD-1007"),
            (" ORD-1007, ", "ORD-1007"),
        ],
    )
    def test_bug_diary_1_strips_stray_trailing_punctuation(self, raw, expected):
        # Regression test for bug diary #1: "Where's ORD-1007?" was being
        # normalized to "ORD-1007?" and then rejected as malformed because
        # the trailing "?" survived normalization.
        assert OrderStore.normalize_order_id(raw) == expected


class TestLookup:
    def test_valid_order_found(self, store):
        result = store.lookup("ord-1007")
        assert result.found
        assert result.data["order_id"] == "ORD-1007"
        assert result.data["status"] == "shipped"

    def test_unknown_order(self, store):
        result = store.lookup("ORD-9999")
        assert not result.found
        assert result.error == "not_found"

    def test_malformed_order_id(self, store):
        result = store.lookup("asdf123")
        assert not result.found
        assert result.error == "malformed"

    def test_order_id_with_trailing_punctuation_still_resolves(self, store):
        # Regression test for bug diary #1.
        result = store.lookup("ORD-1007?")
        assert result.found
        assert result.data["order_id"] == "ORD-1007"

    def test_empty_input(self, store):
        result = store.lookup("   ")
        assert not result.found
        assert result.error == "malformed"


class TestPrivacySanitization:
    def test_no_pii_fields_present(self, store):
        result = store.lookup("ORD-1007")
        data = result.data
        assert "customer" not in data
        assert "internal" not in data
        serialized = str(data)
        # These would only appear if we'd leaked the raw customer/internal dicts.
        assert "ava.morgan@example.test" not in serialized
        assert "King Street" not in serialized
        assert "fraud review" not in serialized
        assert "82" not in [str(v) for v in data.values()]  # risk_score=82 must not leak

    def test_items_only_expose_safe_item_fields(self, store):
        result = store.lookup("ORD-1007")
        for item in result.data["items"]:
            assert set(item.keys()) == {"name", "quantity", "final_sale"}


class TestStatusPrecedence:
    def test_cancelled_order_hides_stale_eta(self, store):
        result = store.lookup("ORD-1004")
        assert result.data["status"] == "cancelled"
        assert result.data["estimated_delivery"] is None
        assert result.data["carrier"] is None

    def test_returned_order_hides_stale_eta(self, store):
        result = store.lookup("ORD-1008")
        assert result.data["status"] == "returned"
        assert result.data["estimated_delivery"] is None

    def test_shipped_without_eta_flags_unavailable(self, store):
        result = store.lookup("ORD-1011")
        assert result.data["status"] == "shipped"
        assert result.data["estimated_delivery"] is None
        assert "unavailable" in result.data["delivery_estimate_note"]

    def test_exception_status_flags_review(self, store):
        result = store.lookup("ORD-1010")
        assert result.data["status"] == "exception"
        assert "human support" in result.data["delivery_estimate_note"]

    def test_normal_shipped_order_keeps_eta(self, store):
        result = store.lookup("ORD-1007")
        assert result.data["status"] == "shipped"
        assert result.data["estimated_delivery"] == "2026-08-22"
