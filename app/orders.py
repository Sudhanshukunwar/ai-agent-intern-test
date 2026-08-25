"""Order-status lookup tool.

The model never sees ``data/orders.json`` in full. It can only call
``order_lookup(order_id)``, which returns a sanitized dict containing only
customer-safe fields (per ``data/orders-data-dictionary.md``). Internal
fields (customer PII, risk scores, warehouse/support notes) are stripped
here, in code, before anything reaches the model -- not by asking the model
to politely avoid repeating them.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

from app.config import ORDERS_PATH

_ORDER_ID_RE = re.compile(r"^ORD-\d+$")

# Fields from the data dictionary's "Customer-safe fields" list.
_SAFE_TOP_LEVEL_FIELDS = (
    "order_id",
    "membership_tier",
    "placed_at",
    "status",
    "status_updated_at",
    "shipped_at",
    "delivered_at",
    "carrier",
    "tracking_number",
    "estimated_delivery",
    "customer_safe_message",
)
_SAFE_ITEM_FIELDS = ("name", "quantity", "final_sale")

# Statuses where a stale estimated_delivery/carrier must not be presented as
# "still arriving".
_TERMINAL_NON_DELIVERY_STATUSES = {"cancelled", "returned"}


@dataclass
class OrderLookupResult:
    found: bool
    order_id_normalized: str
    data: Optional[dict] = None
    error: Optional[str] = None  # "not_found" | "malformed"

    def to_tool_output(self) -> dict:
        """What actually gets sent back to the model as the tool result."""
        if not self.found:
            return {"found": False, "order_id_queried": self.order_id_normalized, "error": self.error}
        return {"found": True, **self.data}


class OrderStore:
    def __init__(self, path=ORDERS_PATH):
        self.path = path
        raw = json.loads(path.read_text(encoding="utf-8"))
        self.snapshot_at: str = raw.get("snapshot_at", "")
        self.dataset_name: str = raw.get("dataset_name", "")
        self._by_id: dict[str, dict] = {o["order_id"]: o for o in raw.get("orders", [])}

    @staticmethod
    def normalize_order_id(raw_id: str) -> str:
        """Normalize harmless differences: case, surrounding whitespace,
        ordinary punctuation/spacing around the dash, and stray trailing
        punctuation a customer (or the model, copying from a sentence) might
        include, e.g. "ORD-1007?" or "ORD-1007.". Does NOT guess a different
        order id -- it only canonicalizes formatting.
        """
        candidate = raw_id.strip().upper()
        # Strip stray leading/trailing punctuation like "?" "." "!" that
        # naturally show up when an order ID is copied out of a sentence
        # (bug diary #1: "Where's ORD-1007?" previously left the "?" in
        # place and the ID was wrongly reported as malformed).
        candidate = candidate.strip(" \t.,!?;:")
        # Collapse things like "ord 1007", "ord_1007", "ord--1007" -> "ORD-1007"
        candidate = re.sub(r"[\s_]+", "-", candidate)
        candidate = re.sub(r"-{2,}", "-", candidate)
        # Insert a dash if someone wrote "ORD1007" with no separator at all.
        candidate = re.sub(r"^ORD(\d)", r"ORD-\1", candidate)
        return candidate

    def lookup(self, raw_order_id: str) -> OrderLookupResult:
        normalized = self.normalize_order_id(raw_order_id)
        if not _ORDER_ID_RE.match(normalized):
            return OrderLookupResult(found=False, order_id_normalized=normalized, error="malformed")
        order = self._by_id.get(normalized)
        if order is None:
            return OrderLookupResult(found=False, order_id_normalized=normalized, error="not_found")
        sanitized = self._sanitize(order)
        return OrderLookupResult(found=True, order_id_normalized=normalized, data=sanitized)

    def _sanitize(self, order: dict) -> dict:
        """Build a fresh dict containing only customer-safe fields.

        Deliberately an allow-list (only ever *adding* known-safe keys) and
        not a copy-then-delete of the full order record -- a deny-list
        approach is one field-name typo away from leaking `internal` or
        `customer` to the model.
        """
        out: dict = {}
        for field in _SAFE_TOP_LEVEL_FIELDS:
            out[field] = order.get(field)

        out["items"] = [
            {k: item.get(k) for k in _SAFE_ITEM_FIELDS} for item in order.get("items", [])
        ]

        status = order.get("status")
        if status in _TERMINAL_NON_DELIVERY_STATUSES:
            # Stale operational fields must not be presented as "in transit".
            out["carrier"] = None
            out["tracking_number"] = None
            out["estimated_delivery"] = None
            out["delivery_estimate_note"] = (
                "not applicable; order is " + status
            )
        elif status == "shipped" and not out.get("estimated_delivery"):
            out["delivery_estimate_note"] = "unavailable; do not invent a date"
        elif status == "exception":
            out["delivery_estimate_note"] = "unavailable; requires human support review"

        return out
