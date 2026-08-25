"""A tiny, dependency-free parser for the simple YAML front matter used in
knowledge-base/*.md.

Replaces ``pyyaml``. The front matter in this project is always flat
``key: value`` scalar pairs (document_id, title, status, effective_date,
last_reviewed, audience, policy_authority, supersedes, superseded_by) --
no lists, no nesting -- so a full YAML parser is more machinery than the
actual format needs. If a document ever needs real YAML (lists/nested
maps), swap this for ``pyyaml`` at that point; ``retrieval.py`` only calls
``parse_front_matter`` so the swap is a one-line change.
"""
from __future__ import annotations


def parse_front_matter(text: str) -> dict:
    result: dict = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if value.startswith(("'", '"')) and value.endswith(("'", '"')) and len(value) >= 2:
            value = value[1:-1]
        if value.lower() == "null" or value == "":
            value = None
        result[key] = value
    return result
