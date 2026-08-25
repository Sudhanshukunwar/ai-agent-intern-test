"""Parsing for the machine-readable trailer the model appends to every reply.

The system prompt requires every final response to end with a block like:

    ---
    SOURCES: 01-returns-policy-current.md; 09-trailplus-membership.md
    HANDOFF: false
    ---

This lets the evaluation harness make deterministic assertions about which
sources were cited and whether a human handoff was recommended, without
relying on another LLM to grade free-text prose. The trailer is stripped
from what's shown to the end user (see ``strip_trailer``); the CLI displays
sources/handoff separately using the parsed structured fields.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_TRAILER_RE = re.compile(
    r"-{2,}\s*\n"
    r"SOURCES:\s*(?P<sources>[^\n]*)\n"
    r"HANDOFF:\s*(?P<handoff>true|false)[.,;\s]*"
    r"(?:\n-{2,})?\s*$",
    re.IGNORECASE,
)


@dataclass
class ParsedResponse:
    display_text: str
    sources: list[str]
    handoff: bool
    trailer_found: bool


def parse_response(raw_text: str) -> ParsedResponse:
    match = _TRAILER_RE.search(raw_text.strip())
    if not match:
        # Fail safe: if the model didn't emit a parseable trailer, don't
        # silently assume no handoff is needed -- but do keep the full text
        # visible rather than hiding it.
        return ParsedResponse(display_text=raw_text.strip(), sources=[], handoff=False, trailer_found=False)

    display_text = raw_text[: match.start()].rstrip()
    sources_raw = match.group("sources").strip()
    sources = []
    if sources_raw and sources_raw.lower() not in {"none", "n/a", "-"}:
        for part in re.split(r"[;,]", sources_raw):
            name = part.strip()
            if name:
                sources.append(name)
    handoff = match.group("handoff").strip().lower() == "true"
    return ParsedResponse(display_text=display_text, sources=sources, handoff=handoff, trailer_found=True)
