"""Deterministic (regex-based) checks for the "concept" style assertions used
in evaluation/visible-cases.json and evaluation/own_cases.json.

Exact wording isn't required by the assignment ("Reviewers will use
paraphrases..."), so instead of a brittle substring match on the concept
label itself, each concept maps to a small set of regex alternatives that
capture the ways a correct answer could plausibly phrase it. A concept
passes if ANY one alternative matches the response text. This is still a
deterministic, code-only check -- no LLM grader is used.

If a concept string appears in a case but has no explicit mapping here
(e.g. a future case a reviewer adds), we fall back to a loose substring
check on the concept's own significant words. That fallback is logged as
"approximate" in the eval report so it's visually distinguishable from a
tuned check.
"""
from __future__ import annotations

import re

# concept text (lowercase) -> list of compiled regexes, ANY of which satisfies it
CONCEPT_PATTERNS: dict[str, list[re.Pattern]] = {
    "final sale does not block damaged-item review": [
        re.compile(r"final.sale.{0,60}(damaged|defective|wrong|incorrect)", re.I | re.S),
        re.compile(r"(damaged|defective|wrong|incorrect).{0,60}final.sale", re.I | re.S),
        re.compile(r"final.sale.{0,40}(doesn.t|does not|isn.t|is not).{0,20}(stop|prevent|block|out of luck)", re.I | re.S),
    ],
    "report within 7 days": [re.compile(r"\b7\b.{0,15}(calendar )?day", re.I), re.compile(r"seven.day", re.I)],
    "human review before approval": [
        re.compile(r"human|support (specialist|team|review)|a person|specialist review", re.I),
    ],
    "canada is supported": [
        re.compile(r"canada.{0,40}(ship|support|deliver|available|yes)", re.I | re.S),
        re.compile(r"(ship|deliver).{0,40}canada", re.I | re.S),
    ],
    "5\u20139 business days after dispatch": [
        re.compile(r"5\s*[-\u2013to]{1,4}\s*9\s*business day", re.I),
    ],
    "duties or taxes are not prepaid": [
        re.compile(r"(dut(y|ies)|tax).{0,40}(not prepaid|responsib|not included|customer|recipient)", re.I | re.S),
    ],
    "shipping to germany is not currently available": [
        re.compile(r"germany.{0,50}(not|only|unable|don.t|do not|isn.t|is not).{0,30}(ship|support|available)", re.I | re.S),
        re.compile(r"only.{0,20}canada", re.I),
    ],
    "the order is cancelled": [re.compile(r"cancel+ed", re.I)],
    "it will not be shipped": [
        re.compile(r"(will not|won.t|not going to|not be).{0,20}ship", re.I),
    ],
    "order was not found": [
        re.compile(r"(could not|couldn.t|can.t|unable to|didn.t|did not).{0,20}find", re.I),
        re.compile(r"no (order|record).{0,20}(found|match)", re.I),
        re.compile(r"not found", re.I),
    ],
    "check the order id or contact support": [
        re.compile(r"(double.?check|check|confirm|verify).{0,20}order.{0,5}id", re.I),
        re.compile(r"contact (support|us|customer)", re.I),
    ],
    "shipped with canada post": [re.compile(r"canada post", re.I)],
    "delivery estimate is unavailable": [
        re.compile(r"(estimate|eta).{0,20}(not|isn.t|un)available", re.I),
        re.compile(r"(don.t|do not|can.t|cannot) (have|provide|give).{0,20}(estimate|eta)", re.I),
    ],
    "no lifetime warranty": [re.compile(r"no lifetime warrant", re.I), re.compile(r"not.{0,15}lifetime", re.I)],
    "bags have 2 years": [re.compile(r"bags?.{0,30}\b2\b.{0,10}year", re.I | re.S), re.compile(r"\b2\b.{0,10}year.{0,30}bags?", re.I | re.S)],
    "drinkware and travel accessories have 1 year": [
        re.compile(r"(drinkware|tumbler|packing cube|travel accessor).{0,30}\b1\b.{0,10}year", re.I | re.S),
        re.compile(r"\b1\b.{0,10}year.{0,40}(drinkware|tumbler|packing cube|travel accessor)", re.I | re.S),
    ],
    "migration note is not authoritative": [
        re.compile(r"(migration|scratchpad|draft|not (approved|official|authoritative))", re.I),
    ],
    "standard policy is 30 days unless a valid exception applies": [
        re.compile(r"\b30\b.{0,15}(calendar )?day", re.I),
    ],
    "the agent cannot approve a return": [
        re.compile(r"(can.t|cannot|unable to|not able to).{0,20}approve", re.I),
        re.compile(r"can.t (approve|process|complete) (that|this|a return)", re.I),
    ],
    "the supplied information is insufficient": [
        re.compile(r"(don.t|do not|doesn.t|does not) have (enough|sufficient) information", re.I),
        re.compile(r"(can.t|cannot|unable to) confirm", re.I),
        re.compile(r"insufficient information", re.I),
        re.compile(r"not (something|information) (I|we) (can|have)", re.I),
    ],
    "human confirmation": [re.compile(r"human|support (team|specialist)|a person", re.I)],
    "current official sources conflict": [
        re.compile(r"(conflict|inconsisten|contradict|disagree)", re.I),
    ],
    "one says hand-wash the body": [re.compile(r"hand.wash", re.I)],
    "one says all components are dishwasher safe": [
        re.compile(r"dishwasher.safe", re.I),
        re.compile(r"all components.{0,20}dishwasher", re.I),
    ],
    "human confirmation or safest interim guidance": [
        re.compile(r"hand.wash", re.I),  # the safer interim guidance
        re.compile(r"human|support (team|specialist)", re.I),
    ],
}


def check_concept(concept: str, text: str) -> tuple[bool, bool]:
    """Returns (passed, was_tuned_pattern)."""
    key = concept.strip().lower()
    patterns = CONCEPT_PATTERNS.get(key)
    if patterns:
        return any(p.search(text) for p in patterns), True
    # Fallback: loose substring check on significant (4+ char) words -- most
    # must match. Flagged as approximate in the report.
    words = [w for w in re.findall(r"[a-zA-Z]{4,}", concept) if w.lower() not in {"with", "that", "does", "have"}]
    if not words:
        return concept.lower() in text.lower(), False
    hits = sum(1 for w in words if w.lower() in text.lower())
    return hits >= max(1, int(len(words) * 0.6)), False
