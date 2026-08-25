"""Retrieval over the markdown knowledge base.

Uses deterministic BM25 with metadata-aware reranking and balanced
document/chunk selection.

The retriever:
- splits Markdown documents by ## headings;
- preserves front-matter metadata;
- prefers active official documents;
- considers title, filename, heading, and query intent;
- preserves genuine conflicts between authoritative documents;
- retrieves multiple complementary sections when necessary;
- avoids allowing one document to consume the entire retrieval budget.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.bm25 import BM25
from app.config import (
    KNOWLEDGE_BASE_DIR,
    LOW_CONFIDENCE_FLOOR,
    TOP_K_CHUNKS,
)
from app.simple_yaml import parse_front_matter


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class Chunk:
    doc_id: str
    filename: str
    title: str
    heading: str
    text: str

    status: str = "unknown"
    policy_authority: str = "unknown"
    audience: str = "customer"
    supersedes: Optional[str] = None
    superseded_by: Optional[str] = None
    effective_date: Optional[str] = None

    @property
    def source_label(self) -> str:
        return f"{self.filename} § {self.heading}"

    @property
    def is_authoritative(self) -> bool:
        return (
            self.status == "active"
            and self.policy_authority == "official"
        )


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float

    @property
    def confidence(self) -> str:
        return (
            "low"
            if self.score < LOW_CONFIDENCE_FLOOR
            else "normal"
        )


def _split_front_matter(raw: str) -> tuple[dict, str]:
    if raw.startswith("---"):
        parts = raw.split("---", 2)

        if len(parts) >= 3:
            meta = parse_front_matter(parts[1]) or {}
            body = parts[2].strip()
            return meta, body

    return {}, raw


def _chunk_body(body: str) -> list[tuple[str, str]]:
    """Split a document into coherent sections using ## headings."""

    lines = body.splitlines()

    chunks: list[tuple[str, str]] = []

    current_heading = "(document intro)"
    current_lines: list[str] = []

    def flush() -> None:
        text = "\n".join(current_lines).strip()

        if text:
            chunks.append(
                (
                    current_heading,
                    text,
                )
            )

    for line in lines:
        if line.startswith("## "):
            flush()
            current_heading = line[3:].strip()
            current_lines = []

        elif line.startswith("# "):
            continue

        else:
            current_lines.append(line)

    flush()

    return chunks


class KnowledgeBase:

    def __init__(
        self,
        directory: Path = KNOWLEDGE_BASE_DIR,
    ):
        self.directory = directory

        self.chunks: list[Chunk] = []

        self._bm25: Optional[BM25] = None
        self._tokenized: list[list[str]] = []

        self._load()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self) -> None:

        for path in sorted(
            self.directory.glob("*.md")
        ):
            raw = path.read_text(
                encoding="utf-8"
            )

            meta, body = _split_front_matter(raw)

            title = meta.get(
                "title",
                path.stem,
            )

            for heading, text in _chunk_body(body):

                self.chunks.append(
                    Chunk(
                        doc_id=str(
                            meta.get(
                                "document_id",
                                path.stem,
                            )
                        ),
                        filename=path.name,
                        title=str(title),
                        heading=heading,
                        text=text,
                        status=str(
                            meta.get(
                                "status",
                                "unknown",
                            )
                        ),
                        policy_authority=str(
                            meta.get(
                                "policy_authority",
                                "unknown",
                            )
                        ),
                        audience=str(
                            meta.get(
                                "audience",
                                "customer",
                            )
                        ),
                        supersedes=meta.get(
                            "supersedes"
                        ),
                        superseded_by=meta.get(
                            "superseded_by"
                        ),
                        effective_date=(
                            str(
                                meta.get(
                                    "effective_date",
                                    "",
                                )
                            )
                            or None
                        ),
                    )
                )

        self._tokenized = [
            _tokenize(
                " ".join(
                    [
                        chunk.title,
                        chunk.heading,
                        chunk.text,
                    ]
                )
            )
            for chunk in self.chunks
        ]

        if self._tokenized:
            self._bm25 = BM25(
                self._tokenized
            )

    # ------------------------------------------------------------------
    # Authority / precedence
    # ------------------------------------------------------------------

    def _status_weight(
        self,
        chunk: Chunk,
    ) -> float:

        weight = 1.0

        if (
            chunk.status == "active"
            and chunk.policy_authority == "official"
        ):
            weight *= 1.35

        elif chunk.status == "superseded":
            weight *= 0.55

        elif (
            chunk.status == "draft"
            or chunk.policy_authority == "none"
        ):
            weight *= 0.50

        if chunk.audience == "internal":
            weight *= 0.70

        return weight

    # ------------------------------------------------------------------
    # Metadata relevance
    # ------------------------------------------------------------------

    @staticmethod
    def _metadata_score(
        query_tokens: set[str],
        chunk: Chunk,
    ) -> float:

        if not query_tokens:
            return 0.0

        title_tokens = set(
            _tokenize(chunk.title)
        )

        heading_tokens = set(
            _tokenize(chunk.heading)
        )

        filename_tokens = set(
            _tokenize(
                Path(chunk.filename)
                .stem
                .replace("-", " ")
            )
        )

        return (
            len(query_tokens & title_tokens) * 3.0
            + len(query_tokens & heading_tokens) * 2.0
            + len(query_tokens & filename_tokens) * 1.5
        )

    # ------------------------------------------------------------------
    # Query intent
    # ------------------------------------------------------------------

    @staticmethod
    def _intent_score(
        query_tokens: set[str],
        chunk: Chunk,
    ) -> float:

        filename = chunk.filename.lower()
        title = chunk.title.lower()
        heading = chunk.heading.lower()

        searchable = (
            f"{filename} {title} {heading}"
        )

        score = 0.0

        # Return policy
        if (
            "return" in query_tokens
            or "returns" in query_tokens
        ):
            if (
                "return" in searchable
                or "returns" in searchable
            ):
                score += 4.0

        # TrailPlus
        if (
            "trailplus" in query_tokens
            or (
                "membership" in query_tokens
                and "return" in query_tokens
            )
        ):
            if (
                "trailplus" in searchable
                or "membership" in searchable
            ):
                score += 5.0

        # Final sale
        if (
            "final" in query_tokens
            and "sale" in query_tokens
        ):
            if (
                "final-sale" in filename
                or "final sale" in searchable
            ):
                score += 4.0

        # Damaged / defective / wrong
        if (
            "damaged" in query_tokens
            or "broken" in query_tokens
            or "defective" in query_tokens
            or "wrong" in query_tokens
        ):
            if (
                "damaged" in searchable
                or "defective" in searchable
                or "wrong" in searchable
            ):
                score += 5.0

        # Reporting deadline
        if (
            "when" in query_tokens
            or "days" in query_tokens
            or "yesterday" in query_tokens
            or "report" in query_tokens
            or "reported" in query_tokens
        ):
            if (
                "reporting" in searchable
                or "window" in searchable
                or "deadline" in searchable
            ):
                score += 5.0

        # Review / resolution / approval
        if (
            "review" in query_tokens
            or "approve" in query_tokens
            or "approval" in query_tokens
            or "refund" in query_tokens
            or "replacement" in query_tokens
        ):
            if (
                "review" in searchable
                or "resolution" in searchable
                or "approval" in searchable
            ):
                score += 4.0

        # International shipping
        if (
            "international" in query_tokens
            or "canada" in query_tokens
            or "germany" in query_tokens
            or "ship" in query_tokens
            or "shipping" in query_tokens
        ):
            if (
                "international" in searchable
                or "shipping" in searchable
            ):
                score += 4.0

        # Warranty
        if (
            "warranty" in query_tokens
            or "warranties" in query_tokens
        ):
            if "warranty" in searchable:
                score += 5.0

        # Order changes / cancellation
        if (
            "cancel" in query_tokens
            or "cancellation" in query_tokens
            or "change" in query_tokens
        ):
            if (
                "order-changes" in filename
                or "cancellation" in searchable
            ):
                score += 4.0

        # Product care
        if (
            "dishwasher" in query_tokens
            or "wash" in query_tokens
            or "care" in query_tokens
        ):
            if (
                "product-care" in filename
                or "product care" in searchable
            ):
                score += 4.0

        # Breeze Tumbler
        if (
            "breeze" in query_tokens
            or "tumbler" in query_tokens
        ):
            if (
                "breeze" in searchable
                or "tumbler" in searchable
            ):
                score += 5.0

        return score

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = TOP_K_CHUNKS,
    ) -> list[RetrievedChunk]:

        if (
            not self._bm25
            or not query.strip()
        ):
            return []

        tokens = _tokenize(query)

        if not tokens:
            return []

        query_tokens = set(tokens)

        # Expand common support intents without changing the original
        # query used for metadata/intent scoring. This helps retrieve
        # complementary policy sections whose headings use different
        # terminology from the customer's wording.
        bm25_tokens = list(tokens)

        if (
            "final" in query_tokens
            and "sale" in query_tokens
            and (
                "damaged" in query_tokens
                or "broken" in query_tokens
                or "defective" in query_tokens
                or "wrong" in query_tokens
            )
        ):
            bm25_tokens.extend(
                [
                    "damaged",
                    "defective",
                    "wrong",
                    "reporting",
                    "window",
                    "days",
                    "delivery",
                    "review",
                    "resolution",
                ]
            )

        raw_scores = self._bm25.get_scores(
            bm25_tokens
        )

        candidates: list[RetrievedChunk] = []

        for chunk, raw_score in zip(
            self.chunks,
            raw_scores,
        ):

            raw_score = float(raw_score)

            metadata_score = (
                self._metadata_score(
                    query_tokens,
                    chunk,
                )
            )

            intent_score = (
                self._intent_score(
                    query_tokens,
                    chunk,
                )
            )

            if (
                raw_score <= 0
                and metadata_score <= 0
                and intent_score <= 0
            ):
                continue

            combined_score = (
                raw_score
                + metadata_score
                + intent_score
            )

            weighted_score = (
                combined_score
                * self._status_weight(chunk)
            )

            candidates.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=weighted_score,
                )
            )

        candidates.sort(
            key=lambda result: result.score,
            reverse=True,
        )

        if not candidates:
            return []

        # --------------------------------------------------------------
        # Balanced selection
        # --------------------------------------------------------------
        #
        # We want:
        #
        #   - the strongest relevant document;
        #   - other genuinely relevant documents;
        #   - additional complementary sections from the same document;
        #
        # without allowing one document to consume the entire top_k.
        # --------------------------------------------------------------

        selected: list[RetrievedChunk] = []
        selected_ids: set[int] = set()

        # Group candidates by filename.
        by_document: dict[
            str,
            list[RetrievedChunk],
        ] = {}

        for result in candidates:
            by_document.setdefault(
                result.chunk.filename,
                [],
            ).append(result)

        # Keep each document's candidates sorted.
        for results in by_document.values():
            results.sort(
                key=lambda result: result.score,
                reverse=True,
            )

        # --------------------------------------------------------------
        # Pass 1:
        # Strongest chunk from each document.
        #
        # This prevents a single document from monopolizing retrieval.
        # --------------------------------------------------------------

        document_heads = sorted(
            (
                results[0]
                for results in by_document.values()
                if results
            ),
            key=lambda result: result.score,
            reverse=True,
        )

        for result in document_heads:

            selected.append(result)
            selected_ids.add(id(result))

            if len(selected) >= top_k:
                break

        # --------------------------------------------------------------
        # Pass 2:
        # Add complementary chunks.
        #
        # A second chunk from a document is allowed when:
        # - it is strongly relevant, and
        # - it adds a meaningful signal such as reporting, review,
        #   exception, resolution, or another directly related heading.
        # --------------------------------------------------------------

        complementary_keywords = {
            "report",
            "reporting",
            "window",
            "deadline",
            "review",
            "resolution",
            "exception",
            "exceptions",
            "eligibility",
            "final-sale",
            "final",
            "sale",
            "available",
            "limitations",
            "shipping",
            "delivery",
        }

        extras: list[RetrievedChunk] = []

        for filename, results in by_document.items():

            if len(results) < 2:
                continue

            best_score = results[0].score

            for result in results[1:]:

                if id(result) in selected_ids:
                    continue

                heading_tokens = set(
                    _tokenize(
                        result.chunk.heading
                    )
                )

                heading_signal = bool(
                    heading_tokens
                    & complementary_keywords
                )

                # The chunk must be reasonably close to the document's
                # strongest chunk, or have a strong complementary heading.
                score_signal = (
                    result.score
                    >= best_score * 0.55
                )

                if (
                    score_signal
                    or heading_signal
                ):
                    extras.append(result)

        extras.sort(
            key=lambda result: result.score,
            reverse=True,
        )

        # --------------------------------------------------------------
        # Pass 3:
        # Add extras while keeping document balance.
        #
        # Do not allow one document to take more than half of the
        # retrieval budget when another relevant document exists.
        # --------------------------------------------------------------

        def document_count(
            filename: str,
        ) -> int:
            return sum(
                1
                for result in selected
                if result.chunk.filename
                == filename
            )

        distinct_documents = len(
            {
                result.chunk.filename
                for result in selected
            }
        )

        for result in extras:

            if len(selected) >= top_k:
                break

            if id(result) in selected_ids:
                continue

            filename = result.chunk.filename

            current_count = document_count(
                filename
            )

            # If there are multiple relevant documents, prevent one
            # document from occupying more than half the retrieval budget.
            if (
                distinct_documents > 1
                and current_count >= max(
                    1,
                    top_k // 2,
                )
            ):
                continue

            selected.append(result)
            selected_ids.add(id(result))

        # --------------------------------------------------------------
        # Pass 4:
        # Fill any remaining slots purely by score.
        # --------------------------------------------------------------

        if len(selected) < top_k:

            for result in candidates:

                if id(result) in selected_ids:
                    continue

                selected.append(result)
                selected_ids.add(id(result))

                if len(selected) >= top_k:
                    break

        selected.sort(
            key=lambda result: result.score,
            reverse=True,
        )

        return selected[:top_k]

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_chunks_by_filename(
        self,
        filename: str,
    ) -> list[Chunk]:

        return [
            chunk
            for chunk in self.chunks
            if chunk.filename == filename
        ]