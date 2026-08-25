"""Agent orchestration.

One turn = one call to ``Agent.handle_message``:

1. Append the user's message to the session's clean text history.
2. Build a retrieval query from the last N user turns (so "what about
   Canada?" inherits "ship internationally" from the prior turn) and search
   the knowledge base.
3. Build the actual API request: system prompt + full text history, with the
   retrieved passages (wrapped as untrusted data, labeled with filename/
   heading/status/authority) inserted alongside the current user turn. The
   retrieved-context wrapper is NOT saved back into session history --
   only the plain user/assistant text is -- so history stays small and
   each turn re-retrieves fresh, relevant context.
4. Run the tool loop: if the model calls order_lookup, execute it against
   the sanitized OrderStore, wrap the result as untrusted data, and send it
   back, up to MAX_TOOL_ITERATIONS.
5. Parse the final assistant text for the SOURCES/HANDOFF trailer.
6. Save the assistant's visible text (trailer stripped) into session
   history, and return a full trace for logging/observability/eval.
"""
from __future__ import annotations

import copy
import time
import uuid
from dataclasses import dataclass, field

from app.config import (
    MAX_TOOL_ITERATIONS,
    RETRIEVAL_HISTORY_TURNS,
    ROOT_DIR,
    SYSTEM_PROMPT_VARIANT,
)
from app.llm_client import LLMClient, build_client
from app.orders import OrderStore
from app.response_format import ParsedResponse, parse_response
from app.retrieval import KnowledgeBase, RetrievedChunk
from app.security import scan_for_injection, wrap_untrusted_block
from app.tools_schema import ALL_TOOLS


_SYSTEM_PROMPT_PATHS = {
    "final": ROOT_DIR / "system_prompts" / "system_prompt_final.txt",
    "baseline": ROOT_DIR / "system_prompts" / "system_prompt_baseline.txt",
}


def _load_system_prompt(variant: str) -> str:
    path = _SYSTEM_PROMPT_PATHS.get(
        variant,
        _SYSTEM_PROMPT_PATHS["final"],
    )
    return path.read_text(encoding="utf-8")


@dataclass
class Session:
    session_id: str
    history: list[dict] = field(default_factory=list)
    # [{"role": "user"/"assistant", "content": str}]


@dataclass
class TurnTrace:
    session_id: str
    user_message: str
    retrieval_query: str
    retrieved: list[dict]
    tool_calls: list[dict]
    injection_flags: list[dict]
    raw_response: str
    parsed: ParsedResponse
    errors: list[str]
    latency_seconds: float


class Agent:
    def __init__(
        self,
        llm_client: LLMClient | None = None,
        knowledge_base: KnowledgeBase | None = None,
        order_store: OrderStore | None = None,
        system_prompt_variant: str = SYSTEM_PROMPT_VARIANT,
        use_mock: bool = False,
    ):
        self.llm = llm_client or build_client(
            use_mock=use_mock
        )
        self.kb = knowledge_base or KnowledgeBase()
        self.orders = order_store or OrderStore()
        self.system_prompt = _load_system_prompt(
            system_prompt_variant
        )
        self._sessions: dict[str, Session] = {}

    # -- session management -------------------------------------------------

    def new_session(self) -> str:
        session_id = str(uuid.uuid4())
        self._sessions[session_id] = Session(
            session_id=session_id
        )
        return session_id

    def _get_session(
        self,
        session_id: str,
    ) -> Session:
        if session_id not in self._sessions:
            self._sessions[session_id] = Session(
                session_id=session_id
            )
        return self._sessions[session_id]

    # -- retrieval ----------------------------------------------------------

    def _build_retrieval_query(
        self,
        session: Session,
        current_message: str,
    ) -> str:
        recent_user_turns = [
            m["content"]
            for m in session.history
            if m["role"] == "user"
        ]

        recent_user_turns = (
            recent_user_turns[
                -(RETRIEVAL_HISTORY_TURNS - 1):
            ]
            if RETRIEVAL_HISTORY_TURNS > 1
            else []
        )

        return " ".join(
            recent_user_turns + [current_message]
        )

    def _format_context_block(
        self,
        retrieved: list[RetrievedChunk],
    ) -> str:
        if not retrieved:
            return wrap_untrusted_block(
                "knowledge_base_search",
                "No matching knowledge-base passages "
                "were found for this query.",
            )

        parts = []

        for r in retrieved:
            c = r.chunk

            meta = (
                f'filename="{c.filename}" '
                f'heading="{c.heading}" '
                f'status="{c.status}" '
                f'policy_authority="{c.policy_authority}" '
                f'audience="{c.audience}" '
                f'score="{r.score:.2f}" '
                f'confidence="{r.confidence}"'
            )

            parts.append(
                f"<passage {meta}>\n"
                f"{c.text}\n"
                f"</passage>"
            )

        return wrap_untrusted_block(
            "knowledge_base_search",
            "\n".join(parts),
        )

    # -- main entry point --------------------------------------------------

    def handle_message(
        self,
        session_id: str,
        user_message: str,
        debug: bool = False,
    ) -> TurnTrace:

        start = time.time()

        session = self._get_session(
            session_id
        )

        errors: list[str] = []

        retrieval_query = (
            self._build_retrieval_query(
                session,
                user_message,
            )
        )

        retrieved = self.kb.search(
            retrieval_query
        )

        context_block = (
            self._format_context_block(
                retrieved
            )
        )

        injection_flags = []

        for r in retrieved:
            hits = scan_for_injection(
                r.chunk.text
            )

            if hits:
                injection_flags.append(
                    {
                        "filename": r.chunk.filename,
                        "heading": r.chunk.heading,
                        "patterns": hits,
                    }
                )

        api_messages = [
            {
                "role": m["role"],
                "content": m["content"],
            }
            for m in session.history
        ]

        api_messages.append(
            {
                "role": "user",
                "content": (
                    f"{context_block}\n\n"
                    f"<user_message>\n"
                    f"{user_message}\n"
                    f"</user_message>"
                ),
            }
        )

        tool_calls: list[dict] = []
        final_text = ""

        # --------------------------------------------------------------
        # Tool loop
        # --------------------------------------------------------------

        for _ in range(
            MAX_TOOL_ITERATIONS
        ):

            response = self.llm.create_message(
                system=self.system_prompt,
                messages=api_messages,
                tools=ALL_TOOLS,
            )

            tool_use_blocks = [
                b
                for b in response.content
                if b.type == "tool_use"
            ]

            text_blocks = [
                b.text
                for b in response.content
                if b.type == "text"
            ]

            if not tool_use_blocks:
                final_text = "\n".join(
                    text_blocks
                ).strip()
                break

            # ----------------------------------------------------------
            # IMPORTANT:
            # Preserve Gemini thought_signature when sending the
            # assistant function call back to the model.
            # ----------------------------------------------------------

            assistant_content = []

            for block in response.content:

                if block.type == "text":

                    assistant_content.append(
                        {
                            "type": "text",
                            "text": block.text,
                        }
                    )

                elif block.type == "tool_use":

                    tool_use_message = {
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    }

                    # Gemini 3 function calls may contain a
                    # thought_signature that MUST be preserved.
                    thought_signature = getattr(
                        block,
                        "thought_signature",
                        "",
                    )

                    if thought_signature:
                        tool_use_message[
                            "thought_signature"
                        ] = thought_signature

                    assistant_content.append(
                        tool_use_message
                    )

            api_messages.append(
                {
                    "role": "assistant",
                    "content": assistant_content,
                }
            )

            # ----------------------------------------------------------
            # Execute tools
            # ----------------------------------------------------------

            tool_result_blocks = []

            for block in tool_use_blocks:

                if block.name == "order_lookup":

                    raw_order_id = str(
                        block.input.get(
                            "order_id",
                            "",
                        )
                    )

                    result = self.orders.lookup(
                        raw_order_id
                    )

                    output = (
                        result.to_tool_output()
                    )

                    tool_calls.append(
                        {
                            "name": "order_lookup",
                            "arguments": {
                                "order_id": raw_order_id,
                            },
                            "normalized_order_id": (
                                result.order_id_normalized
                            ),
                            "found": result.found,
                            "sanitized_output": output,
                        }
                    )

                    hits = scan_for_injection(
                        str(output)
                    )

                    if hits:
                        injection_flags.append(
                            {
                                "filename": (
                                    "order_lookup_tool_result"
                                ),
                                "patterns": hits,
                            }
                        )

                    wrapped = wrap_untrusted_block(
                        (
                            "order_lookup:"
                            f"{result.order_id_normalized}"
                        ),
                        str(output),
                    )

                else:

                    output = {
                        "error": (
                            f"unknown tool "
                            f"{block.name}"
                        )
                    }

                    errors.append(
                        "model requested unknown tool: "
                        f"{block.name}"
                    )

                    wrapped = wrap_untrusted_block(
                        "unknown_tool",
                        str(output),
                    )

                tool_result_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": wrapped,
                    }
                )

            api_messages.append(
                {
                    "role": "user",
                    "content": tool_result_blocks,
                }
            )

        else:

            errors.append(
                "max tool iterations reached "
                "without a final text response"
            )

            final_text = (
                "I'm having trouble completing "
                "that request right now. "
                "Let me connect you with human "
                "support.\n\n"
                "--\n"
                "SOURCES: none\n"
                "HANDOFF: true\n"
                "--"
            )

        # --------------------------------------------------------------
        # Parse final response
        # --------------------------------------------------------------

        parsed = parse_response(
            final_text
        )

        if not parsed.trailer_found:
            errors.append(
                "model did not emit a parseable "
                "SOURCES/HANDOFF trailer"
            )

        # --------------------------------------------------------------
        # Persist only clean text turns
        # --------------------------------------------------------------

        session.history.append(
            {
                "role": "user",
                "content": user_message,
            }
        )

        session.history.append(
            {
                "role": "assistant",
                "content": parsed.display_text,
            }
        )

        # --------------------------------------------------------------
        # Build trace
        # --------------------------------------------------------------

        trace = TurnTrace(
            session_id=session_id,
            user_message=user_message,
            retrieval_query=retrieval_query,
            retrieved=[
                {
                    "filename": r.chunk.filename,
                    "heading": r.chunk.heading,
                    "status": r.chunk.status,
                    "policy_authority": (
                        r.chunk.policy_authority
                    ),
                    "score": round(
                        r.score,
                        3,
                    ),
                }
                for r in retrieved
            ],
            tool_calls=tool_calls,
            injection_flags=injection_flags,
            raw_response=final_text,
            parsed=parsed,
            errors=errors,
            latency_seconds=round(
                time.time() - start,
                3,
            ),
        )

        if debug:
            _print_trace(trace)

        return trace


def _print_trace(
    trace: TurnTrace,
) -> None:

    import json
    import sys

    payload = {
        "session_id": trace.session_id,
        "user_message": trace.user_message,
        "retrieval_query": trace.retrieval_query,
        "retrieved": trace.retrieved,
        "tool_calls": trace.tool_calls,
        "injection_flags": trace.injection_flags,
        "sources": trace.parsed.sources,
        "handoff": trace.parsed.handoff,
        "errors": trace.errors,
        "latency_seconds": trace.latency_seconds,
    }

    print(
        "--- TRACE ---",
        file=sys.stderr,
    )

    print(
        json.dumps(
            payload,
            indent=2,
        ),
        file=sys.stderr,
    )

    print(
        "-------------",
        file=sys.stderr,
    )