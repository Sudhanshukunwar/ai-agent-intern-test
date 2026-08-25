"""LLM client wrapper.

Two implementations:

- ``GeminiClient``: calls the Google Gemini generateContent REST API
  directly over ``urllib.request``. No Gemini SDK is required.
- ``MockClient``: a deterministic offline client used for smoke tests.

The Gemini adapter preserves Gemini function-call thought signatures so
tool calls continue to work correctly with newer Gemini models.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any

from app.config import GEMINI_API_KEY, MODEL_NAME


_API_URL = (
    "https://generativelanguage.googleapis.com/"
    "v1beta/models/{model}:generateContent"
)


class LLMClient(ABC):
    @abstractmethod
    def create_message(
        self,
        *,
        system: str,
        messages: list[dict],
        tools: list[dict],
    ) -> Any:
        ...


class _Block:
    """Shape-compatible content block used by Agent."""

    def __init__(self, raw: dict):
        self.type = raw.get("type", "")
        self.text = raw.get("text", "")
        self.id = raw.get("id", "")
        self.name = raw.get("name", "")
        self.input = raw.get("input", {}) or {}

        # Gemini 3+ may return a thoughtSignature with function calls.
        # It must be preserved and sent back with the function call.
        self.thought_signature = raw.get("thought_signature", "")


class _Response:
    """Shape-compatible response used by Agent."""

    def __init__(self, raw: dict):
        self.content = [
            _Block(block)
            for block in raw.get("content", [])
        ]

        self.stop_reason = raw.get(
            "stop_reason",
            "",
        )

        self._raw = raw


class GeminiClient(LLMClient):
    """Gemini implementation using the REST generateContent API."""

    def __init__(
        self,
        api_key: str = GEMINI_API_KEY,
        model: str = MODEL_NAME,
    ):
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. "
                "Copy .env.example to .env and add your Gemini API key, "
                "or pass --mock to use the offline mock client."
            )

        self._api_key = api_key
        self._model = model

    # ------------------------------------------------------------------
    # Tool conversion
    # ------------------------------------------------------------------

    def _convert_tools(
        self,
        tools: list[dict],
    ) -> list[dict]:

        if not tools:
            return []

        function_declarations = []

        for tool in tools:
            function_declarations.append(
                {
                    "name": tool["name"],
                    "description": tool.get(
                        "description",
                        "",
                    ),
                    "parameters": tool.get(
                        "input_schema",
                        {},
                    ),
                }
            )

        return [
            {
                "functionDeclarations": function_declarations
            }
        ]

    # ------------------------------------------------------------------
    # Conversation conversion
    # ------------------------------------------------------------------

    def _convert_messages(
        self,
        messages: list[dict],
    ) -> list[dict]:

        contents = []

        for message in messages:
            role = message.get("role")
            content = message.get("content", "")

            # ----------------------------------------------------------
            # Normal text message
            # ----------------------------------------------------------

            if isinstance(content, str):
                contents.append(
                    {
                        "role": (
                            "model"
                            if role == "assistant"
                            else "user"
                        ),
                        "parts": [
                            {
                                "text": content
                            }
                        ],
                    }
                )

                continue

            # ----------------------------------------------------------
            # Structured blocks
            # ----------------------------------------------------------

            parts = []

            for block in content:
                block_type = block.get("type")

                # ------------------------------------------------------
                # Assistant text
                # ------------------------------------------------------

                if block_type == "text":
                    text = block.get(
                        "text",
                        "",
                    )

                    if text:
                        parts.append(
                            {
                                "text": text
                            }
                        )

                # ------------------------------------------------------
                # Assistant function call
                # ------------------------------------------------------

                elif block_type == "tool_use":

                    function_call_part = {
                        "functionCall": {
                            "name": block.get(
                                "name",
                                "",
                            ),
                            "args": block.get(
                                "input",
                                {},
                            ),
                            "id": block.get(
                                "id",
                                "",
                            ),
                        }
                    }

                    # IMPORTANT:
                    # Gemini 3+ requires the thoughtSignature returned
                    # with the original function call to be preserved
                    # when the conversation is sent back to Gemini.
                    thought_signature = block.get(
                        "thought_signature"
                    )

                    if thought_signature:
                        function_call_part[
                            "thoughtSignature"
                        ] = thought_signature

                    parts.append(function_call_part)

                # ------------------------------------------------------
                # Function result
                # ------------------------------------------------------

                elif block_type == "tool_result":
                    tool_use_id = block.get(
                        "tool_use_id",
                        "",
                    )

                    tool_name = self._tool_name_from_id(
                        tool_use_id
                    )

                    tool_content = block.get(
                        "content",
                        "",
                    )

                    if not isinstance(
                        tool_content,
                        str,
                    ):
                        tool_content = json.dumps(
                            tool_content
                        )

                    parts.append(
                        {
                            "functionResponse": {
                                "name": tool_name,
                                "id": tool_use_id,
                                "response": {
                                    "result": tool_content
                                },
                            }
                        }
                    )

            if parts:

                # Tool results are supplied as user messages,
                # matching Gemini's function-calling conversation format.
                if any(
                    block.get("type") == "tool_result"
                    for block in content
                ):
                    gemini_role = "user"

                else:
                    gemini_role = (
                        "model"
                        if role == "assistant"
                        else "user"
                    )

                contents.append(
                    {
                        "role": gemini_role,
                        "parts": parts,
                    }
                )

        return contents

    @staticmethod
    def _tool_name_from_id(
        tool_use_id: str,
    ) -> str:

        # Our Gemini responses use IDs such as:
        #
        # gemini_order_lookup
        #
        # The current project has one tool, but this also
        # handles the general "gemini_<tool-name>" format.

        if tool_use_id.startswith("gemini_"):
            return tool_use_id[len("gemini_"):]

        # Fallback for compatibility with the current assignment.
        return "order_lookup"

    # ------------------------------------------------------------------
    # API call
    # ------------------------------------------------------------------

    def create_message(
        self,
        *,
        system: str,
        messages: list[dict],
        tools: list[dict],
    ) -> _Response:

        body: dict = {
            "systemInstruction": {
                "parts": [
                    {
                        "text": system
                    }
                ]
            },
            "contents": self._convert_messages(
                messages
            ),
            "generationConfig": {
                "temperature": 0,
                "maxOutputTokens": 1024,
            },
        }

        converted_tools = self._convert_tools(
            tools
        )

        if converted_tools:
            body["tools"] = converted_tools

            body["toolConfig"] = {
                "functionCallingConfig": {
                    "mode": "AUTO"
                }
            }

        url = _API_URL.format(
            model=self._model
        )

        request = urllib.request.Request(
            url,
            data=json.dumps(
                body
            ).encode("utf-8"),
            method="POST",
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self._api_key,
            },
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=60,
            ) as response:

                raw = json.loads(
                    response.read().decode(
                        "utf-8"
                    )
                )

        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(
                "utf-8",
                errors="replace",
            )

            raise RuntimeError(
                f"Gemini API error "
                f"{exc.code}: {detail}"
            ) from exc

        except urllib.error.URLError as exc:
            raise RuntimeError(
                "Could not reach the Gemini API: "
                f"{exc.reason}"
            ) from exc

        return self._parse_response(raw)

    # ------------------------------------------------------------------
    # Gemini response conversion
    # ------------------------------------------------------------------

    def _parse_response(
        self,
        raw: dict,
    ) -> _Response:

        candidates = raw.get(
            "candidates",
            [],
        )

        if not candidates:
            return _Response(
                {
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "I couldn't generate "
                                "a response."
                            ),
                        }
                    ],
                    "stop_reason": "error",
                }
            )

        candidate = candidates[0]

        response_content = candidate.get(
            "content",
            {},
        )

        parts = response_content.get(
            "parts",
            [],
        )

        content = []

        for part in parts:

            # ----------------------------------------------------------
            # Normal text response
            # ----------------------------------------------------------

            if "text" in part:
                content.append(
                    {
                        "type": "text",
                        "text": part["text"],
                    }
                )

            # ----------------------------------------------------------
            # Function call
            # ----------------------------------------------------------

            elif "functionCall" in part:

                function_call = part[
                    "functionCall"
                ]

                function_name = function_call.get(
                    "name",
                    "",
                )

                arguments = function_call.get(
                    "args",
                    {},
                )

                # Gemini function-call IDs are supported by
                # newer models. If absent, create a stable ID
                # for this application's adapter.
                call_id = function_call.get(
                    "id"
                )

                if not call_id:
                    call_id = (
                        f"gemini_{function_name}"
                    )

                # IMPORTANT:
                # Preserve Gemini's thoughtSignature.
                #
                # Gemini 3 models may require this signature
                # to be returned together with the functionCall
                # on the next request.
                thought_signature = part.get(
                    "thoughtSignature",
                    ""
                )

                content.append(
                    {
                        "type": "tool_use",
                        "id": call_id,
                        "name": function_name,
                        "input": arguments,
                        "thought_signature": thought_signature,
                    }
                )

        has_tool_call = any(
            block.get("type") == "tool_use"
            for block in content
        )

        return _Response(
            {
                "content": content,
                "stop_reason": (
                    "tool_use"
                    if has_tool_call
                    else "end_turn"
                ),
                "_gemini_raw": raw,
            }
        )


class MockClient(LLMClient):
    """Deterministic offline client.

    Used only for --mock / MOCK_MODE.
    """

    _ORDER_ID_RE = re.compile(
        r"\bORD-?[\s_-]?\d{3,}\b",
        re.IGNORECASE,
    )

    def create_message(
        self,
        *,
        system: str,
        messages: list[dict],
        tools: list[dict],
    ) -> _Response:

        last_user_text = ""

        for message in reversed(messages):

            if (
                message.get("role") == "user"
                and isinstance(
                    message.get("content"),
                    str,
                )
            ):
                last_user_text = message[
                    "content"
                ]

                break

        # If the previous turn already received
        # a tool result, finish the mock response.

        if (
            messages
            and messages[-1].get("role") == "user"
            and isinstance(
                messages[-1].get("content"),
                list,
            )
        ):

            for block in messages[-1]["content"]:

                if block.get("type") == "tool_result":

                    return _Response(
                        {
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        "Here is what I found."
                                        "\n\n--\n"
                                        "SOURCES: none\n"
                                        "HANDOFF: false\n"
                                        "--"
                                    ),
                                }
                            ],
                            "stop_reason": "end_turn",
                        }
                    )

        # Detect an order ID and request order_lookup.

        order_match = self._ORDER_ID_RE.search(
            last_user_text
        )

        if order_match and tools:

            return _Response(
                {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "mock_tool_1",
                            "name": "order_lookup",
                            "input": {
                                "order_id": (
                                    order_match.group(0)
                                )
                            },
                            "thought_signature": "",
                        }
                    ],
                    "stop_reason": "tool_use",
                }
            )

        source_match = re.search(
            r'filename="([^"]+)"',
            system + str(messages),
        )

        source = (
            source_match.group(1)
            if source_match
            else "none"
        )

        return _Response(
            {
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "(mock) Based on the "
                            "retrieved policy content."
                            "\n\n--\n"
                            f"SOURCES: {source}\n"
                            "HANDOFF: false\n"
                            "--"
                        ),
                    }
                ],
                "stop_reason": "end_turn",
            }
        )


def build_client(
    use_mock: bool = False,
) -> LLMClient:

    if use_mock:
        return MockClient()

    return GeminiClient()