"""Google Gemini adapter. ``pip install 'governed[gemini]'``.

A second, structurally different provider (distinct SDK, distinct message and
tool-calling shape from Anthropic/OpenAI) to prove the adapter contract holds
across vendors, not just within the Anthropic-shaped or OpenAI-shaped family.
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Any

from .base import LLMClient, LLMResponse, Message, ToolCall, ToolChoice, Usage

__all__ = ["GeminiClient"]

_CHOICE_MAP = {"auto": "AUTO", "required": "ANY", "none": "NONE"}

#: 429 is a `ClientError` (4xx) in google-genai's status-code split, but it
#: means "back off", not "bad request" -- worth retrying unlike other 4xxs.
_RETRYABLE_CLIENT_CODES = {429}

try:
    from google.genai.errors import ClientError, ServerError
except ImportError:  # google-genai not installed; only real API calls raise these,
    ServerError = ClientError = ()  # type: ignore[assignment,misc]  # so nothing to catch


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, ServerError):
        return True
    if isinstance(exc, ClientError):
        return getattr(exc, "code", None) in _RETRYABLE_CLIENT_CODES
    return False


class GeminiClient(LLMClient):
    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        client: Any = None,
        max_retries: int = 2,
        retry_backoff: float = 1.0,
    ) -> None:
        self.model = model
        #: Gemini's backend returns transient 503/429s under load that the SDK's
        #: own retry doesn't always absorb, and the agent loop above only retries
        #: contract violations, not transport errors -- so this client retries
        #: its own transient failures. `max_retries=0` disables it.
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        if client is not None:
            self._client = client
            return
        try:
            from google import genai
        except ImportError as exc:
            raise ImportError(
                "GeminiClient requires the `google-genai` package: "
                "pip install 'governed[gemini]'"
            ) from exc
        kwargs: dict[str, Any] = {"api_key": api_key or os.environ.get("GEMINI_API_KEY")}
        if base_url:
            kwargs["http_options"] = {"base_url": base_url}
        self._client = genai.Client(**kwargs)

    def complete(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: ToolChoice = "auto",
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> LLMResponse:
        config: dict[str, Any] = {
            "system_instruction": system,
            "max_output_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools:
            config["tools"] = _to_gemini_tools(tools)
            config["tool_config"] = {
                "function_calling_config": {"mode": _CHOICE_MAP.get(tool_choice, "AUTO")}
            }

        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client.models.generate_content(
                    model=self.model,
                    contents=_to_gemini_contents(messages),
                    config=config,
                )
                return _from_gemini_response(resp)
            except Exception as exc:
                if attempt >= self.max_retries or not _is_retryable(exc):
                    raise
                time.sleep(self.retry_backoff * (2**attempt))
        raise AssertionError("unreachable")  # loop always returns or raises


def _to_gemini_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "function_declarations": [
                {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": _clean_schema(t["input_schema"]),
                }
                for t in tools
            ]
        }
    ]


def _clean_schema(schema: Any) -> Any:
    """Strip ``additionalProperties`` from a pydantic-generated JSON Schema.

    A ``dict[str, X]`` field (e.g. ``DataAnalysisTool``'s ``agg``) renders as
    ``additionalProperties`` -- valid JSON Schema, and even a field
    ``google-genai``'s own ``Schema`` type declares -- but the Gemini REST API
    rejects it wherever it appears inside a nested ``anyOf`` branch. Drop it
    recursively rather than special-case the one tool.
    """
    if isinstance(schema, dict):
        return {k: _clean_schema(v) for k, v in schema.items() if k != "additionalProperties"}
    if isinstance(schema, list):
        return [_clean_schema(v) for v in schema]
    return schema


def _to_gemini_contents(messages: list[Message]) -> list[dict[str, Any]]:
    """Gemini has two roles (``user``/``model``) and no separate tool-result
    role: a function's output rides as a ``function_response`` part on a
    ``user`` turn, matched back to its call by id. Unlike Anthropic/OpenAI,
    Gemini's ``function_response`` also wants the function *name* -- which our
    provider-agnostic ``ToolResultBlock`` doesn't carry -- so it's recovered
    here from the ``function_call`` seen earlier in the same transcript.
    """
    out: list[dict[str, Any]] = []
    call_names: dict[str, str] = {}
    for m in messages:
        if m.role == "assistant":
            parts: list[dict[str, Any]] = []
            if m.text:
                parts.append({"text": m.text})
            for tc in m.tool_calls:
                call_names[tc.id] = tc.name
                part: dict[str, Any] = {
                    "function_call": {"id": tc.id, "name": tc.name, "args": tc.arguments}
                }
                signature = tc.meta.get("thought_signature")
                if signature is not None:
                    part["thought_signature"] = signature
                parts.append(part)
            out.append({"role": "model", "parts": parts or [{"text": ""}]})
        else:
            parts = []
            for tr in m.tool_results:
                response = {"error": tr.content} if tr.is_error else {"result": tr.content}
                parts.append(
                    {
                        "function_response": {
                            "id": tr.call_id,
                            "name": call_names.get(tr.call_id, tr.call_id),
                            "response": response,
                        }
                    }
                )
            if m.text:
                parts.append({"text": m.text})
            out.append({"role": "user", "parts": parts or [{"text": ""}]})
    return out


def _from_gemini_response(resp: Any) -> LLMResponse:
    text = ""
    tool_calls: list[ToolCall] = []
    candidate = resp.candidates[0]
    for part in candidate.content.parts:
        part_text = getattr(part, "text", None)
        if part_text:
            text += part_text
        fc = getattr(part, "function_call", None)
        if fc is not None:
            call_id = getattr(fc, "id", None) or f"{fc.name}-{uuid.uuid4().hex[:8]}"
            signature = getattr(part, "thought_signature", None)
            meta = {"thought_signature": signature} if signature is not None else {}
            tool_calls.append(
                ToolCall(id=call_id, name=fc.name, arguments=dict(fc.args or {}), meta=meta)
            )

    usage_meta = getattr(resp, "usage_metadata", None)
    return LLMResponse(
        text=text,
        tool_calls=tool_calls,
        usage=Usage(
            input_tokens=getattr(usage_meta, "prompt_token_count", 0) or 0,
            output_tokens=getattr(usage_meta, "candidates_token_count", 0) or 0,
        ),
        raw=resp,
    )
