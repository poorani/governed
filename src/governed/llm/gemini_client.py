"""Google Gemini adapter. ``pip install 'governed[gemini]'``.

A second, structurally different provider (distinct SDK, distinct message and
tool-calling shape from Anthropic/OpenAI) to prove the adapter contract holds
across vendors, not just within the Anthropic-shaped or OpenAI-shaped family.
"""

from __future__ import annotations

import os
import uuid
from typing import Any

from .base import LLMClient, LLMResponse, Message, ToolCall, ToolChoice, Usage

__all__ = ["GeminiClient"]

_CHOICE_MAP = {"auto": "AUTO", "required": "ANY", "none": "NONE"}


class GeminiClient(LLMClient):
    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        client: Any = None,
    ) -> None:
        self.model = model
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

        resp = self._client.models.generate_content(
            model=self.model,
            contents=_to_gemini_contents(messages),
            config=config,
        )
        return _from_gemini_response(resp)


def _to_gemini_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "function_declarations": [
                {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"],
                }
                for t in tools
            ]
        }
    ]


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
                parts.append(
                    {"function_call": {"id": tc.id, "name": tc.name, "args": tc.arguments}}
                )
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
            tool_calls.append(
                ToolCall(id=call_id, name=fc.name, arguments=dict(fc.args or {}))
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
