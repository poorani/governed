"""OpenAI-compatible adapter. ``pip install 'governed[openai]'``.

Works with any endpoint that speaks the OpenAI Chat Completions API --
``base_url`` reaches vLLM, Ollama, Together, LM Studio, and friends.
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Any

from .base import LLMClient, LLMResponse, Message, ToolCall, ToolChoice, Usage

__all__ = ["OpenAIClient"]


class OpenAIClient(LLMClient):
    def __init__(
        self,
        model: str = "gpt-4.1",
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
            import openai
        except ImportError as exc:
            raise ImportError(
                "OpenAIClient requires the `openai` package: pip install 'governed[openai]'"
            ) from exc
        self._client = openai.OpenAI(
            api_key=api_key
            or os.environ.get("OPENAI_API_KEY", "not-needed-for-local-endpoints"),
            base_url=base_url,
        )

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
        kwargs: dict[str, Any] = dict(
            model=self.model,
            messages=[{"role": "system", "content": system}, *_to_openai_messages(messages)],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t["description"],
                        "parameters": t["input_schema"],
                    },
                }
                for t in tools
            ]
            kwargs["tool_choice"] = "required" if tool_choice == "required" else "auto"

        resp = self._client.chat.completions.create(**kwargs)
        return _from_openai_response(resp)


def _to_openai_messages(messages: list[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "assistant":
            entry: dict[str, Any] = {"role": "assistant", "content": m.text or None}
            if m.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                    }
                    for tc in m.tool_calls
                ]
            out.append(entry)
        else:
            # Tool results are their own `role="tool"` messages in this API,
            # sent before the accompanying user text (if any).
            for tr in m.tool_results:
                out.append({"role": "tool", "tool_call_id": tr.call_id, "content": tr.content})
            if m.text:
                out.append({"role": "user", "content": m.text})
    return out


def _from_openai_response(resp: Any) -> LLMResponse:
    choice = resp.choices[0].message
    tool_calls = [
        ToolCall(
            id=tc.id or uuid.uuid4().hex[:12],
            name=tc.function.name,
            arguments=json.loads(tc.function.arguments or "{}"),
        )
        for tc in (choice.tool_calls or [])
    ]
    usage = resp.usage
    return LLMResponse(
        text=choice.content or "",
        tool_calls=tool_calls,
        usage=Usage(
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        ),
        raw=resp,
    )
