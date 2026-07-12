"""Anthropic adapter. ``pip install 'governed[anthropic]'``."""

from __future__ import annotations

import os
from typing import Any

from .base import LLMClient, LLMResponse, Message, ToolCall, ToolChoice, Usage

__all__ = ["AnthropicClient"]

_CHOICE_MAP = {"auto": {"type": "auto"}, "required": {"type": "any"}}


class AnthropicClient(LLMClient):
    def __init__(
        self,
        model: str = "claude-sonnet-5",
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        client: Any = None,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.model = model
        self.extra_headers = extra_headers or {}
        if client is not None:
            self._client = client
            return
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "AnthropicClient requires the `anthropic` package: "
                "pip install 'governed[anthropic]'"
            ) from exc
        self._client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"), base_url=base_url
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
            system=system,
            messages=_to_anthropic_messages(messages),
            max_tokens=max_tokens,
            temperature=temperature,
            extra_headers=self.extra_headers,
        )
        if tools:
            kwargs["tools"] = [
                {
                    "name": t["name"],
                    "description": t["description"],
                    "input_schema": t["input_schema"],
                }
                for t in tools
            ]
            kwargs["tool_choice"] = _CHOICE_MAP.get(tool_choice, {"type": "auto"})

        resp = self._client.messages.create(**kwargs)
        return _from_anthropic_response(resp)


def _to_anthropic_messages(messages: list[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "assistant":
            content: list[dict[str, Any]] = []
            if m.text:
                content.append({"type": "text", "text": m.text})
            for tc in m.tool_calls:
                content.append(
                    {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments}
                )
            out.append(
                {"role": "assistant", "content": content or [{"type": "text", "text": ""}]}
            )
        else:
            content = []
            for tr in m.tool_results:
                content.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tr.call_id,
                        "content": tr.content,
                        "is_error": tr.is_error,
                    }
                )
            if m.text:
                content.append({"type": "text", "text": m.text})
            out.append({"role": "user", "content": content or [{"type": "text", "text": ""}]})
    return out


def _from_anthropic_response(resp: Any) -> LLMResponse:
    text = ""
    tool_calls: list[ToolCall] = []
    for block in resp.content:
        if block.type == "text":
            text += block.text
        elif block.type == "tool_use":
            tool_calls.append(
                ToolCall(id=block.id, name=block.name, arguments=dict(block.input))
            )

    usage = resp.usage
    return LLMResponse(
        text=text,
        tool_calls=tool_calls,
        usage=Usage(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
        ),
        raw=resp,
    )
