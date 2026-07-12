"""A deterministic, offline ``LLMClient`` for tests.

Replays a fixed list of ``LLMResponse``s in order. No network, no flake, and it
lets you assert on exactly what the loop sent -- ``client.calls[i]`` records the
system prompt, messages, and tool schemas for every completion.
"""

from __future__ import annotations

from typing import Any

from .base import LLMClient, LLMResponse, Message, ToolChoice

__all__ = ["ScriptedClient"]


class ScriptedClient(LLMClient):
    def __init__(self, responses: list[LLMResponse], model: str = "scripted") -> None:
        self.model = model
        self._responses = list(responses)
        self._index = 0
        #: One entry per call, in order, for assertions in tests.
        self.calls: list[dict[str, Any]] = []

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
        self.calls.append(
            {
                "system": system,
                "messages": list(messages),
                "tools": tools,
                "tool_names": [t["name"] for t in (tools or [])],
                "tool_choice": tool_choice,
            }
        )
        if self._index >= len(self._responses):
            raise IndexError(
                f"ScriptedClient exhausted after {self._index} calls. Add more "
                "LLMResponse entries, or check the agent is looping unexpectedly."
            )
        resp = self._responses[self._index]
        self._index += 1
        return resp
