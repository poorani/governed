"""The one interface every provider adapter implements.

The core imports no vendor SDK. Bring your own model by implementing
``LLMClient.complete`` -- one method, required to be side-effect free and safe
to retry.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

__all__ = [
    "LLMClient",
    "LLMResponse",
    "Message",
    "ToolCall",
    "ToolChoice",
    "ToolResultBlock",
    "Usage",
]

ToolChoice = Literal["auto", "required", "none"]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    #: Provider-specific round-trip data (e.g. Gemini's ``thought_signature``)
    #: that must be echoed back verbatim on a later turn. Opaque to the core.
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResultBlock:
    call_id: str
    content: str
    is_error: bool = False


@dataclass
class Message:
    role: Literal["user", "assistant", "system"]
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    #: A tool result rides on the *next* user turn, not a separate role -- this
    #: matches how every current provider's API is actually shaped.
    tool_results: list[ToolResultBlock] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    #: Populated when the provider reports prompt-cache activity. Not all do.
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    #: Raw provider payload, kept for debugging. Never read by the core loop.
    raw: Any = None


class LLMClient(ABC):
    """Implement ``complete``. Everything else has a usable default.

    Contract: ``complete`` must be side-effect free and safe to retry -- the
    agent loop retries a phase on contract violation, and a client that mutates
    external state on every call will misbehave under retry.
    """

    model: str

    @abstractmethod
    def complete(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: ToolChoice = "auto",
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> LLMResponse: ...

    def count_tokens(self, text: str) -> int:
        """Cheap estimate used only to decide *when* to compact.

        Never used for billing -- see ``memory.optimizer.CostLedger``, which
        prices from the provider's own reported ``Usage``. Being ~20% off here
        changes only the moment of the fold, not what anything costs.
        """
        return max(1, len(text) // 4)
