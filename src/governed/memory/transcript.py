"""Context compaction: fold old transcript into a rolling summary before the
context window fills.

``Compactor`` is the naive, one-completion version -- fine until the discarded
prefix is itself larger than the model's window, at which point summarising it
in a single call either errors or silently drops the front of the history (the
part with the schema in it). ``memory.optimizer.RecursiveCompactor`` subclasses
this with chunked, recursive folding for that case; ``AgentConfig`` uses it by
default.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ..llm.base import LLMClient, Message

__all__ = ["SUMMARIZE_PROMPT", "CompactionConfig", "Compactor"]


@dataclass
class CompactionConfig:
    #: Fraction of the context window that triggers a fold.
    trigger_ratio: float = 0.7
    context_window_tokens: int = 180_000
    #: Most recent iterations kept verbatim; everything older is folded.
    keep_iterations: int = 3
    max_summary_words: int = 600

    @property
    def trigger_tokens(self) -> int:
        return int(self.context_window_tokens * self.trigger_ratio)


SUMMARIZE_PROMPT = """\
You are compressing part of an autonomous agent's transcript so it fits a \
smaller context window. Preserve, in this order of priority:

1. Facts the agent will need again: paths, schemas, column names, IDs, values.
2. Approaches already tried and their outcomes, especially failures.
3. Decisions made and the reasoning behind them.
4. Unfinished work and open blockers.

Discard restated instructions, boilerplate, and reasoning that led nowhere. \
Dense prose or bullets, no preamble, under {max_words} words.

PRIOR SUMMARY (carry forward anything still relevant):
{prior}

TRANSCRIPT TO COMPRESS:
{history}
"""


class Compactor:
    def __init__(
        self,
        llm: LLMClient,
        config: CompactionConfig | None = None,
        on_compact: Callable[[int, int], object] | None = None,
    ) -> None:
        self.llm = llm
        self.config = config or CompactionConfig()
        self.on_compact: Callable[[int, int], object] = on_compact or (
            lambda before, after: None
        )

    def should_compact(self, messages: list[Message], system: str) -> bool:
        estimate = self.estimate_tokens(messages) + self.llm.count_tokens(system)
        return estimate >= self.config.trigger_tokens

    def estimate_tokens(self, messages: list[Message]) -> int:
        return sum(self.llm.count_tokens(self._render(m)) for m in messages)

    def compact(
        self,
        messages: list[Message],
        prior_summary: str = "",
        current_iteration: int = 0,
    ) -> tuple[list[Message], str]:
        cutoff = current_iteration - self.config.keep_iterations
        split = self._safe_split(messages, cutoff)
        if split <= 1:
            return messages, prior_summary

        old, kept = messages[:split], messages[split:]
        before = self.estimate_tokens(messages)

        rendered = "\n\n".join(self._render(m) for m in old)
        summary = self._summarize(rendered, prior_summary)

        after = self.estimate_tokens(kept) + self.llm.count_tokens(summary)
        self.on_compact(before, after)
        return kept, summary

    # -- machinery ----------------------------------------------------------

    def _safe_split(self, messages: list[Message], cutoff: int) -> int:
        """The last index at which cutting cannot orphan a tool_use from its
        tool_result -- iteration numbers are monotonic and a call/result pair
        always shares one, so cutting on an iteration boundary is always safe.
        """
        split = 0
        for i, m in enumerate(messages):
            if m.meta.get("iteration", 0) < cutoff:
                split = i + 1
            else:
                break
        return split

    def _render(self, m: Message) -> str:
        parts = [f"[{m.role}]"]
        if m.text:
            parts.append(m.text)
        for tc in m.tool_calls:
            parts.append(f"  called {tc.name}({tc.arguments})")
        for tr in m.tool_results:
            status = "error" if tr.is_error else "ok"
            parts.append(f"  -> [{status}] {tr.content}")
        return "\n".join(parts)

    def _summarize(self, history: str, prior_summary: str) -> str:
        prompt = SUMMARIZE_PROMPT.format(
            max_words=self.config.max_summary_words,
            prior=prior_summary or "(none)",
            history=history,
        )
        resp = self.llm.complete(
            system="You compress agent transcripts losslessly with respect to facts.",
            messages=[Message(role="user", text=prompt)],
            tools=None,
            tool_choice="none",
            max_tokens=2048,
            temperature=0.0,
        )
        return resp.text.strip()
