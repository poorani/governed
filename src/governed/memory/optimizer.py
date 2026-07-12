"""Money, context, and knowing when to stop.

Three mechanisms, one concern: an autonomous agent left alone with a credit card
and a ``while`` loop is a liability.

1. ``CostLedger`` prices every completion the moment it returns, from the
   provider's own reported token counts, at the rate card for the model that
   served it.
2. ``RecursiveCompactor`` folds old transcript into summaries before the context
   window fills, so a long run degrades instead of dying.
3. ``CircuitBreaker`` terminates the run -- safely, with state checkpointed and a
   structured answer -- when spend crosses a ceiling or the agent starts going
   in circles.

Honesty about cost accounting
-----------------------------

Costs are computed from ``LLMResponse.usage``, which comes from the provider's
response body. That figure is authoritative and matches the invoice. Costs are
*not* computed from ``LLMClient.count_tokens``, whose default implementation is
``len(text) // 4`` and is wrong by tens of percent. The heuristic is used for
one thing only: deciding *when* to compact, where being 20% off changes nothing
but the moment of the fold.

Two caveats worth knowing before you set ``max_usd`` and walk away:

* The breaker checks spend *after* each completion returns. A single call whose
  output blows the remaining budget will still be paid for. Set the ceiling with
  one call of headroom, or bound ``max_tokens_per_call``.
* If ``resolve_pricing`` does not recognise a model string, its cost is counted
  as zero and a warning is emitted once. A silent zero is the failure mode that
  matters here, so it is not silent -- but a self-hosted model behind
  ``OpenAIClient(base_url=...)`` genuinely costs nothing per token, and pretending
  otherwise would be worse. Pass ``CostConfig(pricing_overrides=...)`` when the
  model is real and unrecognised.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from ..llm.base import LLMClient, Message, Usage
from .transcript import CompactionConfig, Compactor

__all__ = [
    "PRICING",
    "PRICING_AS_OF",
    "CallCost",
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitOpen",
    "CostConfig",
    "CostLedger",
    "ModelPricing",
    "RecursiveCompactor",
    "compaction_for",
    "resolve_pricing",
]


# ---------------------------------------------------------------------------
# Rate card
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelPricing:
    """USD per million tokens, plus the multipliers for cached input."""

    input_per_mtok: float
    output_per_mtok: float
    context_window: int = 200_000
    #: Cache reads are billed at a fraction of the base input rate.
    cache_read_mult: float = 0.10
    #: 5-minute cache writes cost more than fresh input.
    cache_write_mult: float = 1.25
    #: Batch API halves both sides.
    batch_mult: float = 0.50

    def cost(
        self,
        input_tokens: int,
        output_tokens: int,
        *,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        batch: bool = False,
    ) -> float:
        m = self.batch_mult if batch else 1.0
        usd = (
            input_tokens * self.input_per_mtok
            + output_tokens * self.output_per_mtok
            + cache_read_tokens * self.input_per_mtok * self.cache_read_mult
            + cache_write_tokens * self.input_per_mtok * self.cache_write_mult
        ) / 1_000_000
        return usd * m


#: When these numbers were last checked against the vendor's pricing page.
#: Rate cards move. ``CostConfig.pricing_overrides`` exists so you never have to
#: wait for this library to catch up, and you should treat this table as a
#: convenience default rather than a source of truth for finance.
PRICING_AS_OF = "2026-07-08"

PRICING: dict[str, ModelPricing] = {
    # -- Anthropic, current generation -----------------------------------
    "claude-fable-5": ModelPricing(10.00, 50.00, context_window=1_000_000),
    "claude-mythos-5": ModelPricing(10.00, 50.00, context_window=1_000_000),
    "claude-opus-4-8": ModelPricing(5.00, 25.00, context_window=1_000_000),
    "claude-opus-4-7": ModelPricing(5.00, 25.00, context_window=1_000_000),
    "claude-opus-4-6": ModelPricing(5.00, 25.00, context_window=1_000_000),
    "claude-opus-4-5": ModelPricing(5.00, 25.00, context_window=200_000),
    # Sonnet 5 is on introductory pricing ($2/$10) through 2026-08-31, after
    # which it reverts to $3/$15. The table carries the post-introductory rate
    # so a budget set today does not silently overrun in September.
    "claude-sonnet-5": ModelPricing(3.00, 15.00, context_window=1_000_000),
    "claude-sonnet-4-6": ModelPricing(3.00, 15.00, context_window=1_000_000),
    "claude-sonnet-4-5": ModelPricing(3.00, 15.00, context_window=200_000),
    "claude-haiku-4-5": ModelPricing(1.00, 5.00, context_window=200_000),
    # -- Anthropic, legacy. The models named in the original spec. --------
    "claude-3-7-sonnet": ModelPricing(3.00, 15.00, context_window=200_000),
    "claude-3-5-sonnet": ModelPricing(3.00, 15.00, context_window=200_000),
    "claude-opus-4-1": ModelPricing(15.00, 75.00, context_window=200_000),
    # -- OpenAI ----------------------------------------------------------
    "gpt-4.1": ModelPricing(2.00, 8.00, context_window=1_000_000),
    "gpt-4.1-mini": ModelPricing(0.40, 1.60, context_window=1_000_000),
    "gpt-4.1-nano": ModelPricing(0.10, 0.40, context_window=1_000_000),
    "gpt-4o": ModelPricing(2.50, 10.00, context_window=128_000),
    "gpt-4o-mini": ModelPricing(0.15, 0.60, context_window=128_000),
    # -- Local / self-hosted: real models, genuinely free per token. ------
    "scripted": ModelPricing(0.0, 0.0),
    "local": ModelPricing(0.0, 0.0),
}


def resolve_pricing(
    model: str, overrides: dict[str, ModelPricing] | None = None
) -> ModelPricing | None:
    """Longest-prefix match, so ``claude-sonnet-4-6-20260219`` finds its rate.

    Returns ``None`` for an unrecognised model. The caller decides what an
    unknown price means; this function does not guess.
    """
    table = {**PRICING, **(overrides or {})}
    if model in table:
        return table[model]
    matches = [k for k in table if model.startswith(k)]
    if matches:
        return table[max(matches, key=len)]
    return None


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------


@dataclass
class CallCost:
    model: str
    phase: str
    iteration: int
    input_tokens: int
    output_tokens: int
    usd: float
    priced: bool = True
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "phase": self.phase,
            "iteration": self.iteration,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "usd": round(self.usd, 6),
            "priced": self.priced,
            "ts": self.ts,
        }


@dataclass
class CostConfig:
    enabled: bool = True
    #: Rate cards for models this library does not know, or newer than it does.
    pricing_overrides: dict[str, ModelPricing] = field(default_factory=dict)
    #: Requests are sent through the Batch API. Halves every rate.
    batch: bool = False


class CostLedger:
    """Every completion, priced, attributed to the phase that made it.

    Attribution matters more than the total. A run that spends 60% of its money
    in ANALYZE is over-planning; 60% in OBSERVE means tool output is not being
    truncated hard enough; a large ``compaction`` line means the transcript is
    being folded too often and ``keep_iterations`` should come down. The total
    tells you that you spent $4.12. ``by_phase()`` tells you why.
    """

    def __init__(self, config: CostConfig | None = None) -> None:
        self.config = config or CostConfig()
        self.calls: list[CallCost] = []
        self.total_usd: float = 0.0
        self._warned: set[str] = set()
        #: Set by ``Agent`` so an unpriced model is reported once, loudly.
        self.on_unpriced: Callable[[str], object] = lambda model: None

    def seed(self, usd: float) -> None:
        """Restore spend from a checkpoint so ``resume`` cannot reset the meter."""
        self.total_usd = max(self.total_usd, float(usd))

    def record(
        self,
        model: str,
        usage: Usage,
        *,
        phase: str = "",
        iteration: int = 0,
    ) -> CallCost:
        if not self.config.enabled:
            return CallCost(model, phase, iteration, 0, 0, 0.0, priced=False)

        pricing = resolve_pricing(model, self.config.pricing_overrides)
        if pricing is None:
            if model not in self._warned:
                self._warned.add(model)
                self.on_unpriced(model)
            usd, priced = 0.0, False
        else:
            usd = pricing.cost(
                usage.input_tokens, usage.output_tokens, batch=self.config.batch
            )
            priced = True

        cost = CallCost(
            model, phase, iteration, usage.input_tokens, usage.output_tokens, usd, priced
        )
        self.calls.append(cost)
        self.total_usd += usd
        return cost

    # -- reporting --------------------------------------------------------

    def by_phase(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for c in self.calls:
            out[c.phase or "unknown"] = out.get(c.phase or "unknown", 0.0) + c.usd
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    def by_model(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for c in self.calls:
            out[c.model] = out.get(c.model, 0.0) + c.usd
        return out

    @property
    def unpriced_models(self) -> set[str]:
        return set(self._warned)

    def summary(self) -> str:
        lines = [f"Total: ${self.total_usd:.4f} across {len(self.calls)} completions"]
        for phase, usd in self.by_phase().items():
            share = (usd / self.total_usd * 100) if self.total_usd else 0.0
            lines.append(f"  {phase:<12} ${usd:>8.4f}  {share:>5.1f}%")
        if self._warned:
            lines.append(f"  (unpriced, counted as $0: {', '.join(sorted(self._warned))})")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_usd": round(self.total_usd, 6),
            "calls": [c.to_dict() for c in self.calls],
            "by_phase": {k: round(v, 6) for k, v in self.by_phase().items()},
            "unpriced_models": sorted(self._warned),
        }


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------


class CircuitOpen(Exception):
    """The run must stop now. Caught by ``Agent``, which checkpoints and submits.

    ``terminal_status`` distinguishes *ran out of resources* (``exhausted``) from
    *stopped making progress* (``blocked``). The difference matters to whoever
    reads the result: the first says try again with a bigger budget, the second
    says the approach is wrong.
    """

    def __init__(
        self,
        reason: str,
        detail: str = "",
        terminal_status: Literal["exhausted", "blocked"] = "exhausted",
    ) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail
        self.terminal_status = terminal_status


@dataclass
class CircuitBreakerConfig:
    enabled: bool = True

    #: Hard dollar ceiling for the whole task, checked after every completion.
    #: ``None`` disables the cost trip. Set it. This is the one that matters.
    max_usd: float | None = None
    #: Emit a warning once spend crosses this fraction of ``max_usd``.
    warn_at_ratio: float = 0.75

    #: The same tool called with byte-identical arguments this many times in a
    #: run. Legitimate repetition exists (polling, retries after a fix), so the
    #: default is loose enough not to fire on a healthy run.
    max_identical_tool_calls: int = 4
    #: Consecutive iterations, *after the first*, in which the plan repeats the
    #: previous iteration's (step_id, tool) and no plan step is newly completed.
    #: Four means the agent gets five swings at the same step before the breaker
    #: decides it is not swinging at anything.
    max_stalled_iterations: int = 4
    #: Iterations in a row whose evaluation was `failure`. Complements
    #: ``Budget.max_consecutive_failures``, which the agent already enforces.
    max_consecutive_failures: int | None = None


class CircuitBreaker:
    """Trips on money, on repetition, and on stalling.

    The three detectors answer different questions.

    *Money* is a hard ceiling on ``CostLedger.total_usd``. Nothing subtle.

    *Repetition* fingerprints ``(tool, sorted(arguments))``. An agent calling
    ``file_system(read, config.yaml)`` for the fourth time is not learning
    anything new from it.

    *Stalling* watches the plan rather than the tools, because the interesting
    failure is an agent that varies its arguments while making no progress: it
    reads a slightly different file each iteration, forever. A stall is an
    iteration whose plan targets the same ``(step_id, tool)`` as the last one
    *and* whose evaluation completed no new steps. Either alone is fine --
    retrying a step after fixing something is normal, and a plan can legitimately
    revisit a step. Both together, five times running, is a loop.

    None of this catches an agent that is genuinely working but slowly. That is
    what ``max_usd`` and ``Budget.max_wall_seconds`` are for.
    """

    def __init__(
        self, config: CircuitBreakerConfig | None = None, ledger: CostLedger | None = None
    ):
        self.config = config or CircuitBreakerConfig()
        self.ledger = ledger
        self._call_counts: dict[tuple[str, str], int] = {}
        self._last_action: tuple[str, str] | None = None
        self._completed_steps: set[str] = set()
        self._stalled = 0
        self._failures = 0
        self._warned = False
        #: Set by ``Agent``. ``(reason, detail) -> None``.
        self.on_warn: Callable[[str, str], object] = lambda reason, detail: None

    # -- cost -------------------------------------------------------------

    def check_cost(self) -> None:
        c = self.config
        if not c.enabled or c.max_usd is None or self.ledger is None:
            return
        spent = self.ledger.total_usd
        if spent >= c.max_usd:
            raise CircuitOpen(
                "cost_ceiling",
                f"spent ${spent:.4f} of a ${c.max_usd:.2f} ceiling",
                terminal_status="exhausted",
            )
        if not self._warned and spent >= c.max_usd * c.warn_at_ratio:
            self._warned = True
            self.on_warn(
                "cost_warning", f"${spent:.4f} of ${c.max_usd:.2f} ({spent / c.max_usd:.0%})"
            )

    # -- repetition -------------------------------------------------------

    def observe_tool_call(self, tool: str, arguments: dict[str, Any]) -> None:
        c = self.config
        if not c.enabled:
            return
        key = (tool, _fingerprint(arguments))
        n = self._call_counts[key] = self._call_counts.get(key, 0) + 1
        if n >= c.max_identical_tool_calls:
            raise CircuitOpen(
                "repeated_tool_call",
                f"`{tool}` called {n} times with identical arguments",
                terminal_status="blocked",
            )

    # -- stalling ---------------------------------------------------------

    def observe_iteration(self, plan: Any, evaluation: Any) -> None:
        """``plan`` is a ``Plan``, ``evaluation`` an ``Evaluation``. Duck-typed to
        keep this module importable from ``memory`` without a cycle."""
        c = self.config
        if not c.enabled:
            return

        action = (plan.next_action.step_id, plan.next_action.tool)
        completed = set(getattr(evaluation, "completed_step_ids", []) or [])
        progressed = bool(completed - self._completed_steps)
        self._completed_steps |= completed

        if action == self._last_action and not progressed:
            self._stalled += 1
        else:
            # Either the agent moved to a different step, or it completed one.
            # Both are progress. Retrying a step after fixing something is normal.
            self._stalled = 0
        self._last_action = action

        if self._stalled >= c.max_stalled_iterations:
            raise CircuitOpen(
                "stalled",
                f"{self._stalled} consecutive repeats of step {action[0]} via "
                f"`{action[1]}` with no plan step completed",
                terminal_status="blocked",
            )

        if getattr(evaluation, "outcome", "") == "failure":
            self._failures += 1
        else:
            self._failures = 0
        if c.max_consecutive_failures and self._failures >= c.max_consecutive_failures:
            raise CircuitOpen(
                "consecutive_failures",
                f"{self._failures} failed iterations in a row",
                terminal_status="blocked",
            )

    def state(self) -> dict[str, Any]:
        return {
            "stalled_iterations": self._stalled,
            "distinct_calls": len(self._call_counts),
            "max_repeat": max(self._call_counts.values(), default=0),
            "spent_usd": round(self.ledger.total_usd, 6) if self.ledger else None,
        }


def compaction_for(
    model: str,
    *,
    trigger_ratio: float = 0.75,
    keep_iterations: int = 3,
    overrides: dict[str, ModelPricing] | None = None,
) -> CompactionConfig:
    """A ``CompactionConfig`` whose window is the model's *actual* window.

    ``CompactionConfig`` defaults to 180k tokens, which is right for a 200k model
    and leaves 820k on the table for a 1M one. Deriving it from the rate card
    means switching models moves the trigger with you::

        AgentConfig(llm=client, compaction=compaction_for(client.model))

    Unknown model: falls back to a conservative 128k.
    """
    pricing = resolve_pricing(model, overrides)
    window = pricing.context_window if pricing else 128_000
    return CompactionConfig(
        trigger_ratio=trigger_ratio,
        context_window_tokens=window,
        keep_iterations=keep_iterations,
    )


def _fingerprint(arguments: dict[str, Any]) -> str:
    import json

    try:
        return json.dumps(arguments, sort_keys=True, default=str)
    except Exception:
        return repr(sorted(arguments.items()))


# ---------------------------------------------------------------------------
# Recursive context pruning
# ---------------------------------------------------------------------------

_MERGE_PROMPT = """\
You are merging several partial summaries of one autonomous agent's run into a \
single summary. They are in chronological order. Later summaries supersede \
earlier ones where they conflict.

Preserve, in this order of priority:
1. Facts the agent will need again: paths, schemas, column names, IDs, values.
2. Approaches already tried and their outcomes, especially failures.
3. Decisions and their reasoning.
4. Unfinished work and blockers.

Discard restated instructions and reasoning that led nowhere. Dense prose or \
bullets. No preamble. Under {max_words} words.

PARTIAL SUMMARIES:
{parts}
"""


class RecursiveCompactor(Compactor):
    """Compaction that folds instead of truncating.

    ``Compactor`` summarises the whole discarded prefix in one completion. That
    breaks in exactly the case you need it: when the prefix is larger than the
    context window of the model you are asking to summarise it. A 400k-token
    history cannot be summarised in one call by a 200k-token model, and the naive
    implementation either errors or silently drops the front of the history --
    the part with the schema in it.

    So: chunk the prefix into spans that fit ``chunk_tokens``; summarise each
    (level 1); if the concatenated level-1 summaries still exceed the chunk
    budget, summarise *those* (level 2); repeat to ``max_depth``. Fold the prior
    summary in at the last step so it is never re-summarised more than once per
    compaction.

    ``k`` chunks costs ``k`` completions. Those completions are metered: pass
    ``meter=ledger.record``-shaped callable and the fold shows up in
    ``by_phase()["compaction"]``, where you will discover it is not free.

    What is lost, is lost. Summarisation is lossy by construction and recursion
    compounds it -- a level-2 summary is a summary of summaries, and the schema
    that survived level 1 may not survive level 2. That is what the scratchpad is
    for: anything the agent writes there is never compacted, at any depth. If a
    fact must survive an eight-hour run, the agent has to say so out loud.
    """

    def __init__(
        self,
        llm: LLMClient,
        config: CompactionConfig | None = None,
        on_compact: Callable[[int, int], None] | None = None,
        *,
        chunk_tokens: int = 12_000,
        max_depth: int = 3,
        meter: Callable[[Usage], None] | None = None,
    ) -> None:
        super().__init__(llm, config, on_compact)
        self.chunk_tokens = chunk_tokens
        self.max_depth = max_depth
        self.meter = meter or (lambda usage: None)

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

        rendered = [self._render(m) for m in old]
        summary = self._fold(rendered, prior_summary, depth=0)

        after = self.estimate_tokens(kept) + self.llm.count_tokens(summary)
        self.on_compact(before, after)
        return kept, summary

    # -- the fold ---------------------------------------------------------

    def _fold(self, parts: list[str], prior_summary: str, depth: int) -> str:
        chunks = self._chunk(parts)

        if len(chunks) == 1 or depth >= self.max_depth:
            return self._summarize(chunks[0] if chunks else "", prior_summary)

        level_up = [self._summarize(c, "") for c in chunks]
        if depth + 1 >= self.max_depth:
            return self._merge(level_up, prior_summary)
        return self._fold(level_up, prior_summary, depth + 1)

    def _chunk(self, parts: list[str]) -> list[str]:
        """Greedy pack, never splitting a rendered turn across chunks."""
        chunks: list[str] = []
        buf: list[str] = []
        size = 0
        for p in parts:
            n = self.llm.count_tokens(p)
            if buf and size + n > self.chunk_tokens:
                chunks.append("\n\n".join(buf))
                buf, size = [], 0
            buf.append(p)
            size += n
        if buf:
            chunks.append("\n\n".join(buf))
        return chunks

    def _summarize(self, history: str, prior_summary: str) -> str:
        from .transcript import SUMMARIZE_PROMPT

        prompt = SUMMARIZE_PROMPT.format(
            max_words=self.config.max_summary_words,
            prior=prior_summary or "(none)",
            history=history,
        )
        return self._complete(
            "You compress agent transcripts losslessly with respect to facts.", prompt
        )

    def _merge(self, summaries: list[str], prior_summary: str) -> str:
        parts = [prior_summary, *summaries] if prior_summary else summaries
        numbered = "\n\n".join(f"--- part {i + 1} ---\n{s}" for i, s in enumerate(parts))
        prompt = _MERGE_PROMPT.format(max_words=self.config.max_summary_words, parts=numbered)
        return self._complete("You merge partial summaries without losing facts.", prompt)

    def _complete(self, system: str, prompt: str) -> str:
        resp = self.llm.complete(
            system=system,
            messages=[Message(role="user", text=prompt)],
            tools=None,
            tool_choice="none",
            max_tokens=2048,
            temperature=0.0,
        )
        self.meter(resp.usage)
        return resp.text.strip()
