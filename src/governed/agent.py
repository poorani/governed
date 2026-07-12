"""The agentic loop.

    ANALYZE  -> the model writes a plan. Tools are withheld from the API call.
    ACT      -> the model must call the tool its plan named. Prose is rejected.
    EXECUTE  -> the registry runs it. Deterministic; no model involved.
    OBSERVE  -> the model grades its own output against its own criteria.
    ITERATE  -> back to ANALYZE, carrying the plan forward.
    SUBMIT   -> terminal tool. Produces a structured, self-assessed answer.

The separation is the point. Planning and acting in one call lets a model
rationalise whatever it happened to do; acting and observing in one call lets it
mark its own homework in the same breath it does it. Splitting them into
separate completions -- with tools *physically absent* from the ANALYZE and
OBSERVE requests -- makes each commitment falsifiable by the next phase.

Contract violations are not exceptions. They are fed back to the model as
corrective messages, up to ``budget.max_contract_retries`` per phase.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, TypeVar

from .config import AgentConfig, FeatureToggleConfig
from .contracts import (
    ContractViolation,
    Evaluation,
    Phase,
    Plan,
    parse_evaluation,
    parse_plan,
    validate_tool_choice,
)
from .llm.base import (
    LLMClient,
    LLMResponse,
    Message,
    ToolCall,
    ToolChoice,
    ToolResultBlock,
    Usage,
)
from .llm.factory import resolve_llm
from .memory.optimizer import (
    CircuitBreaker,
    CircuitOpen,
    CostLedger,
    RecursiveCompactor,
)
from .memory.session import IterationRecord, RunStatus, SessionState, ToolCallRecord
from .memory.transcript import Compactor
from .observability.decision_ledger import (
    DecisionLedger,
    DecisionLedgerConfig,
    JSONLDecisionLedger,
)
from .observability.events import EventType
from .observability.logger import TraceLogger
from .observability.telemetry import TelemetryCollector
from .prompts.system import (
    ACT_PROMPT,
    ANALYZE_PROMPT,
    BLOCKED_HINT,
    OBSERVE_PROMPT,
    VIOLATION_PROMPT,
    build_system_prompt,
)
from .security.content_safety import ContentSafetyScanner, KeywordSafetyProvider
from .security.guardrails import Gateway, GuardedRegistry, GuardrailConfig, PIIScanner
from .security.policy import GovernancePolicy
from .skills.loader import SkillConfig, SkillLibrary, resolve_skills
from .tools import default_tools, resolve_tools
from .tools.base import ToolConfig, ToolContext, ToolResult, ToolSpec
from .tools.registry import ToolRegistry

_T = TypeVar("_T")


@dataclass
class RunResult:
    status: RunStatus
    answer: str
    confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)
    unmet_requirements: list[str] = field(default_factory=list)
    iterations: int = 0
    total_tokens: int = 0
    #: Priced from the provider's reported token counts. See CostLedger.
    cost_usd: float = 0.0
    duration_s: float = 0.0
    session_id: str = ""
    state: SessionState | None = None

    @property
    def ok(self) -> bool:
        return self.status == "complete"

    def __str__(self) -> str:  # pragma: no cover
        return f"[{self.status}] {self.answer}"


class BudgetExceeded(Exception):
    def __init__(self, which: str) -> None:
        super().__init__(which)
        self.which = which


class Cancelled(Exception):
    """Raised internally when `Agent.cancel()` was called and `_drive`
    noticed at its next checkpoint. Never escapes `run()`/`resume()` --
    caught in `_drive` and turned into a normal `RunResult` with
    `status="cancelled"`, the same way `BudgetExceeded`/`CircuitOpen` become
    `"exhausted"`/`"blocked"` rather than propagating."""

    def __init__(self, reason: str = "") -> None:
        super().__init__(reason or "cancelled")
        self.reason = reason


class Agent:
    """Runs a goal to completion.

    ::

        agent = Agent(AgentConfig(llm=AnthropicClient()))
        result = agent.run("Profile data/sales.csv and report the top 3 regions.")
        print(result.answer)

    ``run()``/``resume()`` are not reentrant -- one `Agent` instance runs one
    goal at a time. Calling either from a second thread while the first is
    still in flight corrupts shared state (``_trace``, ``decision_ledger``,
    ...); construct a separate `Agent` per concurrent run instead. `cancel()`
    is the one method that *is* safe to call from another thread while
    ``run()``/``resume()`` are executing -- see its docstring.
    """

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        # AgentConfig.__post_init__ already resolved LLMConfig -> LLMClient;
        # re-resolving here is a no-op isinstance check, and gives the rest of
        # this class a statically-typed LLMClient instead of the config
        # field's wider `LLMClient | LLMConfig` constructor type.
        self.llm: LLMClient = resolve_llm(config.llm)

        self.workspace = Path(config.workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)

        self.skills = _resolve_skills(config.skills, config.skills_dirs)
        tools = _resolve_tools(config.tools, self.skills)

        self.governance: GovernancePolicy | None = _validate_governance(config.governance)
        if self.governance is not None:
            # Fails loudly, at construction, if a disallowed tool was wired
            # in -- see GovernancePolicy's docstring for why this is a raise
            # and not a silent filter.
            self.governance.enforce_allowed_tools(tools)

        self.features: FeatureToggleConfig | None = config.features

        # Resolved here, not in _drive, so _framework_paths() (below) can
        # protect the ledger's own file the same way it protects trace_path.
        # The DecisionLedger itself (which owns the hash chain) is
        # constructed per-run in _drive -- it needs a run_id.
        self._decision_ledger_config: DecisionLedgerConfig | None = config.decision_ledger
        if (
            self._decision_ledger_config is None
            and self.features is not None
            and self.features.decision_ledger
        ):
            self._decision_ledger_config = DecisionLedgerConfig(enabled=True)
        self._decision_ledger_store = None
        if self._decision_ledger_config is not None and self._decision_ledger_config.enabled:
            self._decision_ledger_store = self._decision_ledger_config.store or (
                JSONLDecisionLedger(
                    self.workspace / ".governed" / "decisions" / "ledger.jsonl"
                )
            )
        self.decision_ledger: DecisionLedger | None = None

        self.telemetry: TelemetryCollector | None = (
            TelemetryCollector()
            if self.features is not None and self.features.telemetry
            else None
        )

        # Money first: the compactor and the semantic scanner both spend it, and
        # both want to be metered.
        self.ledger = CostLedger(config.cost)
        self.breaker = CircuitBreaker(config.circuit_breaker, ledger=self.ledger)

        effective_guardrails = config.guardrails
        if self.governance is not None:
            # Folds sensitive_operations/approval_threshold into whatever
            # GuardrailConfig would otherwise be used, building one if there
            # wasn't one -- governance always ends up with an active gateway.
            effective_guardrails = self.governance.apply(effective_guardrails)
        effective_guardrails = _apply_feature_toggles(self.features, effective_guardrails)

        self.gateway: Gateway | None = None
        self.registry: ToolRegistry
        if effective_guardrails is not None and effective_guardrails.enabled:
            self.gateway = Gateway.from_config(
                effective_guardrails,
                workspace=self.workspace,
                protected_paths=self._framework_paths(),
            )
            self.registry = GuardedRegistry(
                tools, default_timeout_s=config.tool_timeout_s, gateway=self.gateway
            )
        else:
            self.registry = ToolRegistry(tools, default_timeout_s=config.tool_timeout_s)

        if "submit" not in self.registry.names:
            raise ValueError(
                "The tool set must include a terminal `submit` tool; without it a run "
                "can never end successfully. Add governed.tools.SubmitTool()."
            )
        for problem in self.skills.validate_tool_references(self.registry.names):
            raise ValueError(f"Skill/tool mismatch: {problem}")

        self._approve = config.resolve_approval()
        self.compactor: Compactor = (
            RecursiveCompactor(self.llm, config.compaction)
            if config.recursive_compaction
            else Compactor(self.llm, config.compaction)
        )
        # Populated by _execute, consumed by _observe on the same iteration.
        self._pending_results: list[ToolResultBlock] = []
        # Bound per-run in _drive; cost events need somewhere to go.
        self._trace: TraceLogger | None = None
        # threading.Event.set()/.is_set() are safe to call across threads
        # with no extra locking -- see cancel()'s docstring for the intended
        # usage (call run() on a worker thread, cancel() from the caller).
        self._cancel_event = threading.Event()
        self._cancel_reason = ""

    def _framework_paths(self) -> list[Path]:
        """Absolute paths the agent must never write to, whatever it plans.

        The skills it loads, the sessions it resumes from, the trace a human will
        read to find out what it did. An agent that can edit its own audit log
        has no audit log.
        """
        cfg = self.config
        paths: list[Path] = [Path(d) for d in cfg.skills_dirs]
        if cfg.trace_path:
            paths.append(Path(cfg.trace_path))
        store_dir = getattr(cfg.store, "directory", None)
        if store_dir:
            paths.append(Path(store_dir))
        ledger_path = getattr(self._decision_ledger_store, "path", None)
        if ledger_path:
            paths.append(Path(ledger_path))
        return [p for p in paths]

    # -- public API -------------------------------------------------------

    def run(self, goal: str, session_id: str | None = None) -> RunResult:
        self._cancel_event.clear()
        state = SessionState(goal=goal)
        if session_id:
            state.session_id = session_id
        return self._drive(state)

    def resume(self, session_id: str) -> RunResult:
        """Continue a checkpointed run. Side effects already applied are not redone."""
        self._cancel_event.clear()
        state = self.config.store.load(session_id)
        if state is None:
            raise KeyError(f"No session {session_id!r} in {type(self.config.store).__name__}")
        if state.status not in ("running", "exhausted", "cancelled"):
            raise ValueError(f"Session {session_id} already finished as {state.status!r}.")
        state.status = "running"
        return self._drive(state)

    def cancel(self, reason: str = "") -> None:
        """Ask a `run()`/`resume()` call in progress -- on another thread,
        since both are blocking -- to stop at its next checkpoint: before
        the next iteration begins, or after the current iteration's tool
        calls finish and before the next LLM call. Thread-safe.

        This is cooperative, not preemptive. An LLM request or tool call
        already in flight is not interrupted -- only the checkpoint after
        it fires early. There is no way to abort a call already in flight
        to a provider or a subprocess without risking whatever state it was
        mid-writing; the same trade-off `Budget`/`CircuitBreakerConfig`
        already make for their own limits. `RunResult.status` is
        `"cancelled"` when this fires before the run otherwise finished.

        Calling this before `run()`/`resume()` has started, or after either
        has already returned, has no effect -- both clear any prior
        cancellation at the start of a new call, so cancelling one run
        never carries over to the next on a reused `Agent` instance.
        """
        self._cancel_reason = reason
        self._cancel_event.set()

    # -- the loop ---------------------------------------------------------

    def _drive(self, state: SessionState) -> RunResult:
        cfg = self.config
        budget = cfg.budget
        started = time.time()

        extra_subscribers = list(cfg.subscribers)
        if self.telemetry is not None:
            extra_subscribers.append(self.telemetry)
        trace = TraceLogger(
            state.run_id,
            jsonl_path=cfg.trace_path,
            console=cfg.console,
            verbose=cfg.verbose,
            extra_subscribers=extra_subscribers,
        )
        self.compactor.on_compact = lambda b, a: trace.emit(
            EventType.COMPACTION, iteration=state.iteration, before_tokens=b, after_tokens=a
        )
        # The fold is not free. Meter it, so it shows up under `compaction` in
        # ledger.by_phase() instead of hiding inside the model's bill.
        if isinstance(self.compactor, RecursiveCompactor):
            self.compactor.meter = lambda usage: self._record_cost(
                state, usage, phase="compaction", trace=trace
            )

        self.ledger.seed(float(state.scratchpad.get("_cost_usd", 0.0)))
        self.ledger.on_unpriced = lambda model: trace.emit(
            EventType.COST_WARNING,
            iteration=state.iteration,
            message=(
                f"No rate card for {model!r}: its spend is counted as $0.00 and the "
                "cost circuit breaker cannot protect this run. Pass "
                "CostConfig(pricing_overrides={...})."
            ),
        )
        self.breaker.on_warn = lambda reason, detail: trace.emit(
            EventType.COST_WARNING, iteration=state.iteration, message=f"{reason}: {detail}"
        )
        if self.gateway is not None:
            self.gateway.emit = trace.emit
            self.gateway.goal = state.goal

        self.decision_ledger = (
            DecisionLedger(
                state.run_id,
                state.session_id,
                store=self._decision_ledger_store,
                sinks=self._decision_ledger_config.sinks,
            )
            if self._decision_ledger_store is not None and self._decision_ledger_config
            else None
        )

        trace.emit(
            EventType.RUN_START,
            goal=state.goal,
            model=self.llm.model,
            tools=sorted(self.registry.names),
            skills=sorted(self.skills.names),
            resumed=bool(state.iterations),
        )

        self._trace = trace
        signals: dict[str, Any] = {}
        try:
            while True:
                self._check_budget(state, started, trace)
                self._check_cancellation()
                it = state.begin_iteration()
                trace.emit(EventType.ITERATION_START, iteration=it.index)

                plan = self._analyze(state, trace)
                it.plan = plan.to_dict()

                # Gateway.decisions grows monotonically across the whole run;
                # this slice is what screen_call produced for *this*
                # iteration's calls, for the decision ledger below.
                gw_start = len(self.gateway.decisions) if self.gateway else 0
                calls = self._act(state, plan, trace)
                results = self._execute(state, plan, calls, signals, trace)
                safety_checks = list(self.gateway.decisions[gw_start:]) if self.gateway else []

                if "submitted" in signals:
                    self._record_decision(
                        state,
                        it,
                        plan,
                        safety_checks,
                        evaluation=None,
                        final=signals["submitted"],
                    )
                    it.ended_at = time.time()
                    self._checkpoint(state)
                    break

                # A second checkpoint here, not just at the top of the loop:
                # EXECUTE (a tool call, possibly `execute_code` running up to
                # its own timeout) is where an iteration spends most of its
                # wall-clock time. Checking again before OBSERVE's LLM call
                # means a cancellation lands as soon as the current tool
                # call finishes, not after one more full iteration.
                self._check_cancellation()

                evaluation = self._observe(state, plan, results, trace)
                it.evaluation = evaluation.to_dict()
                self._record_decision(
                    state, it, plan, safety_checks, evaluation=it.evaluation, final=None
                )
                it.ended_at = time.time()
                trace.emit(
                    EventType.ITERATION_END, iteration=it.index, duration_s=it.duration_s
                )
                self._checkpoint(state)

                # The breaker's thresholds are deliberately looser than
                # `Budget.max_consecutive_failures`, so the agent's own, more
                # specific "three failures in a row" diagnosis fires first on a
                # run that is failing rather than merely spinning.
                self.breaker.observe_iteration(plan, evaluation)

                if state.consecutive_failures() >= budget.max_consecutive_failures:
                    state.status = "blocked"
                    signals["submitted"] = {
                        "answer": (
                            f"Halted after {budget.max_consecutive_failures} consecutive "
                            f"failed iterations. Last diagnosis: {evaluation.evidence}"
                        ),
                        "status": "blocked",
                        "confidence": 0.0,
                        "evidence": [evaluation.evidence],
                        "unmet_requirements": [evaluation.next_step],
                    }
                    break

        except Cancelled as exc:
            trace.emit(EventType.CANCELLED, iteration=state.iteration, reason=exc.reason)
            state.status = "cancelled"
            signals.setdefault(
                "submitted",
                {
                    "answer": (
                        f"Run cancelled: {exc.reason}" if exc.reason else "Run cancelled."
                    ),
                    "status": "blocked",
                    "confidence": 0.0,
                    "evidence": [],
                    "unmet_requirements": [state.goal],
                },
            )
        except CircuitOpen as exc:
            trace.emit(
                EventType.CIRCUIT_OPEN,
                iteration=state.iteration,
                reason=exc.reason,
                detail=exc.detail,
                **self.breaker.state(),
            )
            state.status = exc.terminal_status
            signals.setdefault(
                "submitted",
                {
                    "answer": (
                        f"Terminated by the circuit breaker ({exc.reason}): {exc.detail}. "
                        "No further work was attempted."
                    ),
                    "status": "blocked",
                    "confidence": 0.0,
                    "evidence": [],
                    "unmet_requirements": [state.goal],
                },
            )
        except BudgetExceeded as exc:
            trace.emit(EventType.BUDGET_EXCEEDED, iteration=state.iteration, which=exc.which)
            state.status = "exhausted"
            signals.setdefault(
                "submitted",
                {
                    "answer": f"Ran out of budget ({exc.which}) before completing the goal.",
                    "status": "blocked",
                    "confidence": 0.0,
                    "evidence": [],
                    "unmet_requirements": [state.goal],
                },
            )
        except ContractViolation as exc:
            trace.emit(
                EventType.ERROR,
                iteration=state.iteration,
                phase=exc.phase.value,
                message=f"Unrecoverable contract violation: {exc.reason}",
            )
            state.status = "failed"
            signals.setdefault(
                "submitted",
                {
                    "answer": (
                        f"The model could not produce valid {exc.phase.value}-phase output "
                        f"after {self.config.budget.max_contract_retries + 1} attempts: "
                        f"{exc.reason}"
                    ),
                    "status": "blocked",
                    "confidence": 0.0,
                    "evidence": [],
                    "unmet_requirements": [state.goal],
                },
            )
        finally:
            self.registry.shutdown()

        return self._finalize(state, signals, started, trace)

    # -- phases -----------------------------------------------------------

    def _analyze(self, state: SessionState, trace: TraceLogger) -> Plan:
        hint = ""
        failures = state.consecutive_failures()
        if failures:
            hint = BLOCKED_HINT.format(n=failures)

        prompt = ANALYZE_PROMPT.format(
            iteration=state.iteration,
            max_iterations=self.config.budget.max_iterations,
            blocked_hint=hint,
        )
        self._push(state, Message(role="user", text=prompt))

        def attempt() -> Plan:
            resp = self._complete(
                state, tools=None, tool_choice="none", phase=Phase.ANALYZE.value
            )
            self._push(state, Message(role="assistant", text=resp.text))
            return parse_plan(resp.text)

        plan = self._with_retries(Phase.ANALYZE, attempt, state, trace)
        trace.emit(
            EventType.PLAN_CREATED,
            iteration=state.iteration,
            phase=Phase.ANALYZE.value,
            **plan.to_dict(),
        )
        return plan

    def _act(self, state: SessionState, plan: Plan, trace: TraceLogger) -> list[ToolCall]:
        prompt = ACT_PROMPT.format(
            iteration=state.iteration,
            tool=plan.next_action.tool,
            step_id=plan.next_action.step_id,
            rationale=plan.next_action.rationale,
        )
        self._push(state, Message(role="user", text=prompt))

        def attempt() -> list[ToolCall]:
            resp = self._complete(
                state,
                tools=self.registry.schemas(),
                tool_choice="required",
                phase=Phase.ACT.value,
            )
            self._push(
                state,
                Message(role="assistant", text=resp.text, tool_calls=resp.tool_calls),
            )
            validate_tool_choice(
                plan, [tc.name for tc in resp.tool_calls], self.registry.names
            )
            return resp.tool_calls

        return self._with_retries(Phase.ACT, attempt, state, trace)

    def _execute(
        self,
        state: SessionState,
        plan: Plan,
        calls: list[ToolCall],
        signals: dict[str, Any],
        trace: TraceLogger,
    ) -> list[ToolResult]:
        ctx = ToolContext(
            workspace=self.workspace,
            scratchpad=state.scratchpad,
            run_id=state.run_id,
            iteration=state.iteration,
            approve=self._approve_and_trace(trace, state),
            signals=signals,
        )

        results: list[ToolResult] = []
        blocks: list[ToolResultBlock] = []

        for call in calls:
            trace.emit(
                EventType.TOOL_CALL,
                iteration=state.iteration,
                phase=Phase.EXECUTE.value,
                tool=call.name,
                call_id=call.id,
                arguments=call.arguments,
                # The "why", carried from the plan that authorised this call.
                rationale=plan.next_action.rationale,
                step_id=plan.next_action.step_id,
            )

            # Terminal tools are exempt: `submit` is how a run is *supposed* to
            # end, and tripping the loop detector on it would be perverse.
            tool = self.registry.get(call.name)
            if tool is None or not getattr(tool, "terminal", False):
                self.breaker.observe_tool_call(call.name, call.arguments)

            result = self.registry.invoke(call.name, call.arguments, ctx)
            state.tool_call_count += 1
            results.append(result)

            preview = result.to_model_text()
            blocks.append(
                ToolResultBlock(call_id=call.id, content=preview, is_error=not result.ok)
            )

            trace.emit(
                EventType.TOOL_RESULT,
                iteration=state.iteration,
                phase=Phase.EXECUTE.value,
                tool=call.name,
                call_id=call.id,
                ok=result.ok,
                error_code=result.error.code.value if result.error else None,
                duration_ms=result.duration_ms,
                truncated=result.truncated,
                preview=preview[:2000],
            )
            if call.name == "load_skill" and result.ok:
                trace.emit(
                    EventType.SKILL_LOADED,
                    iteration=state.iteration,
                    skill=call.arguments.get("name"),
                )

            rec = state.current
            if rec is not None:
                rec.tool_calls.append(
                    ToolCallRecord(
                        call_id=call.id,
                        tool=call.name,
                        arguments=call.arguments,
                        rationale=plan.next_action.rationale,
                        step_id=plan.next_action.step_id,
                        ok=result.ok,
                        result_preview=preview[:2000],
                        error_code=result.error.code.value if result.error else None,
                        duration_ms=result.duration_ms,
                        artifacts=[a.__dict__ for a in result.artifacts],
                    )
                )

        # Tool results ride on the next user turn (see OBSERVE, which reuses it).
        state.scratchpad.update(ctx.scratchpad)
        self._pending_results = blocks
        return results

    def _observe(
        self,
        state: SessionState,
        plan: Plan,
        results: list[ToolResult],
        trace: TraceLogger,
    ) -> Evaluation:
        prompt = OBSERVE_PROMPT.format(
            iteration=state.iteration,
            success_criteria=plan.next_action.success_criteria,
        )
        # One user turn carrying both the tool results and the OBSERVE instruction,
        # so provider role-alternation rules hold.
        self._push(
            state,
            Message(role="user", text=prompt, tool_results=self._pending_results),
        )
        self._pending_results = []

        valid_ids = {s.id for s in plan.steps}

        def attempt() -> Evaluation:
            resp = self._complete(
                state, tools=None, tool_choice="none", phase=Phase.OBSERVE.value
            )
            self._push(state, Message(role="assistant", text=resp.text))
            return parse_evaluation(resp.text, valid_ids)

        evaluation = self._with_retries(Phase.OBSERVE, attempt, state, trace)
        trace.emit(
            EventType.EVALUATION_CREATED,
            iteration=state.iteration,
            phase=Phase.OBSERVE.value,
            **evaluation.to_dict(),
        )
        return evaluation

    # -- machinery --------------------------------------------------------

    def _with_retries(
        self, phase: Phase, attempt: Callable[[], _T], state: SessionState, trace: TraceLogger
    ) -> _T:
        max_retries = self.config.budget.max_contract_retries
        last: ContractViolation | None = None

        for i in range(max_retries + 1):
            try:
                return attempt()
            except ContractViolation as exc:
                last = exc
                trace.emit(
                    EventType.CONTRACT_VIOLATION,
                    iteration=state.iteration,
                    phase=phase.value,
                    reason=exc.reason,
                    attempt=i + 1,
                )
                if state.current is not None:
                    state.current.violations.append(
                        {"phase": phase.value, "reason": exc.reason}
                    )
                if i == max_retries:
                    break
                self._push(state, self._violation_message(state, exc, i + 2, max_retries + 1))

        assert last is not None
        raise last

    def _violation_message(
        self, state: SessionState, exc: ContractViolation, attempt: int, total: int
    ) -> Message:
        text = VIOLATION_PROMPT.format(
            phase=exc.phase.value, feedback=exc.feedback, attempt=attempt, max_attempts=total
        )
        # If the offending assistant turn opened tool_use blocks, every one of them
        # must be closed by a tool_result or the provider rejects the request.
        last = state.transcript[-1] if state.transcript else None
        blocks: list[ToolResultBlock] = []
        if last and last.role == "assistant" and last.tool_calls:
            blocks = [
                ToolResultBlock(
                    call_id=tc.id,
                    content="Not executed: the call violated the phase contract.",
                    is_error=True,
                )
                for tc in last.tool_calls
            ]
        return Message(role="user", text=text, tool_results=blocks)

    def _complete(
        self,
        state: SessionState,
        *,
        tools: list[dict[str, Any]] | None,
        tool_choice: ToolChoice,
        phase: str = "",
    ) -> LLMResponse:
        system = self._system(state)
        self._maybe_compact(state, system)

        trace = self._trace
        model = self.llm.model
        if trace is not None:
            trace.emit(
                EventType.LLM_REQUEST, iteration=state.iteration, phase=phase, model=model
            )
        started = time.monotonic()
        try:
            resp = self.llm.complete(
                system=system,
                messages=state.transcript,
                tools=tools,
                tool_choice=tool_choice,
                max_tokens=self.config.max_tokens_per_call,
                temperature=self.config.temperature,
            )
        except Exception as exc:
            if trace is not None:
                trace.emit(
                    EventType.LLM_RESPONSE,
                    iteration=state.iteration,
                    phase=phase,
                    model=model,
                    latency_ms=_elapsed_ms(started),
                    status="error",
                    error_type=type(exc).__name__,
                    status_code=_extract_status_code(exc),
                )
            raise
        if trace is not None:
            trace.emit(
                EventType.LLM_RESPONSE,
                iteration=state.iteration,
                phase=phase,
                model=model,
                latency_ms=_elapsed_ms(started),
                status="ok",
                input_tokens=resp.usage.input_tokens,
                output_tokens=resp.usage.output_tokens,
            )
        state.record_usage(resp.usage)
        self._record_cost(state, resp.usage, phase=phase, trace=trace)
        # Checked *after* the call: the money is already spent. Bound
        # max_tokens_per_call if you need the ceiling to be exact.
        self.breaker.check_cost()
        return resp

    def _record_cost(
        self, state: SessionState, usage: Usage, *, phase: str, trace: TraceLogger | None
    ) -> None:
        cost = self.ledger.record(
            self.llm.model, usage, phase=phase, iteration=state.iteration
        )
        # Checkpointed under a reserved key. The guardrail refuses model writes to
        # `_`-prefixed keys, so an agent cannot resume with a fresh meter.
        state.scratchpad["_cost_usd"] = self.ledger.total_usd
        if trace is not None:
            trace.emit(
                EventType.COST_RECORDED,
                iteration=state.iteration,
                phase=phase,
                model=cost.model,
                call_usd=round(cost.usd, 6),
                run_usd=round(self.ledger.total_usd, 6),
                priced=cost.priced,
            )

    def _system(self, state: SessionState) -> str:
        prompt = build_system_prompt(
            goal=state.goal,
            tool_specs=self.registry.specs(),
            skill_index=self.skills.index_markdown(),
            extra_instructions=self.config.extra_instructions,
        )
        if state.summary:
            prompt += (
                "\n\n# Summary of earlier iterations\n\n"
                "Older turns were compressed to fit the context window. "
                "Treat the following as established fact.\n\n" + state.summary
            )
        if state.scratchpad:
            keys = ", ".join(
                f"`{k}`" for k in sorted(state.scratchpad) if not k.startswith("_")
            )
            if keys:
                prompt += f"\n\n# Scratchpad keys\n\n{keys}\n\nRead them with `scratchpad`."
        return prompt

    def _maybe_compact(self, state: SessionState, system: str) -> None:
        if not self.compactor.should_compact(state.transcript, system):
            return
        kept, summary = self.compactor.compact(
            state.transcript, state.summary, state.iteration
        )
        # A transcript must open on a user turn.
        while kept and kept[0].role != "user":
            kept.pop(0)
        if not kept:
            return
        state.transcript = kept
        state.summary = summary
        state.summarized_through = max(
            0, state.iteration - self.config.compaction.keep_iterations
        )

    def _push(self, state: SessionState, message: Message) -> None:
        message.meta["iteration"] = state.iteration
        state.add_message(message)

    def _approve_and_trace(
        self, trace: TraceLogger, state: SessionState
    ) -> Callable[[ToolSpec, dict[str, Any]], bool]:
        def approve(spec: ToolSpec, args: dict[str, Any]) -> bool:
            trace.emit(
                EventType.APPROVAL_REQUESTED,
                iteration=state.iteration,
                tool=spec.name,
                safety=spec.safety.value,
            )
            decision = self._approve(spec, args)
            trace.emit(
                EventType.APPROVAL_DECIDED,
                iteration=state.iteration,
                tool=spec.name,
                approved=decision,
            )
            return decision

        return approve

    def _check_budget(self, state: SessionState, started: float, trace: TraceLogger) -> None:
        b = self.config.budget
        if state.iteration >= b.max_iterations:
            raise BudgetExceeded(f"max_iterations={b.max_iterations}")
        if state.usage.total >= b.max_tokens:
            raise BudgetExceeded(f"max_tokens={b.max_tokens}")
        if state.tool_call_count >= b.max_tool_calls:
            raise BudgetExceeded(f"max_tool_calls={b.max_tool_calls}")
        if time.time() - started >= b.max_wall_seconds:
            raise BudgetExceeded(f"max_wall_seconds={b.max_wall_seconds}")

    def _check_cancellation(self) -> None:
        if self._cancel_event.is_set():
            raise Cancelled(self._cancel_reason)

    def _checkpoint(self, state: SessionState) -> None:
        if self.config.checkpoint_every_iteration:
            self.config.store.save(state)

    def _record_decision(
        self,
        state: SessionState,
        it: IterationRecord,
        plan: Plan,
        safety_checks: list[dict[str, Any]],
        *,
        evaluation: dict[str, Any] | None,
        final: dict[str, Any] | None,
    ) -> None:
        if self.decision_ledger is None:
            return
        self.decision_ledger.record(
            iteration=it.index,
            goal=state.goal,
            plan=it.plan,
            rationale=plan.next_action.rationale,
            tool=plan.next_action.tool,
            tool_calls=[dict(vars(tc)) for tc in it.tool_calls],
            safety_checks=safety_checks,
            evaluation=evaluation,
            violations=list(it.violations),
            final=final,
        )

    def _finalize(
        self, state: SessionState, signals: dict[str, Any], started: float, trace: TraceLogger
    ) -> RunResult:
        submitted = signals.get("submitted") or {
            "answer": "The run ended without a submission.",
            "status": "blocked",
            "confidence": 0.0,
            "evidence": [],
            "unmet_requirements": [state.goal],
        }
        state.final_answer = submitted
        if state.status == "running":
            state.status = (
                submitted["status"] if submitted["status"] != "blocked" else "blocked"
            )
        self.config.store.save(state)

        duration = time.time() - started

        # Guaranteed run-end record, however the run ended (submit, budget
        # exceeded, circuit breaker, contract violation) -- _finalize is the
        # one place every path in _drive funnels through.
        if self.decision_ledger is not None:
            self.decision_ledger.record(
                iteration=state.iteration,
                goal=state.goal,
                plan=None,
                rationale="",
                tool="__run_end__",
                tool_calls=[],
                safety_checks=[],
                evaluation=None,
                violations=[],
                final={
                    **submitted,
                    "cost_usd": round(self.ledger.total_usd, 6),
                    "duration_s": duration,
                    "iterations": state.iteration,
                    "total_tokens": state.usage.total,
                },
            )

        trace.emit(
            EventType.RUN_END,
            iteration=state.iteration,
            status=state.status,
            answer=submitted["answer"],
            iterations=state.iteration,
            total_tokens=state.usage.total,
            cost_usd=round(self.ledger.total_usd, 6),
            duration_s=duration,
        )
        trace.close()

        return RunResult(
            status=state.status,
            answer=submitted["answer"],
            confidence=float(submitted.get("confidence", 0.0)),
            evidence=list(submitted.get("evidence", [])),
            unmet_requirements=list(submitted.get("unmet_requirements", [])),
            iterations=state.iteration,
            total_tokens=state.usage.total,
            cost_usd=self.ledger.total_usd,
            duration_s=duration,
            session_id=state.session_id,
            state=state,
        )


def _validate_governance(value: Any) -> GovernancePolicy | None:
    """``AgentConfig.governance`` is typed loosely (see its field comment, and
    ``guardrails`` right above it) so ``config.py`` doesn't need a ``security``
    import. This is where that gets checked for real, the same way
    ``resolve_llm`` checks ``AgentConfig.llm``: a wrong type should fail at
    construction, not three iterations into a run.
    """
    if value is None:
        return None
    if not isinstance(value, GovernancePolicy):
        raise TypeError(
            f"AgentConfig.governance must be a GovernancePolicy, got {type(value).__name__}"
        )
    return value


def _resolve_skills(
    skills: SkillLibrary | SkillConfig | None, skills_dirs: list[str | Path]
) -> SkillLibrary:
    """``AgentConfig.skills`` accepts a live ``SkillLibrary``, a data-only
    ``SkillConfig``, or nothing (fall back to ``skills_dirs``, the original,
    still-supported field).
    """
    if isinstance(skills, SkillConfig):
        return resolve_skills(skills)
    return skills or SkillLibrary.from_dirs(*skills_dirs)


def _resolve_tools(tools: list[Any] | ToolConfig | None, skills: SkillLibrary) -> list[Any]:
    """``AgentConfig.tools`` accepts a live ``list[Tool]``, a data-only
    ``ToolConfig``, or nothing (``default_tools(skills)``)."""
    if tools is None:
        return default_tools(skills)
    if isinstance(tools, ToolConfig):
        return resolve_tools(tools, skills)
    return tools


def _apply_feature_toggles(
    features: FeatureToggleConfig | None, guardrails: Any | None
) -> Any | None:
    """Fill in defaults for whatever ``FeatureToggleConfig`` turns on that
    isn't already configured more specifically. Runs *after* explicit
    ``guardrails``/``governance`` resolution -- see ``FeatureToggleConfig``'s
    docstring: this only fills a gap, it never overrides an explicit setting.
    """
    if features is None:
        return guardrails
    if guardrails is None and features.guardrails:
        guardrails = GuardrailConfig()
    if features.content_safety:
        scanner = ContentSafetyScanner(KeywordSafetyProvider())
        if guardrails is None:
            guardrails = GuardrailConfig(content_safety_scanners=[scanner])
        elif not guardrails.content_safety_scanners:
            guardrails = replace(guardrails, content_safety_scanners=[scanner])
    if features.pii_detection:
        if guardrails is None:
            guardrails = GuardrailConfig(extra_scanners=[PIIScanner()])
        elif not any(isinstance(s, PIIScanner) for s in guardrails.extra_scanners):
            guardrails = replace(
                guardrails, extra_scanners=[*guardrails.extra_scanners, PIIScanner()]
            )
    return guardrails


def _elapsed_ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


def _extract_status_code(exc: BaseException) -> int | None:
    """Best-effort HTTP status from a provider SDK exception.

    ``LLMClient`` is provider-agnostic, so there is no guaranteed status code
    to read. ``anthropic``/``openai`` SDK errors expose ``.status_code``
    directly, or on their nested ``.response``; anything else yields ``None``
    and ``TelemetryCollector`` falls back to the exception's type name.
    """
    code = getattr(exc, "status_code", None)
    if isinstance(code, int):
        return code
    response = getattr(exc, "response", None)
    code = getattr(response, "status_code", None)
    return code if isinstance(code, int) else None
