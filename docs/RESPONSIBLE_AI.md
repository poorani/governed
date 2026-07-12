# Responsible AI usage

`governed` is built to be a **governed, auditable agent runtime**, not just an
agent loop with tools bolted on. This document is the map: what governs an
agent's behaviour, what a human sees and can veto, what gets recorded, where
the actual safety boundaries are (as opposed to where they only look like
boundaries), and how every one of those things can be overridden -- and by
whom.

Read this alongside [`README.md`](../README.md) (mechanics and API) and
[`GUIDE.md`](GUIDE.md) (architecture and worked deployment examples). This
document answers a different question than either: *if I am the person
accountable for what this agent does, what do I actually control, and how do
I prove it after the fact?*

## Table of contents

- [The governance model in one picture](#the-governance-model-in-one-picture)
- [1. Policy: allowed tools, sensitive operations, approval thresholds](#1-policy-allowed-tools-sensitive-operations-approval-thresholds)
- [2. Risk classification](#2-risk-classification)
- [2a. Content safety screening (the Responsible AI execution layer)](#2a-content-safety-screening-the-responsible-ai-execution-layer)
- [3. Human-in-the-loop approval](#3-human-in-the-loop-approval)
- [4. Structured run traces](#4-structured-run-traces)
- [4a. The decision ledger: tamper-evident and exportable](#4a-the-decision-ledger-tamper-evident-and-exportable)
- [5. Observability and audit logging](#5-observability-and-audit-logging)
- [6. Model/provider governance](#6-modelprovider-governance)
- [7. Safety boundaries: what's guaranteed, what's detection](#7-safety-boundaries-whats-guaranteed-whats-detection)
- [8. Override behaviour: who can change what](#8-override-behaviour-who-can-change-what)
- [Tests that verify all of this](#tests-that-verify-all-of-this)
- [Responsible AI readiness checklist](#responsible-ai-readiness-checklist)

## The governance model in one picture

```
 operator sets, once, per deployment
 ┌─────────────────────────────────────────────────────────────────┐
 │  GovernancePolicy                        ProviderPolicy         │
 │  - allowed_tools                         - allowed_providers    │
 │  - sensitive_operations                  - allowed_models       │
 │  - approval_threshold                                           │
 └───────────────┬───────────────────────────────┬─────────────────┘
                 │ enforced at Agent construction  │ enforced by resolve_llm
                 ▼                                 ▼
        Agent(AgentConfig(tools=..., governance=..., llm=LLMConfig(...), provider_policy=...))
                 │
                 ▼
   ANALYZE ──▶ ACT ──▶ EXECUTE ──▶ OBSERVE ──▶ ITERATE ──▶ SUBMIT
       plan     tool      Gateway:              self-       RunResult
                commits   RiskPolicy → tier      grade
                          → scanners (incl. ContentSafetyScanner)
                          → tier DANGER? → approval
                          → tier fallback? → safer redirect
                          (Approver: terminal / webhook / policy)
                 │                        │                        │
                 ▼                        ▼                        ▼
         SessionState.iterations   Event stream            DecisionLedger (if enabled)
         (plan, action, evidence,  (JSONL / console /       one hash-chained record per
          evaluation -- per          your sink)              iteration + guaranteed
          iteration)                     │                   run-end record
                 │                       ▼                        │
                 │               TraceLogger,                     ├──▶ store (JSONL / yours)
                 │               trace_to_markdown,                    verify_chain()
                 │               TelemetryCollector                    detects tampering
                 └───────────┬─────────────────────┐                   │
                              ▼                      ▼                 ▼
                   build_audit_report(agent, result)          sinks: Http / OTel /
                   -> AuditReport (JSON / Markdown)             your own -- Splunk,
                                                                 Datadog, New Relic,
                                                                 Dynatrace, OTel Collector
```

Four pieces, deliberately kept separate:

* **`governed.security.guardrails`** does *per-call* work: compute a risk
  tier from arguments, sweep for injection/secrets/destructive commands, ask
  a human when the tier demands it, sweep tool *results* before they
  re-enter the model's context. This is the mechanism, and the chokepoint
  everything else below plugs into.
* **`governed.security.content_safety`** is the Responsible AI execution
  layer proper: a pluggable `SafetyProvider` interface for harmful-intent,
  policy-violation, bias, and unsafe-behaviour screening, adapted into the
  same `Scanner` protocol the guardrails above already use, plus a third
  disposition (`fallback`) alongside escalate-to-a-human and hard-block.
* **`governed.security.policy.GovernancePolicy`** and
  **`governed.llm.policy.ProviderPolicy`** are *deployment*-level policy:
  the small number of knobs a platform team sets once, that every
  application team building on the framework inherits automatically, instead
  of each one hand-assembling `RiskPolicy`/`GuardrailConfig` from scratch.
* **`governed.observability.decision_ledger`** is the accountability record:
  a hash-chained `DecisionRecord` per iteration, independent of every piece
  above, that can be verified for tampering and streamed to wherever your
  organisation actually reviews agent activity.

## 1. Policy: allowed tools, sensitive operations, approval thresholds

```python
from governed import Agent, AgentConfig, GovernancePolicy, RiskTier, AllowTierApprover

policy = GovernancePolicy(
    # This deployment may only ever have these tools, regardless of what
    # AgentConfig(tools=...) tries to register. `submit` is always implicit.
    allowed_tools=frozenset({"file_system", "analyze_data", "submit"}),

    # Named operations that must always require a human, however the built-in
    # risk tiers would otherwise classify them. Format matches RiskPolicy's
    # discriminator convention: "tool" or "tool:operation".
    sensitive_operations=frozenset({"file_system:delete"}),

    # This deployment runs unattended up to WARNING; DANGER always stops.
    approval_threshold=RiskTier.WARNING,
    approver=AllowTierApprover(RiskTier.WARNING),  # or TerminalApprover(), WebhookApprover(...)
)

agent = Agent(AgentConfig(llm=..., tools=[...], governance=policy))
```

Two properties make this a *governance* layer rather than just another config
knob:

* **`allowed_tools` fails at construction, not at call time.** If
  `AgentConfig(tools=...)` includes a tool outside the allowlist,
  `Agent(...)` raises `GovernanceViolation` immediately -- a
  misconfigured deployment never starts, rather than failing three
  iterations in when the model happens to reach for the disallowed tool. It
  is a raise, not a silent filter: a tool that quietly isn't there is much
  harder to notice in review than a startup error.
* **`sensitive_operations` can only raise a tier, never lower one** -- the
  same invariant `RiskPolicy` itself enforces (see [Risk
  classification](#2-risk-classification)). There is no way to use governance
  to make an operation *less* supervised than the built-in defaults say it
  should be; that would defeat the point of a policy layer whose job is to
  add guarantees, not remove them.

`GovernancePolicy.apply()` folds `sensitive_operations` and
`approval_threshold` into a `GuardrailConfig` -- building one if you didn't
supply one via `AgentConfig(guardrails=...)`, or layering on top of an
explicit one without discarding anything it already set (its scanners, its
`untrusted_result_action`, its own `RiskPolicy` overrides). If a
`GovernancePolicy` is present, guardrails end up enabled, full stop --
`GuardrailConfig(enabled=False)` alongside an active `GovernancePolicy` does
not win.

## 2. Risk classification

Every tool call gets a `RiskTier` before it runs:

| Tier | Meaning | Default disposition |
|---|---|---|
| `SAFE` | Observes, computes, reads. | Executes automatically. |
| `WARNING` | Creates or edits inside the sandbox. | Executes automatically, logged. |
| `DANGER` | Deletes, runs commands, mutates the outside world. | Stops for a human. |

The tier is computed **per call, from the arguments** (`RiskPolicy.assess`),
not just from the tool's declared class-level `ToolSafety`. This is the
difference between `file_system(operation="read")` and
`file_system(operation="delete")` -- one tool, two very different blast
radii, and the built-in `RiskPolicy` already tells them apart via a
discriminator argument (`operation`, `language`, ...). `GovernancePolicy`
composes with this; it doesn't replace it. A deployment can layer
`sensitive_operations={"file_system:write"}` on top of the built-ins, or add
its own escalation callables directly to `RiskPolicy.escalations` for logic
that can't be expressed as a static name.

Scanners (`InjectionScanner`, `SecretExfiltrationScanner`,
`DestructiveCommandScanner`, `SemanticInjectionScanner`) additionally produce
`Finding`s that can escalate a tier or, for a small set of things with no
legitimate reading (private keys, fork bombs, writes to a block device),
block the call outright regardless of who is willing to approve it. See
[Safety boundaries](#7-safety-boundaries-whats-guaranteed-whats-detection)
for what these scanners can and cannot promise.

## 2a. Content safety screening (the Responsible AI execution layer)

Everything in §2 screens for a specific, narrow attack shape. **Content
safety screening is the general-purpose slot**: it evaluates a proposed
action -- a tool call's arguments, or the result it produced -- for
**harmful intent, policy violations, bias, and unsafe behaviour**, before
`EXECUTE` runs it, using a provider you plug in.

```python
from governed import GuardrailConfig, ContentSafetyScanner, CategoryPolicy
from governed.security.content_safety import (
    HARMFUL_INTENT, POLICY_VIOLATION, BIAS, UNSAFE_BEHAVIOR,
)

guardrails = GuardrailConfig(
    content_safety_scanners=[
        ContentSafetyScanner(
            my_safety_provider,  # any SafetyProvider -- see below
            category_policies={
                HARMFUL_INTENT: CategoryPolicy("escalate", RiskTier.DANGER),
                POLICY_VIOLATION: CategoryPolicy("fallback"),
                BIAS: CategoryPolicy("escalate", RiskTier.WARNING),
                UNSAFE_BEHAVIOR: CategoryPolicy("block"),  # your call, not the library's
            },
        ),
    ],
)
```

**The integration seam.** `SafetyProvider` is one method:

```python
class SafetyProvider(Protocol):
    name: str
    def evaluate(self, text: str, *, source: str) -> SafetyVerdict: ...
```

Implement it against your organisation's moderation API (a content-safety
endpoint, an internal OPA-style policy engine, a hosted classifier) and it
plugs straight in. Two reference implementations ship, in the same spirit as
`DestructiveCommandScanner` and `SemanticInjectionScanner` -- honest about
what they can and can't see:

* **`KeywordSafetyProvider`** -- regex, zero dependencies, catches the
  unambiguous cases (weapon synthesis, malware authoring, self-harm
  encouragement, credential harvesting). It deliberately **does not**
  attempt to detect `BIAS`: a fixed word list produces confident-looking
  false positives and negatives in roughly equal, unhelpful measure, and
  bias review needs judgement a regex cannot approximate.
* **`LLMSafetyProvider`** -- any `LLMClient` as a classifier across all four
  categories in one call, the same pattern `SemanticInjectionScanner` uses
  for injection. Point it at a cheap model, ideally not the agent's own
  provider account.

**A third disposition.** `ContentSafetyScanner` reuses the `Finding`
machinery from §2 (`escalate_to`, `block`), plus one new one:
`fallback` -- the call does not run, and the model gets back a
`SAFETY_FALLBACK`-coded error explaining what was flagged and what to do
differently, **without** consulting the approver at all. This is
deliberately distinct from escalation: `sensitive_operations` in
`GovernancePolicy` says "a human must see this"; `fallback` says "this is
clearly out of policy, redirect it, don't page anyone." Use whichever
matches the actual cost of a false positive for that category in your
deployment. The defaults (`DEFAULT_CATEGORY_POLICIES`) escalate harmful
intent and unsafe behaviour to `DANGER`, fall back on plain policy
violations, and escalate bias at `WARNING` -- loud in the trace, not a hard
stop, and never a built-in `block`: that call is a policy decision for the
deployment to make explicitly, not a default this library assumes for you.
An unrecognised category (a provider returning something the four built-ins
don't cover) still escalates by default -- it never degrades to "ignored."

**Configurable, pluggable, optional -- literally.** `content_safety_scanners`
defaults to an empty list: nothing changes for a deployment that doesn't set
it. It's a plain `GuardrailConfig` field, so it composes with everything
else in this document -- `GovernancePolicy.apply()` carries it through
untouched, and it's screened on both the arguments path (pre-execution) and
the results path (post-execution), for free, because it's just another
`Scanner`.

See `security/content_safety.py` for the full API, and
[`tests/test_content_safety.py`](../tests/test_content_safety.py) for the
category → disposition matrix exercised end to end.

## 3. Human-in-the-loop approval

A `DANGER`-tier call does not run without an `ApprovalDecision`. `Approver`
implementations ship for the shapes an approval actually takes in practice:

* `AllowTierApprover(ceiling)` -- non-interactive; approves anything at or
  below `ceiling`. The sane unattended default.
* `TerminalApprover()` -- blocking `y/N` prompt, printing the tool, the
  arguments, and *why the gateway flagged it* (the scanner findings), because
  a human asked to approve `execute_code` blind will approve it.
* `WebhookApprover(url)` -- posts to an HTTP endpoint (Slack, PagerDuty,
  ServiceNow, an internal review queue); supports synchronous responses or a
  poll-until-decided handle. A timeout **denies**. An unreachable approver
  **denies**. Neither is treated as permission.
- `DenyAllApprover()` -- the fail-closed default when no approver is
  configured at all.

Every approval request carries the goal, the tool, the arguments, the
computed tier, and the scanner findings that led to it (`ApprovalRequest`).
Every decision -- approved or not, and by whom (`ApprovalDecision.by`) -- is
recorded in `Gateway.decisions` and emitted as `APPROVAL_REQUESTED` /
`APPROVAL_DECIDED` events, so "who approved this and on what basis" is always
answerable from the trace, not from someone's memory of a Slack thread.

`GovernancePolicy.approval_threshold` is the deployment-level shorthand for
the common case (`AllowTierApprover(threshold)`); pass a real `Approver` via
`GovernancePolicy.approver` (or `GuardrailConfig.approver`) when unattended
approval isn't the right shape for your risk posture.

## 4. Structured run traces

Every iteration already produces a structured record -- this predates and is
untouched by the governance layer, and is what makes the ANALYZE → ACT →
EXECUTE → OBSERVE separation legible after the fact:

```python
result = agent.run("...")
for it in result.state.iterations:
    it.plan          # {"goal_restatement", "steps", "next_action": {"tool", "rationale", "success_criteria", ...}}
    it.tool_calls     # [ToolCallRecord(tool, arguments, rationale, ok, result_preview, error_code, ...)]
    it.evaluation     # {"outcome", "evidence", "goal_status", "next_step", ...}
    it.violations     # contract violations raised and corrected within this iteration
```

The **plan is not inferred** -- it's the model's own structured commitment,
made *before* it could see a tool result, and the ACT-phase contract forbids
calling any tool the plan didn't name. So every `ToolCallRecord.rationale`
is a real, pre-registered justification, not a post-hoc explanation. The
**evidence is not inferred either** -- `evaluation.evidence` is the model's
own citation for why it believes the step succeeded, checked against a
success criterion it wrote down one phase earlier.

## 4a. The decision ledger: tamper-evident and exportable

`result.state.iterations` (§4) is *readable*. It is not *tamper-evident* --
nothing stops a later process from editing `SessionState`'s serialized JSON
and nobody would know. The **decision ledger** is the stricter instrument:
one immutable `DecisionRecord` per completed iteration -- plan, rationale,
selected tool, tool-call outcomes, every safety check `Gateway` performed,
the evaluation's evidence -- plus one guaranteed record at the end of every
run carrying its terminal outcome, however the run ended. Each record is
chained to the one before it by a SHA-256 hash over its own content plus the
previous record's hash:

```python
from governed import AgentConfig, DecisionLedgerConfig, JSONLDecisionLedger

agent = Agent(AgentConfig(
    llm=...,
    decision_ledger=DecisionLedgerConfig(
        enabled=True,                                    # off by default
        store=JSONLDecisionLedger("./ledgers/run.jsonl"), # or InMemoryDecisionLedger(), or your own
    ),
))
result = agent.run("...")

records = agent.decision_ledger.store.read(result.state.run_id)
verify_chain(records)          # raises TamperDetected on the first altered/reordered/deleted entry
```

**Edit, delete, or reorder any entry and every hash from that point forward
stops matching what `verify_chain` recomputes.** This is *detection*, the
same honest standard §7 holds every other guardrail to -- it is not a lock.
Nothing here stops a process with write access to the underlying store from
regenerating the *entire* chain, consistently, from scratch. What it
guarantees is that the realistic tampering scenario -- touching up *one*
inconvenient entry after the fact and leaving the rest alone -- is always
detectable, because that single edit invalidates every hash after it.
`JSONLDecisionLedger.verify(run_id)` does exactly this against a file on
disk; see `tests/test_decision_ledger.py::test_jsonl_ledger_detects_tampering_written_to_disk`
for it catching a hand-edited line.

**Configurable, and independent of `trace_path`.** `enabled=False` (the
default) means zero behavioural change -- nothing is written, nothing is
computed. `store` is pluggable (`DecisionLedgerStore`: two methods,
`append`/`read`) -- `JSONLDecisionLedger` is the reference "structured
append-only registry," `InMemoryDecisionLedger` exists for tests, and your
own implementation can back onto whatever your organisation already treats
as an audit store. Resume-safe: `Agent.resume()` continues the same hash
chain rather than starting a second one pretending to be independent.

**Exported or streamed to external observability systems.**
`DecisionLedgerConfig.sinks` fans every record out live, in addition to
`store` persisting it -- the same `Subscriber`-shaped extension point the
event trace already uses (§5), typed to `DecisionRecord` instead of `Event`:

```python
from governed import DecisionLedgerConfig, HttpDecisionLedgerSink, OTelDecisionLedgerSink

DecisionLedgerConfig(
    enabled=True,
    sinks=[
        # Splunk HEC / Datadog logs intake / New Relic Log API / Dynatrace
        # ingest API all accept arbitrary structured JSON over HTTP with an
        # auth header -- point this at whichever one your deployment uses.
        HttpDecisionLedgerSink("https://http-inputs-<host>.splunkcloud.com/...",
                                headers={"Authorization": "Splunk <hec-token>"}),
        # OTLP/HTTP, hand-built against the documented wire format -- no
        # opentelemetry-sdk dependency. Point it at a Collector, or directly
        # at any backend with native OTLP ingestion (all four vendors above
        # have one today).
        OTelDecisionLedgerSink("https://otel-collector.internal:4318"),
    ],
)
```

Neither ships a vendor SDK -- both are plain HTTP, `transport` is injectable
for testing without a network, and this project's one-runtime-dependency
promise (`pydantic`, full stop) is unaffected. Need a tighter integration
against a vendor's own client library instead? Implement
`DecisionLedgerSink` (one method: `__call__(self, record) -> None`) --
the same "implement one method" contract this project applies to
`LLMClient`, `Tool`, `Scanner`, and `SafetyProvider`. A sink that raises
never breaks the run (same rule as `EventBus`); it just doesn't get that
record. Already have history you want backfilled into a system configured
after the fact? `export_decisions(store.read(run_id), sink)` replays it.

The ledger and `AuditReport` (§5) are complementary, not competing: the
ledger is the raw, verifiable record stream built for export; `AuditReport`
is a human/compliance-readable *summary*, built straight from
`SessionState`/`Gateway` regardless of whether a decision ledger is even
enabled.

## 5. Observability and audit logging

Configurable per run, all optional, all composable:

```python
AgentConfig(
    ...,
    trace_path="./traces/run.jsonl",  # JSONLSink: append-only, canonical record
    console=True,                     # ConsoleSink: terse running commentary
    verbose=False,
    subscribers=[my_datadog_subscriber, LoggingSink()],  # anything: Event -> None
)
```

* `read_trace(path)` / `trace_to_markdown(path)` turn the JSONL back into a
  reviewable narrative: plan, action, why, result, verdict, per iteration.
* `TelemetryCollector` aggregates LLM call stats, tool call stats, safety
  stats (approvals granted/denied, findings by rule), and HITL idle time
  across a run or a fleet of runs.
* `build_audit_report(agent, result)` -- new in this release -- assembles a
  single **compliance-facing** object from data the framework already
  collected: the plan/action/evaluation history, the guardrail decision log,
  the governance policy that was in effect, cost by phase, and the final
  answer with its confidence and evidence. `.to_markdown()` for a human
  reviewer, `.to_json()` for a downstream system:

  ```python
  from governed import build_audit_report

  report = build_audit_report(agent, result)
  print(report.to_markdown())
  ```

  This is a *view*, not a second bookkeeping system -- it reads
  `SessionState.iterations`, `Gateway.decisions`, and `CostLedger`, and
  reassembles them. Nothing about how a run executes changes because you
  asked for a report afterwards.

Nothing in the core imports a logging framework; every sink is a plain
`Event -> None` callable, so wiring this into your existing observability
stack (Datadog, Prometheus, CloudWatch, Splunk, your own audit database) is
writing one function, not adopting a new one.

## 6. Model/provider governance

Complementary restriction on the model side, using the config-driven LLM
selection described in the README's ["Configuring the LLM by
config"](../README.md#configuring-the-llm-by-config):

```python
from governed import Agent, AgentConfig, LLMConfig, ProviderPolicy

agent = Agent(AgentConfig(
    llm=LLMConfig(provider="anthropic", model="claude-sonnet-5", api_key=...),
    provider_policy=ProviderPolicy(
        allowed_providers=frozenset({"anthropic"}),
        allowed_models={"anthropic": frozenset({"claude-sonnet-5", "claude-haiku-4-5"})},
    ),
))
```

`ProviderPolicy` is checked by `resolve_llm` *before* an adapter is
constructed. Get the provider or the model wrong and `AgentConfig(...)`
raises `ProviderPolicyViolation` at construction -- the same fail-fast
contract as `GovernancePolicy.allowed_tools`.

**Scope, stated plainly, because a governance feature that overstates its own
coverage is worse than none:** this only intercepts the config-driven path.
An `LLMClient` constructed directly (`llm=AnthropicClient(...)`) bypasses it
-- there is no reliable way to recover "which provider and model is this
opaque object" from an arbitrary client instance. **Deployments that need
provider governance actually enforced should standardise on `LLMConfig` as
their integration contract**, and treat direct client construction as a
development/testing convenience, not a production path. This is the same
honesty this project applies to its scanners (see next section): stating a
boundary's real scope is what makes the boundary trustworthy.

## 7. Safety boundaries: what's guaranteed, what's detection

Not everything named "guardrail" is equally strong. Being precise about this
distinction is itself a Responsible AI requirement -- a false sense of
security is a safety defect.

**Structurally guaranteed, because there is no code path around them:**

* No tool call reaches a tool without passing through `Gateway.screen_call`
  -- the gateway is installed by subclassing `ToolRegistry`, the framework's
  single dispatch chokepoint.
* A `DANGER`-tier call cannot execute without an explicit
  `ApprovalDecision`. Denial returns a non-retryable error; the side effect
  never happens.
* A call whose `ContentSafetyScanner` finding sets `fallback` never reaches
  the tool either -- `GuardedRegistry.invoke` checks `CallDecision.blocked`
  before dispatch, for a hard block and a fallback redirect alike. The
  difference between the two is which error code and remediation the model
  sees, not whether the call ran.
* A disallowed tool named in `GovernancePolicy.allowed_tools` never even
  reaches the registry -- `Agent.__init__` raises before construction
  completes.
* An out-of-policy provider/model named in `ProviderPolicy` never gets an
  adapter constructed for it.
* The agent cannot rewrite its own configuration. `SelfModificationGuard`
  holds the absolute, resolved paths of the skills directory, the session
  store, the trace file, and (when enabled) the decision ledger's own file,
  and refuses any write reaching them. The system prompt is rebuilt fresh
  from `AgentConfig` every turn; no tool argument can reach it.
* Reserved scratchpad keys (`_`-prefixed, e.g. the cost ledger's checkpoint)
  are not writable by the model.
* Each `DecisionRecord`'s `entry_hash`/`prev_hash` is computed once, inside
  `DecisionLedger.record`, from a deep copy of its inputs -- there is no
  tool, and no later code path in `Agent`, that can construct or edit one
  directly. (This guarantees the record is written correctly; it does not
  guarantee the *store* it's written to can't be edited afterwards -- see
  the next list, and §4a.)
* The workspace sandbox (`ToolContext.resolve`) rejects absolute paths, `..`
  traversal, and symlinks resolving outside the root.

**Not guaranteed -- detection, which raises the cost of an attack without
making it impossible:**

* `InjectionScanner` / `SecretExfiltrationScanner` / `DestructiveCommandScanner`
  / `PIIScanner` are regex sweeps. They catch the attacks (or, for
  `PIIScanner`, the structured PII formats) in their pattern list and the
  lazy variants of those. Base64, homoglyphs, a novel phrasing, PII embedded
  in free text, or an instruction split across two files can pass them.
  `PIIScanner` additionally never blocks by design -- see its own
  docstring for why that's a deliberate choice, not an oversight.
* `SemanticInjectionScanner` asks a language model whether a span of text is
  an injection attempt. That classifier reads attacker-controlled text and
  can, in principle, be argued out of its verdict by the same class of attack
  it is looking for.
* `ContentSafetyScanner` is only as good as the `SafetyProvider` behind it.
  `KeywordSafetyProvider` has the exact limitations of any regex list, and
  says so in its own docstring rather than pretending otherwise.
  `LLMSafetyProvider` (or any classifier-backed provider you plug in) has the
  same "reads attacker-controlled text" limitation as `SemanticInjectionScanner`.
  Wiring in a real external moderation API changes the detection quality; it
  does not change this being detection rather than a boundary.
* `execute_code`'s default `SubprocessBackend` (resource limits, stripped
  environment, its own process group) stops an agent's *mistakes*. It is not
  a jail against an adversary and does not block network egress. For
  untrusted-input deployments: swap in
  `CodeExecutionTool(backend=DockerCodeExecutionBackend())` for real
  namespace/cgroup isolation (no network, read-only root filesystem,
  memory/CPU/process caps -- see the README's Safety section), drop the
  tool (`default_tools(include_code_execution=False)`), or run the whole
  agent inside a locked-down gVisor/Firecracker sandbox and treat either
  backend's in-process limits as a second layer, not the only one.
* The decision ledger's hash chain is *tamper-evident*, not *tamper-proof*.
  Given write access to the store (the JSONL file, or a custom
  `DecisionLedgerStore`'s backing system) and enough patience, someone could
  regenerate every hash in a chain consistently, end to end, and
  `verify_chain` would pass. What it actually defends against -- and does so
  completely -- is the realistic case: editing one record without redoing
  the whole chain after it. Real tamper-*proofing* means the store itself
  must be genuinely append-only and access-controlled (write-once storage, a
  remote log service, a separate system the agent's own process cannot
  reach) -- a deployment concern this library deliberately leaves to you,
  the same way it leaves TLS and secret storage to you.

Treat the scanners, the semantic classifier, and the ledger's hash chain as
a smoke alarm. Treat the sandbox, the approval gate,
`GovernancePolicy.allowed_tools`, and `ProviderPolicy` as the fire door. One further defence is emergent rather
than implemented: because the ANALYZE contract forces the model to state, in
a structured plan, which tool it is about to call and why *before* it may
call anything, an injected instruction that redirects the agent shows up in
the plan's `rationale`, in the trace, and in front of the human at the
approval prompt. It does not make the agent safe. It makes the agent's
compromise legible, which is the next best thing -- and it is exactly what
[structured run traces](#4-structured-run-traces) are for.

## 8. Override behaviour: who can change what

Two very different kinds of "override," kept structurally separate:

**Operator-time overrides -- what a human configuring a deployment can do,**
in `AgentConfig`, before a run starts:

* Set `GovernancePolicy`/`ProviderPolicy` to add restrictions beyond the
  framework's defaults (allowlists, sensitive operations, provider/model
  pins).
* Set `GuardrailConfig` directly for full control over `RiskPolicy`,
  scanners, and the approver, when the `GovernancePolicy` shorthand isn't
  expressive enough.
* Set `GuardrailConfig.content_safety_scanners` to decide, per category
  (`CategoryPolicy`), whether harmful intent, policy violations, bias, or
  unsafe behaviour escalate to a human, redirect to a safer fallback, or
  block outright -- and to plug in whichever `SafetyProvider` (internal
  policy engine, external moderation API, the shipped reference providers)
  matches the deployment's actual risk.
* Use `RiskPolicy.downgrade` to explicitly lower a tool's *default* tier
  (e.g. "I know `http_get` is NETWORK-classed, it only reads our own status
  page, stop asking me") -- always an explicit, named, reviewable choice in
  code, never something the running agent can do to itself.
* Choose an `Approver` shape appropriate to the deployment: fully unattended
  (`AllowTierApprover`), interactive (`TerminalApprover`), or routed to a
  real approval system (`WebhookApprover`).
* Set `CircuitBreakerConfig.max_usd` (and the repetition/stall detectors) as
  a hard ceiling independent of any of the above -- money, loops, and risk
  tier are three different questions, and this project treats them as three
  different knobs on purpose.
* Turn the decision ledger on or off (`DecisionLedgerConfig.enabled`),
  choose its storage backend (`store`), and choose which external systems it
  streams to (`sinks`) -- entirely independent of `trace_path`, guardrails,
  and governance. A deployment can run fully governed with the ledger off,
  or turn it on later without touching anything else.

Two configuration-time invariants are enforced, loudly, at `AgentConfig`
construction rather than silently: `guardrails`/`governance` supersede
`approval_policy` (setting both is a `ValueError`, not an implicit
precedence rule you have to read the source to discover), and a disallowed
tool or provider raises immediately rather than being dropped without
comment.

**Run-in-progress overrides -- what a human watching a live run can do,**
after it has started and before it would otherwise finish:

* `Agent.cancel(reason)` -- callable from another thread, since `run()`/
  `resume()` block. Cooperative, not preemptive: it takes effect at the next
  checkpoint (top of the loop, or right after a tool call finishes), not
  mid-request. This is the operator's kill switch for a run that's visibly
  going wrong -- looping on a bad approach, burning budget on something no
  longer worth it, or simply no longer needed -- independent of and faster
  than waiting for a budget ceiling or the circuit breaker to catch it.
  `RunResult.status` comes back `"cancelled"`, with a normal trace and
  decision-ledger record, not a dropped connection with no audit trail. The
  CLI wires this to Ctrl-C automatically.

**Run-time overrides -- what the running agent (i.e., the model) can never
do,** by construction, regardless of what it plans, argues, or is told by
data it reads:

* It cannot approve its own `DANGER`-tier calls. Approval requires a real
  `ApprovalDecision` from the configured `Approver`; there is no tool whose
  arguments reach the approver's verdict.
* It cannot talk its way out of a `fallback` redirect after the fact -- the
  decision is made by `ContentSafetyScanner`/`CategoryPolicy` from the
  arguments it already committed to in ACT, and that specific call does not
  run. It can, in the next iteration, propose a genuinely different call --
  which is the point of the remediation text, not a loophole -- and that new
  call is screened fresh, from scratch, on its own arguments.
* It cannot edit `GovernancePolicy`, `ProviderPolicy`, `GuardrailConfig`, or
  any other part of `AgentConfig`. These live in the host process and are
  read fresh each turn; no tool takes an argument that reaches them.
* It cannot widen its own tool allowlist, weaken its own risk tiers, or
  swap its own approver -- all of that is `GovernancePolicy`/`RiskPolicy`
  state the model has no path to.
* It cannot silence the trace. `TraceLogger` is constructed by `Agent`, not
  exposed as a tool; the model cannot stop an event from being emitted.
* It cannot silence the decision ledger either, or skip a record, or write
  one out of order -- `DecisionLedger.record` is called from `_drive`
  itself, not from anything a tool touches, once per completed iteration and
  once, guaranteed, at run end. It cannot edit a record already written:
  `DecisionRecord` is frozen, and even if it could mutate the Python object,
  the hash was already computed and (if `sinks` are configured) already
  streamed out.
* It cannot reset the cost ledger or edit reserved scratchpad state --
  `SelfModificationGuard` refuses `_`-prefixed scratchpad writes.
* It cannot cancel its own run, or stop `Agent.cancel()` from taking effect
  once called. `cancel()` is a method on the host-process `Agent` object,
  not a tool; nothing in the model's context can reach it.

If a deployment ever finds the model apparently "overriding" a policy, that
is a bug report, not an expected escape hatch -- file it against the specific
guarantee in [§7](#7-safety-boundaries-whats-guaranteed-whats-detection) that
should have held.

## Tests that verify all of this

* [`tests/test_governance.py`](../tests/test_governance.py) -- `GovernancePolicy`
  in isolation (allowlist enforcement, tier escalation, approver composition)
  and end-to-end through a real `Agent` run: a sensitive operation denied
  with no human present, the same operation approved once one is, and
  `build_audit_report` verified against both.
* [`tests/test_llm_factory.py`](../tests/test_llm_factory.py) -- `ProviderPolicy`
  enforcement (allowed providers, allowed models per provider, the documented
  scope boundary around directly-constructed clients) and config-only
  provider swaps.
* [`tests/test_guardrails.py`](../tests/test_guardrails.py) -- the underlying
  `RiskPolicy`/`Gateway`/scanner/approver mechanics that both policy layers
  build on.
* [`tests/test_content_safety.py`](../tests/test_content_safety.py) --
  `KeywordSafetyProvider`/`LLMSafetyProvider` in isolation, the full
  category → disposition matrix on `ContentSafetyScanner` (escalate,
  fallback, block, and the unrecognised-category default), a spy approver
  proving `fallback` never consults a human, results screening, the audit
  trail, and an end-to-end `Agent` run redirected to a safer path.
* [`tests/test_decision_ledger.py`](../tests/test_decision_ledger.py) -- the
  hash chain (genesis linkage, chaining, deep-copy-on-write, resume
  continuity), `verify_chain` actually catching altered/deleted/reordered
  records, `JSONLDecisionLedger` catching a hand-edited line on disk, both
  streaming sinks against an injected transport, a failing sink not breaking
  a run, `export_decisions` replaying history, and an end-to-end `Agent` run
  producing one record per iteration plus a guaranteed run-end record --
  including confirming the ledger's own file is protected from the agent's
  own writes.

All of it runs offline, against `ScriptedClient` and injected fake SDK
clients -- no network, no API key, and it belongs in every CI pipeline that
touches this code (`pytest && ruff check . && mypy src`).

## Responsible AI readiness checklist

Before calling a deployment "governed":

- [ ] `GovernancePolicy.allowed_tools` set to an explicit list, not `None`,
      for any deployment that shouldn't get new tools by accident.
- [ ] `GovernancePolicy.sensitive_operations` names every operation this
      deployment's stakeholders would want a human to see before it happens.
- [ ] An `Approver` appropriate to the deployment's attendance model is wired
      in -- not the fail-closed `DenyAllApprover` default, unless that's
      actually the intended posture.
- [ ] `ProviderPolicy` set if the deployment must stay on approved
      providers/models, and the integration uses `LLMConfig`
      (not direct client construction) so that policy is actually enforced.
- [ ] `GuardrailConfig.content_safety_scanners` set if this deployment
      generates or acts on open-ended content (drafts, emails, code, anything
      a person didn't write) -- and a real `SafetyProvider` behind it, not
      just the keyword reference implementation, if bias detection matters
      here.
- [ ] `CategoryPolicy` dispositions reviewed per category, not left at the
      library defaults without reading them -- `fallback` skips human review
      by design, so decide deliberately which categories that's right for.
- [ ] `FeatureToggleConfig(pii_detection=True)` (or `PIIScanner()` in
      `GuardrailConfig.extra_scanners`) set if this deployment's tool
      arguments or results can contain customer PII -- and understood as
      visibility, not prevention: it escalates to a human, it does not block.
- [ ] If `execute_code` runs against untrusted input, `CodeExecutionTool`
      is using `DockerCodeExecutionBackend` (or an equivalent
      container/gVisor/Firecracker isolation), not the default
      `SubprocessBackend` -- see §7.
- [ ] `CircuitBreakerConfig.max_usd` set. This is the one guardrail with no
      good default, because only you know what the task is worth.
- [ ] A live `Agent` reference is reachable from wherever this deployment's
      operators actually watch a run, so `agent.cancel()` is a real button
      someone can press, not a capability that only exists in the SDK.
      (Automatic for the CLI, which wires it to Ctrl-C. A background worker
      needs its own path -- an admin endpoint, a queue message, a signal
      handler -- to the same `Agent` instance a run is executing on.)
- [ ] `trace_path` set to a durable, writable location, and treated as an
      audit artifact (retained, not scratch).
- [ ] `DecisionLedgerConfig.enabled=True` with a durable `store` if this
      deployment needs to prove, after the fact, that its own audit record
      wasn't quietly edited -- `trace_path` alone doesn't give you that.
- [ ] `DecisionLedgerConfig.sinks` wired to wherever this organisation
      actually reviews agent activity, not left to accumulate in a local
      file nobody looks at until there's an incident.
- [ ] The store backing the decision ledger is genuinely append-only and
      access-controlled at the infrastructure level if the deployment's
      threat model requires tamper-*proof*, not just tamper-*evident* -- see
      §7's caveat on the hash chain before assuming otherwise.
- [ ] Someone has read [§7](#7-safety-boundaries-whats-guaranteed-whats-detection)
      and knows which of this deployment's protections are structural
      boundaries and which are detection.
