# Roadmap: known gaps

A snapshot, not a promise. This is where the last several build sessions
left off — what's genuinely still missing, ranked by how cheap it is to
close and how much it matters. Written so a future session (or a human) can
pick an item and start without re-deriving context from scratch.

As of this writing: 334 tests passing (1 conditionally skipped), ruff/mypy
strict clean, ~90% test coverage (CI gates at 85%). The original 11-item
implementation plan (config-first runtime, provider factory, plugin
interfaces, governance/safety, feature toggles, monitoring pluggability,
pre-execution screening, the decision ledger, docs/examples, tests/CI,
CLI/bootstrap) is done, as is a follow-on pass closing responsible-AI-
framework gaps: `LICENSE`, `SECURITY.md`, `CODE_OF_CONDUCT.md`,
`CONTRIBUTING.md`, CI dependency scanning (`pip-audit` + CycloneDX SBOM +
Dependabot), coverage enforcement, `PIIScanner`, `DockerCodeExecutionBackend`,
and `Agent.cancel()` (a cooperative kill switch, wired to Ctrl-C in the CLI).
See [`RESPONSIBLE_AI.md`](RESPONSIBLE_AI.md) for what's actually guaranteed
vs. detection, and `README.md`/`CONTRIBUTING.md` for how all of the above
works today.

Update this file when you close or discover a gap — stale roadmaps are worse
than none.

## Quick wins (docs-only or well under an hour)

- **State the multi-tenancy/RBAC scope explicitly.** `RESPONSIBLE_AI.md`
  documents every structural boundary the framework *does* provide, but
  never explicitly says the obvious converse: there is no multi-tenancy,
  RBAC, or per-caller identity concept anywhere in the codebase. It's a
  single-process library; access control between callers is entirely the
  embedding application's job. This has been flagged in two prior sessions
  and still isn't written down. Add a short, explicit "out of scope, by
  design" note to `RESPONSIBLE_AI.md` (§8 is the natural home, next to the
  override-behaviour bullets) so an adopter doesn't assume isolation the
  framework doesn't enforce.

- ~~Replace the `your-org` placeholders before real publication.~~ **Done**
  — `pyproject.toml`'s `[project.urls]` and the `git clone` line in
  `README.md`/`CONTRIBUTING.md` now point at `github.com/poorani/governed`.
  `LICENSE`'s copyright line intentionally still reads
  `governed contributors`, 2026 — asked and confirmed: kept generic rather
  than naming an individual, to be revisited if/when others contribute.

- **Gemini has no entry in the pricing table.** `PRICING` in
  `memory/optimizer.py` covers Anthropic and OpenAI models but not one
  `gemini-*` model — `resolve_pricing("gemini-2.5-flash")` (or any other
  Gemini model) returns `None`, so every Gemini run's `cost_usd` reads
  `$0.0000` and the cost circuit breaker (`Budget.max_usd`) cannot protect
  it, no matter how much is actually spent. This isn't silent (the
  once-per-model `COST_WARNING` event fires, per `agent.py:331-337`, and
  the audit report surfaces the same `$0.0000`), but it means anyone
  running Gemini through `governed` today needs
  `CostConfig(pricing_overrides={"gemini-...": ModelPricing(...)})` just to
  get real numbers — not documented as a required step anywhere. Cheap fix:
  add current Gemini rates to `PRICING`, same shape as the existing
  Anthropic/OpenAI entries, and note the `PRICING_AS_OF` check date.

- **Wire SIGTERM the way Ctrl-C (SIGINT) already is.** `cli.py`'s
  `_install_cancel_on_sigint` only handles `SIGINT`. A process manager
  (systemd, Kubernetes, most container orchestrators) sends `SIGTERM` on
  shutdown, not `SIGINT` — right now that still hits Python's default
  handler and exits immediately, skipping the same graceful
  `Agent.cancel()` finalization Ctrl-C now gets. Closely related to the
  cancellation work already done; should be a small addition to the same
  function (`signal.signal(signal.SIGTERM, handler)` alongside `SIGINT`),
  plus a test mirroring `test_sigint_handler_cancels_then_restores_the_default`
  in `tests/test_cli.py`.

## Structural gaps

- **`Agent.run()`/`.resume()` are not reentrant.** Discovered and
  documented (not fixed) while building `Agent.cancel()`: two threads
  calling `run()` on the *same* `Agent` instance concurrently will corrupt
  shared state (`_trace`, `decision_ledger`, `_pending_results`, ...) with
  no error — it just silently produces wrong results. `cancel()` is the
  only method actually safe to call from a second thread while a run is in
  flight. The cheap fix: a lock in `_drive` that raises a clear
  `RuntimeError` if a second concurrent call is attempted, so the failure
  mode is loud instead of silent corruption. A deeper fix (making all
  per-run state live on a call-scoped object instead of `self`) would
  actually *allow* concurrent runs on one `Agent`, but is a real refactor —
  start with the loud-failure version unless concurrent runs on one
  instance turns out to be something people actually want.

- **No async-native API.** Everything is synchronous;
  `Agent.run()`/`.resume()` block the calling thread. An `asyncio`
  application has to run them via `asyncio.to_thread`/`run_in_executor`,
  which does compose correctly with `cancel()` (verified — it's
  thread-safe), but there's no first-class `async def arun()`. Doing this
  properly needs either an async variant of `LLMClient`/`complete()` (the
  Anthropic/OpenAI/Gemini SDKs all have async clients available) or an
  adapter layer; a real design decision, not a quick patch. Worth scoping
  before starting.

- **Only three first-class LLM providers; no generic/protocol-level
  adapter.** Today, supporting a new model means one of two things: it's
  Anthropic, OpenAI (or an OpenAI-compatible endpoint via `base_url` — vLLM,
  Ollama, Together, LM Studio all work today, since they speak the OpenAI
  wire format), or Gemini — the only three shipped `LLMClient`
  implementations — or someone hand-writes a fourth adapter class and calls
  `register_provider` (one `complete()` method; see [Bringing your own
  LLM](../README.md#bringing-your-own-llm)). There's no protocol-level
  adapter that could talk to an arbitrary model given just a base URL and a
  wire-format hint, the way, say, a generic OpenAI-compatible client
  already halfway does for self-hosted models. Worth scoping: is the real
  gap "more vendor SDKs bundled in" (cheap, mechanical, one file each) or
  "a declarative schema for describing a new provider's request/response
  shape without writing Python" (a real design problem)? Flagged here
  rather than fixed because that scoping question hasn't been answered.

- **No rate limiting or quota system.** Nothing in the framework bounds
  concurrent runs, per-tenant spend, or request rate — `Budget` and
  `CircuitBreakerConfig` bound *one run's* resources, not a fleet's. Was
  flagged in the very first gap analysis of this framework and never
  revisited. Probably belongs at the deployment layer (a queue, a
  semaphore in front of `Agent` construction) rather than inside
  `governed` itself, but that judgment call hasn't actually been made —
  worth a deliberate "yes, deployment's job, here's the recommended
  pattern" note in `RESPONSIBLE_AI.md` even if the framework itself
  shouldn't own it.

## Test coverage worth a closer look

The 85% CI floor is a floor, not a signal that everything above it is
adequately tested. Two genuinely undertested modules were closed out this
session (`llm/gemini_client.py` 0% → 94%, `tools/data_analysis.py` 39% →
100%) after being flagged from a `--cov-report=term-missing` scan. That scan
surfaced others that weren't chased down — not because they're fine, but
because closing them wasn't this session's task:

- `src/governed/memory/store.py` — 63%. This backs session persistence and
  `resume()`; worth understanding exactly what's untested before trusting
  it in a resumable-session-heavy deployment.
- `src/governed/tools/filesystem.py` — 65%. This is where the workspace
  sandbox boundary (`ToolContext.resolve`) actually lives — arguably the
  single most safety-critical file in the tool layer, and 65% coverage on
  a sandbox boundary deserves real scrutiny, not a shrug.
- `src/governed/observability/logger.py` — 70%.
- `src/governed/llm/anthropic_client.py` (69%) / `openai_client.py` (79%)
  — untouched this session; only `gemini_client.py` was closed (0% → 94%).
  These two are exercised incidentally through `test_llm_factory.py`'s
  config-resolution smoke tests (one `.complete()` call each via a fake
  SDK double), but nothing tests their message/tool-call translation logic
  the thorough way `test_gemini_client.py` now tests `gemini_client.py`'s
  — multi-turn conversations, tool-result translation, edge cases in
  parsing the response. Same fix shape: a dedicated
  `test_anthropic_client.py`/`test_openai_client.py` per client, following
  `test_gemini_client.py` as the template.

Run `pytest --cov=governed --cov-report=term-missing` to get current
numbers and exact missing line ranges before starting on any of these.

- **`DockerCodeExecutionBackend` has no real-Docker integration test.**
  `tests/test_code_execution.py` verifies command construction, the
  timeout/kill path, and the "CLI not found" path entirely through a fake
  `subprocess.Popen` double — real Docker is never invoked. That's the
  right call for unit tests (fast, no Docker dependency in every dev's
  environment), but there's no test anywhere that a container this backend
  builds actually runs, actually has no network, actually can't write
  outside `/workspace`. GitHub Actions' `ubuntu-latest` runners have Docker
  preinstalled — a real integration test job (skipped gracefully where
  Docker isn't available, same pattern `test_gemini_client.py` uses for
  the optional `google-genai` package) would close this without much
  infra cost.

## Known, accepted limitations — not gaps, don't "fix" these

Listed so they aren't rediscovered and "fixed" into something worse. All are
deliberate design choices, documented at the point they're made:

- `GuardrailConfig.extra_scanners` (including `PIIScanner`) only screens
  tool *arguments*, not *results* — see `Gateway.from_config` in
  `guardrails.py`. This means PII arriving via a tool's output (reading a
  customer record from a file, say) isn't screened, only PII the model
  writes into an argument. This is a real, easy-to-miss gap in what
  `pii_detection=True` actually covers in practice, distinct from
  "PIIScanner is a regex reference implementation" (which *is* documented).
  Worth a closer look on whether this is actually correct-as-designed or
  an oversight in `Gateway.from_config`'s scanner wiring — flagged here
  rather than in "gaps" because it wasn't investigated deeply enough this
  session to be sure which it is.
- `CodeExecutionTool`'s default `SubprocessBackend` is a guardrail, not a
  jail (same-user process, resource limits only) — `DockerCodeExecutionBackend`
  is the answer for untrusted input, and exists now.
- The decision ledger's hash chain is tamper-*evident*, not tamper-*proof*
  — see `RESPONSIBLE_AI.md` §7 for the exact distinction.
- The framework is single-agent by design — no orchestrator, no
  agent-to-agent protocol. See "What this framework is not" in
  `README.md`'s Design notes.
