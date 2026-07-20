# Contributing to governed

Issues and PRs welcome. This document is the practical how-to; the
[README](README.md) is the reference for what the framework does and why it's
built the way it is — read that first if you're touching unfamiliar code,
since most non-obvious design choices here are argued for there, not repeated
in this file.

Found a security vulnerability rather than a bug? See
[SECURITY.md](SECURITY.md) instead of opening a public issue or PR.

Participation in this project — issues, PRs, discussions — is governed by the
[Code of Conduct](CODE_OF_CONDUCT.md).

## Setup

```bash
git clone https://github.com/poorani/governed && cd governed
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
```

`[dev]` pulls in every optional provider extra (`anthropic`, `openai`,
`gemini`, `data`, `yaml`) plus `pytest`, `mypy`, and `ruff` — the same set CI
installs, so a clean local run means a clean CI run. Create and activate the
venv first — installing into the wrong environment (or a globally installed
`pytest` outside any venv) is the most common source of the failure below.

If `pytest` fails immediately with `ImportError while loading conftest
'.../tests/conftest.py'` and `ModuleNotFoundError: No module named
'governed'`, the package itself isn't installed in the environment `pytest`
is running in — `tests/conftest.py` imports `governed.tools.base` directly,
so collection fails before any test runs. Re-run `pip install -e '.[dev]'`
with the same venv active that you're invoking `pytest` from.

## Before opening a PR

```bash
ruff check . && ruff format --check .
mypy src
pytest
```

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs exactly these
four checks on every push and PR, across Python 3.10/3.11/3.12. The whole
suite is offline — no network, no API keys, ever (see "Testing conventions"
below) — so there's no reason a local run and CI should disagree on whether
the tests pass.

CI's test step additionally runs with coverage measurement
(`pytest --cov=governed --cov-report=term-missing`), gated by
`fail_under` in `pyproject.toml`'s `[tool.coverage.report]` — currently 85%,
a floor below the actual total (~90%) on purpose, so it catches a genuinely
untested new module or code path, not everyday variance. Plain `pytest` (no
`--cov`) never measures or gates on coverage, so this doesn't slow down or
complicate the everyday local loop above; run
`pytest --cov=governed --cov-report=term-missing` yourself before a PR that
adds a new file or a large new code path, to see the same numbers CI will.

A separate `dependency-audit` job runs on the same triggers: `pip-audit`
against the full dependency tree (core plus every optional provider), and a
CycloneDX SBOM of the resolved environment, uploaded as a build artifact.
[Dependabot](.github/dependabot.yml) opens a PR weekly for anything outdated
or flagged, for both Python dependencies and the GitHub Actions themselves.
You don't need to run either locally; they don't gate on dependencies your
PR doesn't touch.

## Where new code goes

- **A new tool** belongs in `src/governed/tools/`, ships with tests, and
  its docstring must justify why it isn't better served by `execute_code` —
  see "Adding a tool" in the README for the shape (`name`/`description`/
  `safety`/`Input`/`run`) and the guidelines that earn their keep
  (`description` is written for the model, not your teammates; every error
  gets a `remediation`; output is bounded).
- **A new skill** belongs in `skills/<name>/SKILL.md` and should encode
  judgement, not syntax — "Writing a skill" in the README has the frontmatter
  shape and what makes a skill worth shipping.
- **A new LLM provider, or anything else selectable by name from config**
  (a tool, a skill source, a decision-ledger sink/store, an event sink, a
  state store) almost always belongs behind the existing plugin-registry
  pattern rather than a new core dependency or a new special case in
  `bootstrap.py`. `register_provider`/`register_tool`/`register_skill_source`/
  `register_event_sink`/`register_decision_ledger_sink`/
  `register_decision_ledger_store`/`register_state_store` all follow the
  same shape: a factory function, registered under a name, selectable from
  data from then on. See "Plugin registries" in the README before adding a
  vendor SDK to `pyproject.toml`'s core dependencies — the core has exactly
  one (`pydantic`), and that's deliberate.
- **A new deterministic scanner** (in the shape of `InjectionScanner` /
  `SecretExfiltrationScanner` / `PIIScanner`) belongs in
  `security/guardrails.py`, subclassing `_RegexScanner`. State plainly, in
  the docstring, what it does and doesn't catch — every scanner in this
  codebase is upfront about being detection, not a boundary. If the change
  touches guardrails, governance, the decision ledger, or approval flow,
  update [`docs/RESPONSIBLE_AI.md`](docs/RESPONSIBLE_AI.md) too: it's the
  one place that draws the line between what's structurally guaranteed and
  what's detection, and a new scanner or toggle that isn't reflected there
  is a documentation regression, not just a missing docstring.

## Testing conventions

No test may require network access or a real API key. The two seams that
make this possible, used throughout:

- **`ScriptedClient`** (`governed.llm`) for anything exercising the agent
  loop end-to-end — hand it a scripted sequence of `LLMResponse`s instead of
  a real provider.
- **Injected fake SDK doubles** for anything exercising a specific
  provider's request/response translation — `tests/test_llm_factory.py`
  stands in a `SimpleNamespace` for `anthropic.Anthropic`/`openai.OpenAI`
  via each client's existing `client=` constructor argument, so the test
  verifies governed's own translation logic, not a vendor's wire format.

For anything that shells out to a subprocess or an external CLI (a code
execution backend, an HTTP sink), monkeypatch `subprocess.Popen`/`.run` or
the relevant transport rather than requiring the real binary or endpoint —
`tests/test_code_execution.py`'s `DockerCodeExecutionBackend` tests are the
template: real Docker is never required, but the command construction, the
timeout/kill path, and the "CLI not found" path are all exercised.

Registry mutations in tests go through `monkeypatch.setitem` on the
underlying private dict (e.g. `_TOOL_REGISTRY`, `_SKILL_SOURCE_REGISTRY`),
not the public `register_*()` function directly — this gets automatic
cleanup with no risk of one test's registration leaking into the next.
`tests/test_plugin_registries.py` is the reference.

## Style

`ruff` (lint + format) and `mypy --strict` are the enforced style — there's
no separate style guide beyond what those two catch. Beyond that, match the
codebase's existing voice: docstrings explain *why* a design choice was
made and what its limits are, not just what a function does; comments are
rare and reserved for non-obvious constraints. Read a few existing modules
(`security/guardrails.py` and `llm/factory.py` are good examples) before
writing a large amount of new code.

## License

By contributing, you agree your contribution is licensed under this
project's [Apache License 2.0](LICENSE), per §5 of that license.
