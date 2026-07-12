# Security policy

This document is about vulnerabilities in governed itself — bugs that let
untrusted input escape a boundary the framework claims to hold. It is not
about an agent doing something unwise within the boundaries it was
configured with; see [`docs/RESPONSIBLE_AI.md`](docs/RESPONSIBLE_AI.md) for
that distinction (§7, "Safety boundaries: what's guaranteed, what's
detection") and for the governance/guardrail knobs that address it.

## Supported versions

governed is pre-1.0 (`0.x`). Until a `1.0` release, only the latest published
version on PyPI receives security fixes — there are no parallel maintenance
branches yet.

| Version | Supported |
| ------- | --------- |
| Latest `0.x` | Yes |
| Older `0.x`  | No |

## Reporting a vulnerability

Please report suspected vulnerabilities privately, not as a public GitHub
issue — the usual reasons apply: a public issue is a live exploit
advisory for every user until a fix ships.

Use **GitHub's private vulnerability reporting** for this repository
(Security tab → "Report a vulnerability"). If that isn't available to you,
open an issue asking for a private channel and we'll follow up off-thread.

Include, as far as you can:

- The affected version (`pip show governed`) and provider/tool combination.
- A minimal reproduction — a scanner bypass, a sandbox escape from
  `FileSystemTool`/`CodeExecutionTool`, an injection that defeats the
  `Gateway`, an approval flow that fires without a human, a decision-ledger
  entry that verifies despite tampering, or similar.
- What you expected the boundary to do, and what it actually did.

## What counts as a vulnerability here

In scope — a bug in governed's own code that breaks a stated boundary:

- Path traversal or symlink escape out of `ToolContext`'s workspace sandbox.
- A way to make `Gateway`/`RiskPolicy` skip approval for a call that should
  have required it, or to falsify a decision-ledger entry without breaking
  the hash chain (`verify_chain` should always catch this — if it doesn't,
  that's the report).
- Secrets (API keys, environment) leaking into a subprocess, a trace, or a
  tool result despite `env_allowlist`/`SecretExfiltrationScanner`.
- `SelfModificationGuard` failing to stop a write to a protected path it was
  configured to protect.

Out of scope — known, documented limitations, not vulnerabilities:

- Regex-based scanners (`InjectionScanner`, `DestructiveCommandScanner`,
  `PIIScanner`, the keyword `SafetyProvider`) missing an attack pattern they
  were never claimed to catch. These are detection, not boundaries — see
  §7 of `RESPONSIBLE_AI.md`. If you have a pattern that should be added,
  open a normal issue or PR instead of a security report.
- `CodeExecutionTool`'s default `SubprocessBackend` running as the same OS
  user with resource limits, not a container. This is stated in its own
  docstring as "a guardrail, not a jail." Use
  `DockerCodeExecutionBackend` (or your own) if same-user execution isn't
  an acceptable boundary for your threat model.
- Vulnerabilities in a third-party LLM provider's API, SDK, or model output
  itself — report those to the provider.
- Denial of service via a deliberately expensive goal against your own
  deployment (that's what `Budget`/`CircuitBreakerConfig` are for).

## Response

We aim to acknowledge reports within 5 business days and to ship a fix or a
documented mitigation before any public disclosure of the details. Credit is
offered, not assumed — tell us if you'd rather stay anonymous.
