"""The Responsible AI execution layer: screens what a tool call is about to
*do* or *write* -- harmful intent, policy violations, bias, unsafe behaviour
-- before it reaches ``EXECUTE``. Nothing here is a new chokepoint: it plugs
into the same ``Scanner`` protocol and the same ``Gateway.screen_call`` path
that ``InjectionScanner`` and friends already use, so "screen tool arguments
before they run" is structural, not something this module has to reinvent.

Two things are new:

1. **A provider-agnostic integration seam.** ``SafetyProvider`` is the
   interface an external moderation API (a content-safety endpoint, an LLM
   asked to classify, an internal OPA-style policy engine) implements.
   ``ContentSafetyScanner`` adapts any ``SafetyProvider`` into a ``Scanner``,
   so plugging in your organisation's safety stack is "implement one method,"
   the same contract this project already applies to ``LLMClient`` and
   ``Tool``.
2. **A third disposition, alongside escalate-to-a-human and hard-block.**
   ``CategoryPolicy(disposition="fallback")`` means: don't run the call as
   written, don't necessarily interrupt a human either -- redirect
   deterministically to a safer path (a corrective error the model reads and
   must act on differently), which is the right answer for content that is
   clearly out of policy but not the kind of thing worth paging someone
   about. ``"escalate"`` still means what it means everywhere else in this
   codebase: raise the risk tier and let ``RiskPolicy``/``Approver`` decide.
   ``"block"`` still means: refuse outright, regardless of who would approve
   it.

Ship two reference ``SafetyProvider``s, deliberately modest, in the same
spirit as ``DestructiveCommandScanner`` and ``SemanticInjectionScanner``:
``KeywordSafetyProvider`` (regex, zero dependencies, catches the unambiguous
cases) and ``LLMSafetyProvider`` (any ``LLMClient`` as a classifier, for
everything a keyword list cannot see -- bias chief among them; see its
docstring for why keyword-matching bias is not attempted here at all). Read
``guardrails.py``'s module docstring before trusting either: they are
detection, not a security boundary.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..llm.base import LLMClient, Message
from .guardrails import Finding, RiskTier, Severity

__all__ = [
    "BIAS",
    "HARMFUL_INTENT",
    "POLICY_VIOLATION",
    "UNSAFE_BEHAVIOR",
    "CategoryPolicy",
    "ContentSafetyScanner",
    "KeywordSafetyProvider",
    "LLMSafetyProvider",
    "SafetyProvider",
    "SafetyVerdict",
]

#: Well-known categories. A ``SafetyProvider`` is free to return others --
#: ``ContentSafetyScanner`` falls back to ``CategoryPolicy()``'s default
#: (escalate to DANGER) for any category it doesn't have a policy for, so an
#: unrecognised category degrades to "ask a human," never to "ignored."
HARMFUL_INTENT = "harmful_intent"
POLICY_VIOLATION = "policy_violation"
BIAS = "bias"
UNSAFE_BEHAVIOR = "unsafe_behavior"


@dataclass
class SafetyVerdict:
    """What a ``SafetyProvider`` reports about one span of text."""

    flagged: bool
    categories: list[str] = field(default_factory=list)
    severity: Severity = Severity.WARN
    reason: str = ""
    confidence: float = 1.0
    #: Name of the provider that produced this verdict, for the audit trail.
    provider: str = ""
    #: Raw provider payload, kept for debugging. Never read by the scanner.
    raw: Any = None


class SafetyProvider(Protocol):
    """Implement ``evaluate``. This is the plug point for an external
    moderation API or an internal policy engine -- ``ContentSafetyScanner``
    doesn't care what's on the other side of it.
    """

    name: str

    def evaluate(self, text: str, *, source: str) -> SafetyVerdict: ...


@dataclass
class CategoryPolicy:
    disposition: str = "escalate"  # "escalate" | "fallback" | "block"
    escalate_to: RiskTier = RiskTier.DANGER


#: Sane defaults, overridable per category via ``ContentSafetyScanner(category_policies=...)``.
#: Harmful intent and unsafe behaviour stop for a human. A clear policy
#: violation redirects rather than paging anyone. Bias is real but often
#: contestable and rarely an emergency, so it escalates at WARNING -- loud in
#: the trace, not a hard stop -- unless the provider itself reports CRITICAL
#: severity, which still only escalates (bias is deliberately never a
#: built-in "block" category: that call is a policy decision for the
#: deployment to make explicitly, not a default this library should assume).
DEFAULT_CATEGORY_POLICIES: dict[str, CategoryPolicy] = {
    HARMFUL_INTENT: CategoryPolicy("escalate", RiskTier.DANGER),
    UNSAFE_BEHAVIOR: CategoryPolicy("escalate", RiskTier.DANGER),
    POLICY_VIOLATION: CategoryPolicy("fallback", RiskTier.DANGER),
    BIAS: CategoryPolicy("escalate", RiskTier.WARNING),
}


def _clip(s: str, n: int = 200) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1] + "…"


class ContentSafetyScanner:
    """Adapts a ``SafetyProvider`` into the ``Scanner`` protocol.

    Drop the result into ``GuardrailConfig(content_safety_scanners=[...])``
    (screens both tool arguments and tool results) and it composes with
    everything else in ``Gateway`` for free -- risk-tier escalation, the
    audit trail, the approval gate.

    ``fail_open`` decides what happens when the provider errors: ``True``
    (default) means an unreachable moderation API does not halt the run;
    ``False`` means the call is blocked. Same trade-off, same default, as
    ``SemanticInjectionScanner`` -- see its docstring for the reasoning.
    """

    name = "content_safety"

    def __init__(
        self,
        provider: SafetyProvider,
        *,
        category_policies: dict[str, CategoryPolicy] | None = None,
        fail_open: bool = True,
        max_chars: int = 6_000,
    ) -> None:
        self.provider = provider
        self.category_policies = {**DEFAULT_CATEGORY_POLICIES, **(category_policies or {})}
        self.fail_open = fail_open
        self.max_chars = max_chars

    def scan(self, text: str, source: str) -> list[Finding]:
        if not text.strip():
            return []
        try:
            verdict = self.provider.evaluate(text[: self.max_chars], source=source)
        except Exception as exc:
            return [
                Finding(
                    "SAF000",
                    Severity.INFO if self.fail_open else Severity.CRITICAL,
                    f"Content safety provider {self.provider.name!r} unavailable "
                    f"({type(exc).__name__}). "
                    + ("Failing open." if self.fail_open else "Failing closed."),
                    source=source,
                    block=not self.fail_open,
                )
            ]

        if not verdict.flagged or not verdict.categories:
            return []

        findings: list[Finding] = []
        for category in verdict.categories:
            policy = self.category_policies.get(category, CategoryPolicy())
            findings.append(
                Finding(
                    rule_id=f"SAF:{category}",
                    severity=verdict.severity,
                    message=(
                        f"Flagged by {verdict.provider or self.provider.name} for "
                        f"{category} (confidence={verdict.confidence:.2f}): {verdict.reason}"
                    ),
                    evidence=_clip(text),
                    source=source,
                    escalate_to=(
                        policy.escalate_to if policy.disposition != "block" else None
                    ),
                    block=policy.disposition == "block",
                    fallback=policy.disposition == "fallback",
                )
            )
        return findings


class KeywordSafetyProvider:
    """Regex-based ``SafetyProvider``. Zero dependencies, catches the
    unambiguous cases, misses everything a keyword list structurally cannot
    see -- the same honesty applies here as to ``DestructiveCommandScanner``.

    Deliberately does **not** attempt to detect ``BIAS``. A fixed word list is
    the wrong tool for that job: it produces confident-looking false
    positives and negatives in roughly equal, unhelpful measure, and bias
    review needs judgement a regex cannot approximate. Use
    ``LLMSafetyProvider``, or a real moderation API wrapped as a
    ``SafetyProvider``, for that category.
    """

    name = "keyword"

    PATTERNS: tuple[tuple[str, str, str], ...] = (
        (
            "weapon_synthesis",
            r"\b(step[- ]by[- ]step|detailed?)\b[^.\n]{0,60}"
            r"\b(synthesi[sz]e|build|construct|manufacture)\b[^.\n]{0,40}"
            r"\b(explosive|bomb|nerve agent|bioweapon|chemical weapon)\b",
            HARMFUL_INTENT,
        ),
        (
            "malware_authoring",
            r"\b(write|create|generate)\b[^.\n]{0,40}"
            r"\b(ransomware|keylogger|rootkit|self[- ]propagating (virus|worm))\b",
            HARMFUL_INTENT,
        ),
        (
            "self_harm_encouragement",
            r"\b(how to|ways to|method[s]? (to|for))\b[^.\n]{0,30}"
            r"\b(kill (myself|yourself)|end (my|your) life|commit suicide)\b",
            HARMFUL_INTENT,
        ),
        (
            "credential_harvesting",
            r"\b(phishing (page|kit|email)|steal (passwords|credentials)|"
            r"credential harvest(er|ing))\b",
            UNSAFE_BEHAVIOR,
        ),
    )

    def __init__(self, extra: tuple[tuple[str, str, str], ...] = ()) -> None:
        self._rules = [
            (rid, re.compile(rx, re.IGNORECASE), cat)
            for rid, rx, cat in (*self.PATTERNS, *extra)
        ]

    def evaluate(self, text: str, *, source: str) -> SafetyVerdict:
        hits: list[tuple[str, str]] = []
        for rule_id, rx, category in self._rules:
            if rx.search(text):
                hits.append((rule_id, category))
        if not hits:
            return SafetyVerdict(flagged=False, provider=self.name)
        return SafetyVerdict(
            flagged=True,
            categories=sorted({c for _, c in hits}),
            severity=Severity.CRITICAL,
            reason=f"matched: {', '.join(rid for rid, _ in hits)}",
            confidence=1.0,
            provider=self.name,
        )


_CLASSIFIER_SYSTEM = """\
You are a content-safety classifier for an autonomous agent's proposed \
actions. You will be shown text the agent is about to write, send, or act on \
-- a tool call's arguments, a file it is about to write, an answer it is \
about to submit.

Classify it against exactly these categories:
- "harmful_intent": facilitates violence, weapons, malware, or serious harm.
- "policy_violation": violates ordinary content policy (harassment, illegal \
activity, deceptive/fraudulent content) without necessarily being violent.
- "bias": expresses or amplifies unfair discrimination or stereotyping \
against a protected characteristic (race, gender, religion, disability, etc).
- "unsafe_behavior": the agent taking an action a reasonable operator would \
not want an autonomous system to take unsupervised, distinct from the above.

Answer with one JSON object and nothing else:
{"flagged": true|false, "categories": ["..."], "severity": "info"|"warn"|"critical", \
"confidence": 0.0-1.0, "reason": "<one sentence>"}\
"""

_SEVERITY_MAP = {"info": Severity.INFO, "warn": Severity.WARN, "critical": Severity.CRITICAL}


class LLMSafetyProvider:
    """``SafetyProvider`` backed by any ``LLMClient``, classifying across all
    four categories in one call. Point it at a cheap model, ideally *not*
    the agent's own provider account -- same reasoning as
    ``SemanticInjectionScanner``: a compromised classifier should not be able
    to burn the agent's budget.

    To integrate a real external moderation API instead of an LLM, implement
    ``SafetyProvider`` directly against that API's client -- this class is a
    reference implementation of the protocol, not the only way to satisfy it.
    """

    name = "llm_classifier"

    def __init__(self, llm: LLMClient, *, meter: Any = None) -> None:
        self.llm = llm
        #: Optional hook so the classifier's own token spend lands in a ledger.
        self.meter = meter

    def evaluate(self, text: str, *, source: str) -> SafetyVerdict:
        payload = f"<proposed-action source={source!r}>\n{text}\n</proposed-action>"
        resp = self.llm.complete(
            system=_CLASSIFIER_SYSTEM,
            messages=[Message(role="user", text=payload)],
            tools=None,
            tool_choice="none",
            max_tokens=250,
            temperature=0.0,
        )
        if self.meter:
            self.meter(resp.usage)
        verdict = json.loads(_strip_fences(resp.text))
        return SafetyVerdict(
            flagged=bool(verdict.get("flagged", False)),
            categories=list(verdict.get("categories", [])),
            severity=_SEVERITY_MAP.get(str(verdict.get("severity", "warn")), Severity.WARN),
            reason=str(verdict.get("reason", ""))[:300],
            confidence=float(verdict.get("confidence", 0.0)),
            provider=self.name,
            raw=resp,
        )


def _strip_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", s).strip()
    return s
