"""Deployment-time governance: allowed tools, sensitive operations, approval
thresholds -- one object an enterprise operator sets once per deployment.

This sits *above* ``guardrails.py``, not instead of it. ``RiskPolicy`` and
``Gateway`` already do per-call risk assessment, scanning, and human approval;
``GovernancePolicy`` is what a platform team hands to every application team
building on this framework, so each one doesn't have to reconstruct that
machinery by hand. It answers three questions at the deployment level, not
the call level:

* Which tools may this agent even have? (``allowed_tools``)
* Which named operations are always sensitive enough to need a person,
  regardless of what the built-in risk tiers would otherwise compute?
  (``sensitive_operations``)
* Below what tier does this deployment run unattended? (``approval_threshold``)

``GovernancePolicy.enforce_allowed_tools`` runs once, at ``Agent``
construction, and fails loudly and immediately if a disallowed tool was
configured. A disallowed tool must never silently vanish from the registry --
a tool that is quietly just not there is much harder to notice in review than
a ``GovernanceViolation`` raised at startup.

``GovernancePolicy.apply`` folds ``sensitive_operations`` and
``approval_threshold`` into a ``GuardrailConfig`` -- building one if the
caller didn't supply one, or layering on top of an explicit one without
discarding anything it already set -- so ``Agent`` always has a single,
already-merged policy to hand to ``Gateway.from_config``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field, replace

from ..tools.base import Tool
from .guardrails import AllowTierApprover, Approver, GuardrailConfig, RiskTier, Scanner

__all__ = ["GovernancePolicy", "GovernanceViolation"]


class GovernanceViolation(Exception):
    """Raised at configuration time when a deployment violates its own policy.

    Distinct from ``ToolErrorCode.POLICY_VIOLATION`` (a runtime error handed
    back to the model when the *gateway* denies one call): this is raised in
    Python during ``Agent.__init__``, before a run ever starts. A
    mis-configured deployment should fail on startup, not three iterations
    into a run that happens to touch the disallowed tool.
    """


@dataclass
class GovernancePolicy:
    """The one object a platform team sets per deployment.

    ``allowed_tools=None`` (the default) means no restriction -- any tool the
    caller registers is fine, matching today's behaviour exactly. Set it to
    lock a deployment to an explicit allowlist; ``submit`` is always
    implicitly permitted, since a run cannot end without it.

    ``sensitive_operations`` names tools or ``"tool:operation"`` pairs
    (matching ``RiskPolicy``'s discriminator convention, e.g.
    ``"file_system:delete"``) that must always require human approval on this
    deployment, regardless of what ``RiskPolicy`` would otherwise compute. It
    can only *raise* a tier, never lower one -- the same invariant
    ``RiskPolicy`` itself enforces, for the same reason: a policy that can
    only escalate is one nobody can quietly weaken by adding an entry.

    ``approval_threshold`` is the risk tier this deployment may run
    unattended up to; anything above it needs a human. It only takes effect
    when neither this policy's ``approver`` nor the effective
    ``GuardrailConfig.approver`` is already set.

    ``apply()`` always returns a ``GuardrailConfig`` with ``enabled=True``: if
    a ``GovernancePolicy`` is in effect, guardrails are on. Don't pass
    ``AgentConfig(guardrails=GuardrailConfig(enabled=False), governance=...)``
    expecting the ``False`` to win -- it won't, deliberately.
    """

    allowed_tools: frozenset[str] | None = None
    sensitive_operations: frozenset[str] = field(default_factory=frozenset)
    approval_threshold: RiskTier = RiskTier.WARNING
    approver: Approver | None = None
    extra_scanners: list[Scanner] = field(default_factory=list)

    # -- tool allowlist -------------------------------------------------

    def enforce_allowed_tools(self, tools: Iterable[Tool]) -> None:
        if self.allowed_tools is None:
            return
        allowed = self.allowed_tools | {"submit"}
        offending = sorted({t.name for t in tools} - allowed)
        if offending:
            raise GovernanceViolation(
                f"Tool(s) {offending} are not permitted by "
                f"GovernancePolicy.allowed_tools ({sorted(allowed)}). Remove them from "
                "AgentConfig(tools=...), or add them to the allowlist if this "
                "deployment is meant to have them."
            )

    # -- guardrail composition -------------------------------------------

    def apply(self, base: GuardrailConfig | None) -> GuardrailConfig:
        """Fold sensitive operations and the approval threshold into a
        ``GuardrailConfig`` -- building one if ``base`` is ``None``, or
        layering on top of an explicit one without discarding anything it
        already set.
        """
        cfg = base if base is not None else GuardrailConfig()

        risk_policy = cfg.risk_policy
        if self.sensitive_operations:
            tool_tiers = dict(risk_policy.tool_tiers)
            operation_tiers = dict(risk_policy.operation_tiers)
            for entry in self.sensitive_operations:
                if ":" in entry:
                    tool, op = entry.split(":", 1)
                    key = (tool, op)
                    operation_tiers[key] = max(
                        operation_tiers.get(key, RiskTier.SAFE), RiskTier.DANGER
                    )
                else:
                    tool_tiers[entry] = max(
                        tool_tiers.get(entry, RiskTier.SAFE), RiskTier.DANGER
                    )
            risk_policy = replace(
                risk_policy, tool_tiers=tool_tiers, operation_tiers=operation_tiers
            )

        approver = cfg.approver or self.approver or AllowTierApprover(self.approval_threshold)
        extra_scanners = [*cfg.extra_scanners, *self.extra_scanners]

        return replace(
            cfg,
            enabled=True,
            risk_policy=risk_policy,
            approver=approver,
            extra_scanners=extra_scanners,
        )
