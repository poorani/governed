"""Guardrails: the dual-layer gateway every tool call passes through.

Read ``guardrails.py``'s module docstring before relying on any of this. The
short version: the sandbox and the approval gate are boundaries; the scanners
are detection. Do not confuse the two.
"""

from __future__ import annotations

from .content_safety import (
    CategoryPolicy,
    ContentSafetyScanner,
    KeywordSafetyProvider,
    LLMSafetyProvider,
    SafetyProvider,
    SafetyVerdict,
)
from .guardrails import (
    UNTRUSTED_CLOSE,
    UNTRUSTED_OPEN,
    AllowTierApprover,
    ApprovalDecision,
    ApprovalRequest,
    Approver,
    CallDecision,
    DenyAllApprover,
    DestructiveCommandScanner,
    Finding,
    Gateway,
    GuardedRegistry,
    GuardrailConfig,
    InjectionScanner,
    PIIScanner,
    RiskPolicy,
    RiskTier,
    Scanner,
    ScreenedResult,
    SecretExfiltrationScanner,
    SelfModificationGuard,
    SemanticInjectionScanner,
    Severity,
    TerminalApprover,
    WebhookApprover,
)
from .policy import GovernancePolicy, GovernanceViolation

__all__ = [
    "UNTRUSTED_CLOSE",
    "UNTRUSTED_OPEN",
    "AllowTierApprover",
    "ApprovalDecision",
    "ApprovalRequest",
    "Approver",
    "CallDecision",
    "DenyAllApprover",
    "DestructiveCommandScanner",
    "Finding",
    "Gateway",
    "GuardedRegistry",
    "GuardrailConfig",
    "InjectionScanner",
    "PIIScanner",
    "RiskPolicy",
    "RiskTier",
    "Scanner",
    "ScreenedResult",
    "SecretExfiltrationScanner",
    "SelfModificationGuard",
    "SemanticInjectionScanner",
    "Severity",
    "TerminalApprover",
    "WebhookApprover",
    # governance
    "GovernancePolicy",
    "GovernanceViolation",
    # content safety (Responsible AI execution layer)
    # category name constants (HARMFUL_INTENT, POLICY_VIOLATION, BIAS,
    # UNSAFE_BEHAVIOR) live in governed.security.content_safety, not here --
    # avoids a confusing echo with ToolErrorCode.POLICY_VIOLATION.
    "CategoryPolicy",
    "ContentSafetyScanner",
    "KeywordSafetyProvider",
    "LLMSafetyProvider",
    "SafetyProvider",
    "SafetyVerdict",
]
