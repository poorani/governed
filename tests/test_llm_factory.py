"""Config-driven LLM selection: LLMConfig -> resolve_llm -> the right adapter.

No network and no real API keys anywhere here. Anthropic/OpenAI adapters are
exercised through their existing `client=` injection seam with a fake SDK
double standing in for `anthropic.Anthropic`/`openai.OpenAI`, so these tests
verify governed's own translation and dispatch logic, not a vendor's wire
format.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from governed import AgentConfig
from governed.llm import (
    LLMClient,
    LLMConfig,
    LLMResponse,
    Message,
    ProviderPolicy,
    ProviderPolicyViolation,
    ScriptedClient,
    Usage,
)
from governed.llm.anthropic_client import AnthropicClient
from governed.llm.factory import _REGISTRY, registered_providers, resolve_llm
from governed.llm.openai_client import OpenAIClient


def _fake_anthropic_sdk(text: str) -> SimpleNamespace:
    response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )
    return SimpleNamespace(messages=SimpleNamespace(create=lambda **_: response))


def _fake_openai_sdk(text: str) -> SimpleNamespace:
    message = SimpleNamespace(content=text, tool_calls=None)
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(prompt_tokens=8, completion_tokens=4),
    )
    completions = SimpleNamespace(create=lambda **_: response)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


def test_resolve_llm_passes_an_existing_client_through_unchanged() -> None:
    client = ScriptedClient([LLMResponse(text="ok")])
    assert resolve_llm(client) is client


def test_resolve_llm_rejects_values_that_are_neither_client_nor_config() -> None:
    with pytest.raises(TypeError):
        resolve_llm("gpt-4.1")  # type: ignore[arg-type]


def test_unknown_provider_raises_and_lists_whats_registered() -> None:
    with pytest.raises(ValueError, match="anthropic"):
        resolve_llm(LLMConfig(provider="not-a-real-provider", model="x"))


def test_anthropic_resolves_from_config_and_completes() -> None:
    config = LLMConfig(
        provider="anthropic",
        model="claude-sonnet-5",
        api_key="sk-test",
        extra={"client": _fake_anthropic_sdk("hello from anthropic")},
    )
    client = resolve_llm(config)

    assert isinstance(client, AnthropicClient)
    assert client.model == "claude-sonnet-5"

    resp = client.complete(system="sys", messages=[Message(role="user", text="hi")])
    assert resp.text == "hello from anthropic"
    assert resp.usage.input_tokens == 10


def test_openai_resolves_from_config_and_completes() -> None:
    # Provider names match case-insensitively.
    config = LLMConfig(
        provider="OpenAI",
        model="gpt-4.1",
        api_key="sk-test",
        base_url="https://example.internal/v1",
        extra={"client": _fake_openai_sdk("hello from openai")},
    )
    client = resolve_llm(config)

    assert isinstance(client, OpenAIClient)
    assert client.model == "gpt-4.1"

    resp = client.complete(system="sys", messages=[Message(role="user", text="hi")])
    assert resp.text == "hello from openai"
    assert resp.usage.input_tokens == 8


def test_provider_swap_via_configuration_only() -> None:
    """The call site is identical for both providers -- only the LLMConfig
    passed into it changes. This is the acceptance case: swapping models is a
    config edit, not a code change."""

    def ask(config: LLMConfig) -> LLMResponse:
        client = resolve_llm(config)
        return client.complete(system="sys", messages=[Message(role="user", text="hi")])

    anthropic_reply = ask(
        LLMConfig(
            provider="anthropic",
            model="claude-sonnet-5",
            extra={"client": _fake_anthropic_sdk("swap works: anthropic")},
        )
    )
    openai_reply = ask(
        LLMConfig(
            provider="openai",
            model="gpt-4.1",
            extra={"client": _fake_openai_sdk("swap works: openai")},
        )
    )

    assert anthropic_reply.text == "swap works: anthropic"
    assert openai_reply.text == "swap works: openai"


def test_agent_config_resolves_llm_config_on_construction() -> None:
    config = AgentConfig(
        llm=LLMConfig(
            provider="anthropic",
            model="claude-sonnet-5",
            extra={"client": _fake_anthropic_sdk("hi")},
        )
    )
    assert isinstance(config.llm, AnthropicClient)
    assert config.llm.model == "claude-sonnet-5"


def test_register_provider_extends_the_resolver_without_touching_the_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A user can plug in a provider governed doesn't ship, purely via
    config -- the small factory layer the framework itself uses."""

    class EchoClient(LLMClient):
        def __init__(self, model: str) -> None:
            self.model = model

        def complete(
            self,
            *,
            system: str,
            messages: list[Message],
            tools: list[dict[str, Any]] | None = None,
            tool_choice: str = "auto",
            max_tokens: int = 4096,
            temperature: float = 0.0,
        ) -> LLMResponse:
            return LLMResponse(text=f"echo:{messages[-1].text}", usage=Usage(1, 1))

    # Registration is process-global; monkeypatch restores it after the test.
    monkeypatch.setitem(_REGISTRY, "echo", lambda cfg: EchoClient(model=cfg.model))
    assert "echo" in registered_providers()

    client = resolve_llm(LLMConfig(provider="echo", model="echo-1"))
    assert isinstance(client, EchoClient)

    resp = client.complete(system="sys", messages=[Message(role="user", text="ping")])
    assert resp.text == "echo:ping"


# ---------------------------------------------------------------------------
# ProviderPolicy: model/provider governance
# ---------------------------------------------------------------------------


def test_provider_policy_permits_an_allowed_provider() -> None:
    policy = ProviderPolicy(allowed_providers=frozenset({"anthropic"}))
    config = LLMConfig(
        provider="anthropic",
        model="claude-sonnet-5",
        extra={"client": _fake_anthropic_sdk("hi")},
    )
    client = resolve_llm(config, policy=policy)
    assert isinstance(client, AnthropicClient)


def test_provider_policy_rejects_a_provider_outside_the_allowlist() -> None:
    policy = ProviderPolicy(allowed_providers=frozenset({"anthropic"}))
    config = LLMConfig(provider="openai", model="gpt-4.1", api_key="sk-test")
    with pytest.raises(ProviderPolicyViolation, match="openai"):
        resolve_llm(config, policy=policy)


def test_provider_policy_restricts_models_within_an_allowed_provider() -> None:
    policy = ProviderPolicy(
        allowed_providers=frozenset({"anthropic"}),
        allowed_models={"anthropic": frozenset({"claude-sonnet-5"})},
    )
    ok = LLMConfig(
        provider="anthropic",
        model="claude-sonnet-5",
        extra={"client": _fake_anthropic_sdk("hi")},
    )
    resolve_llm(ok, policy=policy)  # does not raise

    disallowed_model = LLMConfig(provider="anthropic", model="claude-opus-4-8")
    with pytest.raises(ProviderPolicyViolation, match="claude-opus-4-8"):
        resolve_llm(disallowed_model, policy=policy)


def test_provider_policy_is_not_consulted_for_an_already_built_client() -> None:
    """Documented scope boundary: an opaque LLMClient bypasses the policy --
    there is no provider/model to check it against. See ProviderPolicy's
    docstring."""
    policy = ProviderPolicy(allowed_providers=frozenset({"anthropic"}))
    client = ScriptedClient([LLMResponse(text="ok")], model="totally-unapproved-model")
    assert resolve_llm(client, policy=policy) is client


def test_agent_config_enforces_provider_policy_at_construction() -> None:
    with pytest.raises(ProviderPolicyViolation, match="openai"):
        AgentConfig(
            llm=LLMConfig(provider="openai", model="gpt-4.1", api_key="sk-test"),
            provider_policy=ProviderPolicy(allowed_providers=frozenset({"anthropic"})),
        )

    # The same policy admits the provider it names.
    config = AgentConfig(
        llm=LLMConfig(
            provider="anthropic",
            model="claude-sonnet-5",
            extra={"client": _fake_anthropic_sdk("hi")},
        ),
        provider_policy=ProviderPolicy(allowed_providers=frozenset({"anthropic"})),
    )
    assert isinstance(config.llm, AnthropicClient)
