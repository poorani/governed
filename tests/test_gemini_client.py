"""GeminiClient: message/tool-call translation to and from Gemini's shape,
exercised through the same `client=` injection seam
`test_llm_factory.py` uses for Anthropic/OpenAI -- a fake SDK double stands
in for `google.genai.Client`, so these verify governed's own translation
logic, not Google's wire format. No network, no `google-genai` package
required to run.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest

from governed.llm.base import Message, ToolCall, ToolResultBlock
from governed.llm.gemini_client import (
    GeminiClient,
    _from_gemini_response,
    _to_gemini_contents,
    _to_gemini_tools,
)


def _fake_gemini_sdk(
    text: str = "",
    function_calls: list[SimpleNamespace] | None = None,
    *,
    prompt_tokens: int = 12,
    candidate_tokens: int = 6,
) -> SimpleNamespace:
    parts: list[SimpleNamespace] = []
    if text:
        parts.append(SimpleNamespace(text=text, function_call=None))
    for fc in function_calls or []:
        parts.append(SimpleNamespace(text=None, function_call=fc))
    response = SimpleNamespace(
        candidates=[SimpleNamespace(content=SimpleNamespace(parts=parts))],
        usage_metadata=SimpleNamespace(
            prompt_token_count=prompt_tokens, candidates_token_count=candidate_tokens
        ),
    )
    captured: dict[str, Any] = {}

    def generate_content(**kwargs: Any) -> SimpleNamespace:
        captured.update(kwargs)
        return response

    client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    client.captured = captured  # type: ignore[attr-defined]
    return client


# ---------------------------------------------------------------------------
# GeminiClient.__init__
# ---------------------------------------------------------------------------


def test_client_injection_skips_the_sdk_entirely() -> None:
    fake = _fake_gemini_sdk("hi")
    client = GeminiClient(model="gemini-2.5-flash", client=fake)
    assert client.model == "gemini-2.5-flash"
    assert client._client is fake


def test_missing_google_genai_package_is_a_clear_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "google", None)
    monkeypatch.setitem(sys.modules, "google.genai", None)
    with pytest.raises(ImportError, match="google-genai"):
        GeminiClient(model="gemini-2.5-flash", api_key="x")


def test_real_sdk_constructor_forwards_api_key_and_base_url() -> None:
    """No `client=` override -- exercises the actual `genai.Client(**kwargs)`
    construction path. Only meaningful (and only runs) where `google-genai`
    is installed (`pip install 'governed[gemini]'`, part of `[dev]`/`[all]`)
    -- constructing the SDK client stores config, it makes no network call,
    so this stays offline like everything else here."""
    pytest.importorskip("google.genai")
    client = GeminiClient(model="gemini-2.5-flash", api_key="sk-test")
    assert client.model == "gemini-2.5-flash"
    assert client._client is not None

    client_with_base_url = GeminiClient(
        model="gemini-2.5-flash", api_key="sk-test", base_url="https://example.internal"
    )
    assert client_with_base_url._client is not None


# ---------------------------------------------------------------------------
# GeminiClient.complete -- end to end through the fake SDK double
# ---------------------------------------------------------------------------


def test_complete_returns_text_and_usage() -> None:
    fake = _fake_gemini_sdk("hello from gemini", prompt_tokens=20, candidate_tokens=9)
    client = GeminiClient(client=fake)

    resp = client.complete(system="sys", messages=[Message(role="user", text="hi")])

    assert resp.text == "hello from gemini"
    assert resp.tool_calls == []
    assert resp.usage.input_tokens == 20
    assert resp.usage.output_tokens == 9


def test_complete_without_tools_omits_tool_config() -> None:
    fake = _fake_gemini_sdk("ok")
    client = GeminiClient(client=fake)
    client.complete(system="sys", messages=[Message(role="user", text="hi")])
    assert "tools" not in fake.captured["config"]  # type: ignore[attr-defined]
    assert "tool_config" not in fake.captured["config"]  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "choice,expected", [("auto", "AUTO"), ("required", "ANY"), ("none", "NONE")]
)
def test_complete_maps_tool_choice(choice: str, expected: str) -> None:
    fake = _fake_gemini_sdk("ok")
    client = GeminiClient(client=fake)
    tools = [{"name": "search", "description": "look things up", "input_schema": {}}]

    client.complete(
        system="sys",
        messages=[Message(role="user", text="hi")],
        tools=tools,
        tool_choice=choice,  # type: ignore[arg-type]
    )

    config = fake.captured["config"]  # type: ignore[attr-defined]
    assert config["tool_config"]["function_calling_config"]["mode"] == expected
    assert config["tools"][0]["function_declarations"][0]["name"] == "search"


def test_complete_returns_tool_calls() -> None:
    fc = SimpleNamespace(id="call-1", name="search", args={"q": "weather"})
    fake = _fake_gemini_sdk(function_calls=[fc])
    client = GeminiClient(client=fake)

    resp = client.complete(
        system="sys",
        messages=[Message(role="user", text="hi")],
        tools=[{"name": "search", "description": "d", "input_schema": {}}],
        tool_choice="required",
    )

    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0] == ToolCall(
        id="call-1", name="search", arguments={"q": "weather"}
    )


def test_complete_synthesizes_a_call_id_when_the_sdk_omits_one() -> None:
    fc = SimpleNamespace(id=None, name="search", args=None)
    fake = _fake_gemini_sdk(function_calls=[fc])
    client = GeminiClient(client=fake)

    resp = client.complete(system="sys", messages=[Message(role="user", text="hi")])

    assert resp.tool_calls[0].name == "search"
    assert resp.tool_calls[0].id.startswith("search-")
    assert resp.tool_calls[0].arguments == {}


def test_complete_tolerates_missing_usage_metadata() -> None:
    response = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(parts=[SimpleNamespace(text="hi", function_call=None)])
            )
        ],
        usage_metadata=None,
    )
    fake = SimpleNamespace(models=SimpleNamespace(generate_content=lambda **_: response))
    client = GeminiClient(client=fake)

    resp = client.complete(system="sys", messages=[Message(role="user", text="hi")])

    assert resp.usage.input_tokens == 0
    assert resp.usage.output_tokens == 0


# ---------------------------------------------------------------------------
# GeminiClient.complete -- retry on transient transport errors
#
# The agent loop (agent.py's _with_retries) only retries ContractViolation,
# never transport errors, so a Gemini 503/429 would otherwise kill the whole
# run -- this is the client's own safety net.
# ---------------------------------------------------------------------------


def _flaky_gemini_sdk(exc_sequence: list[Exception], text: str = "ok") -> SimpleNamespace:
    response = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(parts=[SimpleNamespace(text=text, function_call=None)])
            )
        ],
        usage_metadata=SimpleNamespace(prompt_token_count=1, candidates_token_count=1),
    )
    calls = {"n": 0}

    def generate_content(**_: Any) -> SimpleNamespace:
        i = calls["n"]
        calls["n"] += 1
        if i < len(exc_sequence):
            raise exc_sequence[i]
        return response

    client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    client.calls = calls  # type: ignore[attr-defined]
    return client


def test_complete_does_not_retry_a_non_transient_error() -> None:
    """No `google-genai` install needed: a plain exception (stands in for any
    non-ServerError/ClientError failure) is never retryable, regardless of
    max_retries."""
    fake = _flaky_gemini_sdk([RuntimeError("boom")])
    client = GeminiClient(client=fake, max_retries=3)

    with pytest.raises(RuntimeError, match="boom"):
        client.complete(system="sys", messages=[Message(role="user", text="hi")])
    assert fake.calls["n"] == 1  # type: ignore[attr-defined]


def test_complete_retries_on_server_error_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    errors = pytest.importorskip("google.genai.errors")
    sleeps: list[float] = []
    monkeypatch.setattr(sys.modules["time"], "sleep", lambda s: sleeps.append(s))

    fake = _flaky_gemini_sdk(
        [
            errors.ServerError(503, {"message": "overloaded", "status": "UNAVAILABLE"}),
            errors.ServerError(503, {"message": "overloaded", "status": "UNAVAILABLE"}),
        ]
    )
    client = GeminiClient(client=fake, max_retries=2, retry_backoff=1.0)

    resp = client.complete(system="sys", messages=[Message(role="user", text="hi")])

    assert resp.text == "ok"
    assert fake.calls["n"] == 3  # type: ignore[attr-defined]
    assert sleeps == [1.0, 2.0]  # exponential backoff


def test_complete_raises_after_exhausting_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    errors = pytest.importorskip("google.genai.errors")
    monkeypatch.setattr(sys.modules["time"], "sleep", lambda _s: None)

    def make() -> Exception:
        return errors.ServerError(503, {"message": "overloaded", "status": "UNAVAILABLE"})

    fake = _flaky_gemini_sdk([make(), make(), make()])
    client = GeminiClient(client=fake, max_retries=2)

    with pytest.raises(errors.ServerError):
        client.complete(system="sys", messages=[Message(role="user", text="hi")])
    assert fake.calls["n"] == 3  # type: ignore[attr-defined]


def test_complete_retries_a_429_client_error(monkeypatch: pytest.MonkeyPatch) -> None:
    errors = pytest.importorskip("google.genai.errors")
    monkeypatch.setattr(sys.modules["time"], "sleep", lambda _s: None)

    fake = _flaky_gemini_sdk(
        [errors.ClientError(429, {"message": "rate limited", "status": "RESOURCE_EXHAUSTED"})]
    )
    client = GeminiClient(client=fake, max_retries=1)

    resp = client.complete(system="sys", messages=[Message(role="user", text="hi")])
    assert resp.text == "ok"


def test_complete_does_not_retry_a_400_client_error(monkeypatch: pytest.MonkeyPatch) -> None:
    errors = pytest.importorskip("google.genai.errors")
    monkeypatch.setattr(sys.modules["time"], "sleep", lambda _s: None)

    fake = _flaky_gemini_sdk(
        [errors.ClientError(400, {"message": "bad request", "status": "INVALID_ARGUMENT"})]
    )
    client = GeminiClient(client=fake, max_retries=3)

    with pytest.raises(errors.ClientError):
        client.complete(system="sys", messages=[Message(role="user", text="hi")])
    assert fake.calls["n"] == 1  # type: ignore[attr-defined]


def test_max_retries_zero_disables_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    errors = pytest.importorskip("google.genai.errors")
    monkeypatch.setattr(sys.modules["time"], "sleep", lambda _s: None)

    fake = _flaky_gemini_sdk(
        [errors.ServerError(503, {"message": "overloaded", "status": "UNAVAILABLE"})]
    )
    client = GeminiClient(client=fake, max_retries=0)

    with pytest.raises(errors.ServerError):
        client.complete(system="sys", messages=[Message(role="user", text="hi")])
    assert fake.calls["n"] == 1  # type: ignore[attr-defined]


def test_retry_settings_are_configurable_via_constructor() -> None:
    fake = _fake_gemini_sdk("ok")
    client = GeminiClient(client=fake, max_retries=5, retry_backoff=0.1)
    assert client.max_retries == 5
    assert client.retry_backoff == 0.1


def test_retry_settings_default_to_a_couple_of_quick_attempts() -> None:
    fake = _fake_gemini_sdk("ok")
    client = GeminiClient(client=fake)
    assert client.max_retries == 2
    assert client.retry_backoff == 1.0


# ---------------------------------------------------------------------------
# _to_gemini_tools
# ---------------------------------------------------------------------------


def test_to_gemini_tools_wraps_every_tool_in_one_function_declarations_block() -> None:
    tools = [
        {"name": "a", "description": "does a", "input_schema": {"type": "object"}},
        {"name": "b", "description": "does b", "input_schema": {"type": "object"}},
    ]
    out = _to_gemini_tools(tools)
    assert len(out) == 1
    names = [fd["name"] for fd in out[0]["function_declarations"]]
    assert names == ["a", "b"]


def test_to_gemini_tools_strips_additional_properties_at_every_depth() -> None:
    """The Gemini REST API rejects `additionalProperties` nested inside an
    `anyOf` branch -- reproduces the shape pydantic generates for a
    `dict[str, str] | None` field (e.g. DataAnalysisTool's `agg`)."""
    tools = [
        {
            "name": "analyze_data",
            "description": "d",
            "input_schema": {
                "type": "object",
                "properties": {
                    "agg": {
                        "anyOf": [
                            {"type": "object", "additionalProperties": {"type": "string"}},
                            {"type": "null"},
                        ],
                    }
                },
            },
        }
    ]
    out = _to_gemini_tools(tools)
    params = out[0]["function_declarations"][0]["parameters"]
    any_of = params["properties"]["agg"]["anyOf"]
    assert "additionalProperties" not in any_of[0]
    assert any_of[0] == {"type": "object"}


# ---------------------------------------------------------------------------
# _to_gemini_contents -- the role/shape translation, including the
# function_response name recovery from an earlier function_call.
# ---------------------------------------------------------------------------


def test_to_gemini_contents_assistant_text_only() -> None:
    out = _to_gemini_contents([Message(role="assistant", text="thinking...")])
    assert out == [{"role": "model", "parts": [{"text": "thinking..."}]}]


def test_to_gemini_contents_assistant_with_tool_calls() -> None:
    msg = Message(
        role="assistant",
        text="",
        tool_calls=[ToolCall(id="c1", name="search", arguments={"q": "x"})],
    )
    out = _to_gemini_contents([msg])
    assert out[0]["role"] == "model"
    assert out[0]["parts"] == [
        {"function_call": {"id": "c1", "name": "search", "args": {"q": "x"}}}
    ]


def test_to_gemini_contents_replays_thought_signature_on_function_call() -> None:
    """Gemini's "thinking" models reject a replayed function_call that's
    missing the thought_signature from the turn that produced it (400
    INVALID_ARGUMENT) -- it must round-trip through ToolCall.meta."""
    msg = Message(
        role="assistant",
        text="",
        tool_calls=[
            ToolCall(
                id="c1",
                name="search",
                arguments={"q": "x"},
                meta={"thought_signature": b"sig"},
            )
        ],
    )
    out = _to_gemini_contents([msg])
    assert out[0]["parts"] == [
        {
            "function_call": {"id": "c1", "name": "search", "args": {"q": "x"}},
            "thought_signature": b"sig",
        }
    ]


def test_to_gemini_contents_assistant_empty_falls_back_to_empty_text_part() -> None:
    out = _to_gemini_contents([Message(role="assistant", text="")])
    assert out == [{"role": "model", "parts": [{"text": ""}]}]


def test_to_gemini_contents_user_tool_result_recovers_function_name() -> None:
    messages = [
        Message(
            role="assistant",
            text="",
            tool_calls=[ToolCall(id="c1", name="search", arguments={})],
        ),
        Message(
            role="user",
            tool_results=[ToolResultBlock(call_id="c1", content="42 degrees")],
        ),
    ]
    out = _to_gemini_contents(messages)
    fr = out[1]["parts"][0]["function_response"]
    assert fr["name"] == "search"
    assert fr["response"] == {"result": "42 degrees"}


def test_to_gemini_contents_error_tool_result_uses_error_key() -> None:
    messages = [
        Message(
            role="assistant",
            text="",
            tool_calls=[ToolCall(id="c1", name="search", arguments={})],
        ),
        Message(
            role="user",
            tool_results=[ToolResultBlock(call_id="c1", content="boom", is_error=True)],
        ),
    ]
    out = _to_gemini_contents(messages)
    fr = out[1]["parts"][0]["function_response"]
    assert fr["response"] == {"error": "boom"}


def test_to_gemini_contents_tool_result_with_unknown_call_falls_back_to_call_id() -> None:
    messages = [
        Message(role="user", tool_results=[ToolResultBlock(call_id="mystery", content="x")])
    ]
    out = _to_gemini_contents(messages)
    assert out[0]["parts"][0]["function_response"]["name"] == "mystery"


def test_to_gemini_contents_user_text_after_tool_result() -> None:
    messages = [
        Message(
            role="assistant",
            text="",
            tool_calls=[ToolCall(id="c1", name="search", arguments={})],
        ),
        Message(
            role="user",
            text="thanks",
            tool_results=[ToolResultBlock(call_id="c1", content="42")],
        ),
    ]
    out = _to_gemini_contents(messages)
    assert out[1]["parts"][-1] == {"text": "thanks"}


def test_to_gemini_contents_empty_user_message_falls_back_to_empty_text_part() -> None:
    out = _to_gemini_contents([Message(role="user", text="")])
    assert out == [{"role": "user", "parts": [{"text": ""}]}]


# ---------------------------------------------------------------------------
# _from_gemini_response
# ---------------------------------------------------------------------------


def test_from_gemini_response_concatenates_multiple_text_parts() -> None:
    resp = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[
                        SimpleNamespace(text="hello ", function_call=None),
                        SimpleNamespace(text="world", function_call=None),
                    ]
                )
            )
        ],
        usage_metadata=SimpleNamespace(prompt_token_count=1, candidates_token_count=1),
    )
    out = _from_gemini_response(resp)
    assert out.text == "hello world"


def test_from_gemini_response_captures_thought_signature_into_tool_call_meta() -> None:
    fc = SimpleNamespace(id="c1", name="search", args={"q": "x"})
    resp = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[
                        SimpleNamespace(text=None, function_call=fc, thought_signature=b"sig")
                    ]
                )
            )
        ],
        usage_metadata=SimpleNamespace(prompt_token_count=1, candidates_token_count=1),
    )
    out = _from_gemini_response(resp)
    assert out.tool_calls[0].meta == {"thought_signature": b"sig"}


def test_from_gemini_response_tolerates_missing_thought_signature() -> None:
    """Parts built via `_fake_gemini_sdk` (and older SDK responses) have no
    `thought_signature` attribute at all -- must not raise."""
    fc = SimpleNamespace(id="c1", name="search", args={})
    fake = _fake_gemini_sdk(function_calls=[fc])
    client = GeminiClient(client=fake)

    resp = client.complete(system="sys", messages=[Message(role="user", text="hi")])

    assert resp.tool_calls[0].meta == {}
