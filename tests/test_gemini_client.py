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
