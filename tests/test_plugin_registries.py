"""Plugin interfaces: config-driven registration of tools, skills, and
observability sinks/stores -- one `register_x(name, factory)` /
`{"type": name, ...}` pattern, mirrored across all five registries this
framework ships (LLM providers, tools, skill sources, decision-ledger
sinks/stores, event sinks, state stores).

Every test that mutates a module-global registry does so through
`monkeypatch.setitem` on the underlying private dict, so nothing leaks
between tests -- the same pattern `test_llm_factory.py` already established
for `register_provider`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

from governed import (
    Agent,
    AgentConfig,
    Budget,
    InMemoryStore,
    LLMResponse,
    Skill,
    SkillLibrary,
    Tool,
    ToolResult,
    ToolSafety,
)
from governed.bootstrap import (
    _DECISION_STORE_BUILDERS,
    _EVENT_SINK_BUILDERS,
    _SINK_BUILDERS,
    _STATE_STORE_BUILDERS,
    agent_config_from_mapping,
    registered_decision_ledger_sinks,
    registered_decision_ledger_stores,
    registered_event_sinks,
    registered_state_stores,
)
from governed.llm import ScriptedClient
from governed.skills.loader import (
    _SKILL_SOURCE_REGISTRY,
    SkillConfig,
    registered_skill_sources,
    resolve_skills,
)
from governed.tools import _TOOL_REGISTRY, registered_tool_names, resolve_tools
from governed.tools.base import ToolConfig, ToolContext

# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------


class _PingTool(Tool):
    name = "ping"
    description = "replies pong"
    safety = ToolSafety.READ_ONLY
    returns = "the string 'pong'"

    class Input(BaseModel):
        pass

    def run(self, args: Input, ctx: ToolContext) -> ToolResult:
        return ToolResult.success("pong")


def test_register_tool_extends_the_resolver(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(_TOOL_REGISTRY, "ping", _PingTool)
    assert "ping" in registered_tool_names()

    tools = resolve_tools(ToolConfig(names=["ping", "submit"]), skills=None)
    assert {t.name for t in tools} == {"ping", "submit"}


def test_register_tool_can_replace_a_builtin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(_TOOL_REGISTRY, "scratchpad", _PingTool)
    tools = resolve_tools(ToolConfig(names=["scratchpad", "submit"]), skills=None)
    scratchpad = next(t for t in tools if t.name == "ping" or t.name == "scratchpad")
    assert isinstance(scratchpad, _PingTool)


def test_unregistered_tool_name_still_raises() -> None:
    with pytest.raises(ValueError, match="nonexistent_tool"):
        resolve_tools(ToolConfig(names=["nonexistent_tool"]), skills=None)


def test_registered_tool_names_includes_the_builtins() -> None:
    builtins = {"file_system", "scratchpad", "submit", "execute_code", "analyze_data"}
    assert builtins <= set(registered_tool_names())


def test_agent_construction_picks_up_a_registered_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(_TOOL_REGISTRY, "ping", _PingTool)
    ws = tmp_path / "workspace"
    ws.mkdir()
    agent = Agent(
        AgentConfig(
            llm=ScriptedClient([LLMResponse(text="unused")]),
            workspace=ws,
            skills_dirs=[],
            store=InMemoryStore(),
            budget=Budget(max_iterations=3),
            tools=ToolConfig(names=["ping", "submit"]),
        )
    )
    assert sorted(agent.registry.names) == ["ping", "submit"]


# ---------------------------------------------------------------------------
# Skill source registry
# ---------------------------------------------------------------------------


def _fake_skill_source(config: SkillConfig) -> SkillLibrary:
    return SkillLibrary({"remote": Skill(name="remote", description="loaded off-disk")})


def test_register_skill_source_extends_the_resolver(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(_SKILL_SOURCE_REGISTRY, "fake", _fake_skill_source)
    assert "fake" in registered_skill_sources()

    library = resolve_skills(SkillConfig(source="fake", dirs=["irrelevant-here"]))
    assert library.names == {"remote"}


def test_default_source_is_directory_scanning(tmp_path: Path) -> None:
    skill_dir = tmp_path / "skills" / "demo"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: demo\ndescription: x\n---\nbody\n")
    library = resolve_skills(SkillConfig(dirs=[str(tmp_path / "skills")]))
    assert library.names == {"demo"}


def test_unregistered_skill_source_raises() -> None:
    with pytest.raises(ValueError, match="bogus"):
        resolve_skills(SkillConfig(source="bogus"))


# ---------------------------------------------------------------------------
# Bootstrap: decision-ledger sink/store, event sink, state store registries
# ---------------------------------------------------------------------------


class _FakeSink:
    def __init__(self, **kw: object) -> None:
        self.kw = kw

    def __call__(self, *args: object, **kwargs: object) -> None:
        pass


class _FakeStore:
    def __init__(self, **kw: object) -> None:
        self.kw = kw


def test_register_event_sink_resolves_from_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(_EVENT_SINK_BUILDERS, "cloudwatch", lambda d: _FakeSink(**d))
    assert "cloudwatch" in registered_event_sinks()

    cfg = agent_config_from_mapping(
        {
            "llm": {"provider": "openai", "model": "gpt-4.1", "api_key": "x"},
            "observability": {"subscribers": [{"type": "cloudwatch", "group": "my-group"}]},
        }
    )
    assert isinstance(cfg.subscribers[0], _FakeSink)
    assert cfg.subscribers[0].kw["group"] == "my-group"


def test_register_decision_ledger_sink_resolves_from_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(_SINK_BUILDERS, "carrier_pigeon", lambda d: _FakeSink(**d))
    assert "carrier_pigeon" in registered_decision_ledger_sinks()

    cfg = agent_config_from_mapping(
        {
            "llm": {"provider": "openai", "model": "gpt-4.1", "api_key": "x"},
            "observability": {
                "decision_ledger": {"sinks": [{"type": "carrier_pigeon", "coop": "north"}]}
            },
        }
    )
    assert cfg.decision_ledger is not None
    assert isinstance(cfg.decision_ledger.sinks[0], _FakeSink)


def test_register_decision_ledger_store_resolves_from_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(_DECISION_STORE_BUILDERS, "s3", lambda d: _FakeStore(**d))
    assert "s3" in registered_decision_ledger_stores()

    cfg = agent_config_from_mapping(
        {
            "llm": {"provider": "openai", "model": "gpt-4.1", "api_key": "x"},
            "observability": {
                "decision_ledger": {"store": {"type": "s3", "bucket": "ledgers"}}
            },
        }
    )
    assert cfg.decision_ledger is not None
    assert isinstance(cfg.decision_ledger.store, _FakeStore)


def test_register_state_store_resolves_from_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(_STATE_STORE_BUILDERS, "redis", lambda d: _FakeStore(**d))
    assert "redis" in registered_state_stores()

    cfg = agent_config_from_mapping(
        {
            "llm": {"provider": "openai", "model": "gpt-4.1", "api_key": "x"},
            "store": {"type": "redis", "url": "redis://localhost:6379"},
        }
    )
    assert isinstance(cfg.store, _FakeStore)


def test_resolve_typed_is_case_insensitive() -> None:
    cfg = agent_config_from_mapping(
        {
            "llm": {"provider": "openai", "model": "gpt-4.1", "api_key": "x"},
            "observability": {
                "subscribers": [{"type": "HTTP", "url": "https://logs.example/ingest"}]
            },
        }
    )
    assert type(cfg.subscribers[0]).__name__ == "HttpEventSink"


def test_unregistered_type_error_message_names_the_registered_ones() -> None:
    with pytest.raises(ValueError, match="Known:"):
        agent_config_from_mapping(
            {
                "llm": {"provider": "openai", "model": "gpt-4.1", "api_key": "x"},
                "store": {"type": "not_a_real_store"},
            }
        )


# ---------------------------------------------------------------------------
# End to end: custom tool + custom skill source + custom event sink, all
# registered as plugins, all selected from one config-only bootstrap.
# ---------------------------------------------------------------------------


def test_full_plugin_stack_resolves_from_one_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(_TOOL_REGISTRY, "ping", _PingTool)
    monkeypatch.setitem(_SKILL_SOURCE_REGISTRY, "fake", _fake_skill_source)
    monkeypatch.setitem(_EVENT_SINK_BUILDERS, "cloudwatch", lambda d: _FakeSink(**d))

    cfg = agent_config_from_mapping(
        {
            "llm": {"provider": "openai", "model": "gpt-4.1", "api_key": "x"},
            "tools": {"names": ["ping", "submit"]},
            "skills": {"source": "fake", "dirs": ["irrelevant"]},
            "observability": {"subscribers": [{"type": "cloudwatch", "group": "g"}]},
        },
        overrides={"workspace": tmp_path / "workspace"},
    )
    agent = Agent(cfg)

    assert sorted(agent.registry.names) == ["ping", "submit"]
    assert agent.skills.names == {"remote"}
    assert isinstance(agent.config.subscribers[0], _FakeSink)
