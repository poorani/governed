"""Monitoring pluggability for the event trace: HttpEventSink/OTelEventSink
(the shared exporters.py machinery applied to Event instead of
DecisionRecord), and their config-driven resolution in bootstrap.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from governed import (
    Agent,
    AgentConfig,
    Budget,
    ConsoleSink,
    EventType,
    HttpEventSink,
    InMemoryStore,
    LLMResponse,
    ObservabilityConfig,
    OTelEventSink,
)
from governed.bootstrap import agent_config_from_mapping
from governed.llm import ScriptedClient, ToolCall, Usage
from governed.observability.events import Event
from governed.observability.exporters import (
    otlp_kv,
    otlp_log_record,
    otlp_resource_logs,
    otlp_value,
)
from governed.tools import SubmitTool

# ---------------------------------------------------------------------------
# Shared OTLP builders
# ---------------------------------------------------------------------------


def test_otlp_value_encodes_bool_before_int() -> None:
    # bool is an int subclass in Python -- must check bool first or every
    # True/False gets misencoded as intValue.
    assert otlp_value(True) == {"boolValue": True}
    assert otlp_value(3) == {"intValue": "3"}
    assert otlp_value("x") == {"stringValue": "x"}


def test_otlp_kv_wraps_key_and_value() -> None:
    assert otlp_kv("k", 3) == {"key": "k", "value": {"intValue": "3"}}


def test_otlp_log_record_converts_seconds_to_nanoseconds() -> None:
    record = otlp_log_record(ts=1.5, body="hi", attributes={"a": 1})
    assert record["timeUnixNano"] == str(int(1.5 * 1e9))
    assert record["body"] == {"stringValue": "hi"}
    assert record["attributes"] == [{"key": "a", "value": {"intValue": "1"}}]


def test_otlp_resource_logs_envelope_shape() -> None:
    body = otlp_resource_logs(service_name="svc", scope_name="scope", log_records=[{"x": 1}])
    resource = body["resourceLogs"][0]
    assert resource["resource"]["attributes"] == [otlp_kv("service.name", "svc")]
    assert resource["scopeLogs"][0]["scope"]["name"] == "scope"
    assert resource["scopeLogs"][0]["logRecords"] == [{"x": 1}]


# ---------------------------------------------------------------------------
# HttpEventSink
# ---------------------------------------------------------------------------


def test_http_event_sink_posts_the_event_as_json() -> None:
    sent = []
    sink = HttpEventSink(
        "https://logs.example/ingest",
        headers={"X-Api-Key": "k"},
        transport=lambda u, p, h: sent.append((u, p, h)),
    )
    event = Event(type=EventType.RUN_START, run_id="r1", data={"goal": "g"})
    sink(event)

    assert len(sent) == 1
    url, payload, headers = sent[0]
    assert url == "https://logs.example/ingest"
    assert payload == event.to_dict()
    assert headers["X-Api-Key"] == "k"


def test_http_event_sink_forwards_everything_by_default() -> None:
    sent = []
    sink = HttpEventSink("https://x", transport=lambda u, p, h: sent.append(p))
    sink(Event(type=EventType.TOOL_CALL, run_id="r1"))
    sink(Event(type=EventType.RUN_END, run_id="r1"))
    assert len(sent) == 2


def test_http_event_sink_event_types_filter() -> None:
    sent = []
    sink = HttpEventSink(
        "https://x",
        event_types={EventType.RUN_END},
        transport=lambda u, p, h: sent.append(p),
    )
    sink(Event(type=EventType.TOOL_CALL, run_id="r1"))
    sink(Event(type=EventType.RUN_END, run_id="r1"))
    assert len(sent) == 1
    assert sent[0]["type"] == "run.end"


# ---------------------------------------------------------------------------
# OTelEventSink
# ---------------------------------------------------------------------------


def test_otel_event_sink_posts_a_valid_otlp_logs_shape() -> None:
    sent = []
    sink = OTelEventSink(
        "https://collector.example:4318", transport=lambda u, p, h: sent.append((u, p, h))
    )
    sink(
        Event(type=EventType.TOOL_CALL, run_id="r1", iteration=2, data={"tool": "file_system"})
    )

    url, body, _ = sent[0]
    assert url == "https://collector.example:4318/v1/logs"
    log_record = body["resourceLogs"][0]["scopeLogs"][0]["logRecords"][0]
    attrs = {a["key"]: a["value"] for a in log_record["attributes"]}
    assert attrs["governed.run_id"] == {"stringValue": "r1"}
    assert attrs["governed.event_type"] == {"stringValue": "tool.call"}
    assert json.loads(log_record["body"]["stringValue"])["run_id"] == "r1"


def test_otel_event_sink_event_types_filter() -> None:
    sent = []
    sink = OTelEventSink(
        "https://x",
        event_types={EventType.RUN_START},
        transport=lambda u, p, h: sent.append(p),
    )
    sink(Event(type=EventType.TOOL_CALL, run_id="r1"))
    assert sent == []
    sink(Event(type=EventType.RUN_START, run_id="r1"))
    assert len(sent) == 1


def test_a_failing_event_sink_does_not_break_a_run(tmp_path: Path) -> None:
    """EventBus already suppresses subscriber exceptions -- confirm that
    still holds with a monitoring sink wired in via subscribers."""

    def broken_transport(url, payload, headers):
        raise RuntimeError("downstream is down")

    sink = HttpEventSink("https://x", transport=broken_transport)
    ws = tmp_path / "workspace"
    ws.mkdir()
    script = [
        LLMResponse(
            text="<plan>"
            + json.dumps(
                {
                    "goal_restatement": "do nothing, just submit",
                    "steps": [{"id": "s1", "description": "submit", "done": False}],
                    "next_action": {
                        "step_id": "s1",
                        "tool": "submit",
                        "rationale": "nothing to do",
                        "success_criteria": "the run ends",
                    },
                }
            )
            + "</plan>",
            usage=Usage(100, 10),
        ),
        LLMResponse(
            tool_calls=[
                ToolCall(
                    "c1",
                    "submit",
                    {
                        "answer": "done",
                        "status": "complete",
                        "confidence": 1.0,
                        "evidence": ["n/a"],
                        "unmet_requirements": [],
                    },
                )
            ],
            usage=Usage(100, 10),
        ),
    ]
    agent = Agent(
        AgentConfig(
            llm=ScriptedClient(script),
            workspace=ws,
            skills_dirs=[],
            store=InMemoryStore(),
            budget=Budget(max_iterations=3),
            tools=[SubmitTool()],
            observability=ObservabilityConfig(subscribers=[sink], console=False),
        )
    )
    result = agent.run("do nothing, just submit")
    assert result.ok  # completed normally despite the sink's transport raising


# ---------------------------------------------------------------------------
# Config-driven resolution (bootstrap.py)
# ---------------------------------------------------------------------------


def test_bootstrap_resolves_typed_event_sinks() -> None:
    cfg = agent_config_from_mapping(
        {
            "llm": {"provider": "openai", "model": "gpt-4.1", "api_key": "x"},
            "observability": {
                "subscribers": [
                    {"type": "http", "url": "https://logs.example/ingest"},
                    {"type": "otel", "endpoint": "https://collector.example:4318"},
                    {"type": "console", "verbose": True},
                    {"type": "logging"},
                ]
            },
        }
    )
    kinds = [type(s).__name__ for s in cfg.subscribers]
    assert kinds == ["HttpEventSink", "OTelEventSink", "ConsoleSink", "LoggingSink"]
    assert isinstance(cfg.subscribers[2], ConsoleSink) and cfg.subscribers[2].verbose is True


def test_bootstrap_event_sink_event_types_resolve_by_value() -> None:
    cfg = agent_config_from_mapping(
        {
            "llm": {"provider": "openai", "model": "gpt-4.1", "api_key": "x"},
            "observability": {
                "subscribers": [
                    {
                        "type": "http",
                        "url": "https://x",
                        "event_types": ["tool.call", "run.end"],
                    }
                ]
            },
        }
    )
    sink = cfg.subscribers[0]
    assert sink.event_types == {EventType.TOOL_CALL, EventType.RUN_END}


def test_bootstrap_unknown_event_sink_type_raises() -> None:
    with pytest.raises(ValueError, match="bogus"):
        agent_config_from_mapping(
            {
                "llm": {"provider": "openai", "model": "gpt-4.1", "api_key": "x"},
                "observability": {"subscribers": [{"type": "bogus"}]},
            }
        )
