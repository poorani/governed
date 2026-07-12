"""Shared machinery for streaming governed's observability data to external
systems: a default HTTP POST transport, and OTLP/HTTP's JSON wire-format
builders. One exporter implementation, reused by both observability streams
this framework has -- the decision ledger's sinks
(``HttpDecisionLedgerSink``/``OTelDecisionLedgerSink`` in
``decision_ledger.py``) and the event trace's sinks (``HttpEventSink``/
``OTelEventSink`` in ``logger.py``) -- rather than duplicated per stream.

Hand-built against OTLP's documented JSON mapping, not the
``opentelemetry-sdk`` package: no new dependency, and no exposure to that
SDK's Logs API, which has moved more across versions than Traces/Metrics
have. Point any ``OTel*Sink`` at an OTel Collector's ``/v1/logs`` route, or
directly at any backend with native OTLP ingestion -- Datadog, New Relic,
Dynatrace, and recent Splunk all have one today.
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Callable
from typing import Any

__all__ = [
    "HttpTransport",
    "default_http_transport",
    "otlp_kv",
    "otlp_log_record",
    "otlp_resource_logs",
    "otlp_value",
]

#: ``(url, json_payload, headers) -> None``. The shape every ``Http*Sink`` in
#: this framework accepts as an injectable ``transport``, so each is testable
#: without a network.
HttpTransport = Callable[[str, dict[str, Any], dict[str, str]], None]


def default_http_transport(url: str, payload: dict[str, Any], headers: dict[str, str]) -> None:
    """POST ``payload`` as JSON to ``url``. The default transport for every
    ``Http*Sink``/``OTel*Sink`` in this framework."""
    data = json.dumps(payload, default=str).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=10) as resp:
        resp.read()


def otlp_value(value: Any) -> dict[str, Any]:
    """One value in OTLP's tagged-union JSON encoding for attribute values."""
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    return {"stringValue": str(value)}


def otlp_kv(key: str, value: Any) -> dict[str, Any]:
    return {"key": key, "value": otlp_value(value)}


def otlp_log_record(
    *,
    ts: float,
    body: str,
    attributes: dict[str, Any],
    severity_number: int = 9,  # OTLP's SEVERITY_NUMBER_INFO
    severity_text: str = "INFO",
) -> dict[str, Any]:
    """One OTLP ``LogRecord``. ``ts`` is a Unix timestamp in seconds (as every
    ``time.time()`` call in this codebase already produces); OTLP wants
    nanoseconds as a string, converted here so callers don't have to think
    about it."""
    return {
        "timeUnixNano": str(int(ts * 1e9)),
        "severityNumber": severity_number,
        "severityText": severity_text,
        "body": {"stringValue": body},
        "attributes": [otlp_kv(k, v) for k, v in attributes.items()],
    }


def otlp_resource_logs(
    *, service_name: str, scope_name: str, log_records: list[dict[str, Any]]
) -> dict[str, Any]:
    """Wrap one or more ``otlp_log_record()`` results in OTLP's
    ``resourceLogs``/``scopeLogs`` envelope -- the body an OTLP/HTTP
    ``/v1/logs`` request expects."""
    return {
        "resourceLogs": [
            {
                "resource": {"attributes": [otlp_kv("service.name", service_name)]},
                "scopeLogs": [{"scope": {"name": scope_name}, "logRecords": log_records}],
            }
        ]
    }
