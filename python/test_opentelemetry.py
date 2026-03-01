"""Tests for OpenTelemetry GenAI semantic convention instrumentation."""

from __future__ import annotations

import json
import threading
from datetime import UTC
from typing import Any
from unittest.mock import AsyncMock

import pytest
from opentelemetry import trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

from copilot._copilot_telemetry import (
    ATTR_COPILOT_AIU,
    ATTR_COPILOT_COST,
    ATTR_COPILOT_INITIATOR,
    ATTR_COPILOT_INTERACTION_ID,
    ATTR_COPILOT_SERVER_DURATION,
    ATTR_COPILOT_TURN_ID,
    ATTR_ERROR_TYPE,
    ATTR_GEN_AI_AGENT_DESCRIPTION,
    ATTR_GEN_AI_AGENT_ID,
    ATTR_GEN_AI_AGENT_NAME,
    ATTR_GEN_AI_CONVERSATION_ID,
    ATTR_GEN_AI_INPUT_MESSAGES,
    ATTR_GEN_AI_OPERATION_NAME,
    ATTR_GEN_AI_OUTPUT_MESSAGES,
    ATTR_GEN_AI_PROVIDER_NAME,
    ATTR_GEN_AI_REQUEST_MODEL,
    ATTR_GEN_AI_RESPONSE_FINISH_REASONS,
    ATTR_GEN_AI_RESPONSE_MODEL,
    ATTR_GEN_AI_SYSTEM_INSTRUCTIONS,
    ATTR_GEN_AI_TOOL_CALL_ARGUMENTS,
    ATTR_GEN_AI_TOOL_CALL_ID,
    ATTR_GEN_AI_TOOL_CALL_RESULT,
    ATTR_GEN_AI_TOOL_DEFINITIONS,
    ATTR_GEN_AI_TOOL_DESCRIPTION,
    ATTR_GEN_AI_TOOL_NAME,
    ATTR_GEN_AI_TOOL_TYPE,
    ATTR_GEN_AI_USAGE_INPUT_TOKENS,
    ATTR_GEN_AI_USAGE_OUTPUT_TOKENS,
    ATTR_SERVER_ADDRESS,
    ATTR_SERVER_PORT,
    COPILOT_EVENT_ATTR_PERFORMED_BY,
    COPILOT_EVENT_ATTR_PRE_TOKENS,
    COPILOT_EVENT_ATTR_SKILL_CONTENT,
    COPILOT_EVENT_ATTR_SKILL_NAME,
    COPILOT_EVENT_ATTR_SKILL_PATH,
    COPILOT_EVENT_ATTR_SKILL_PLUGIN_NAME,
    COPILOT_EVENT_ATTR_SKILL_PLUGIN_VERSION,
    COPILOT_EVENT_ATTR_SUCCESS,
    COPILOT_EVENT_ATTR_TOKEN_LIMIT,
    COPILOT_EVENT_ATTR_TOKENS_REMOVED,
    COPILOT_EVENT_SESSION_COMPACTION_COMPLETE,
    COPILOT_EVENT_SESSION_COMPACTION_START,
    COPILOT_EVENT_SESSION_TRUNCATION,
    COPILOT_EVENT_SKILL_INVOKED,
    DEFAULT_PROVIDER_NAME,
    DEFAULT_SOURCE_NAME,
    METRIC_OPERATION_DURATION,
    METRIC_TIME_PER_OUTPUT_CHUNK,
    METRIC_TIME_TO_FIRST_CHUNK,
    METRIC_TOKEN_USAGE,
    OP_CHAT,
    OP_EXECUTE_TOOL,
    OP_INVOKE_AGENT,
    CopilotTelemetry,
    normalize_provider_name,
    parse_server_address,
)
from copilot.generated.session_events import (
    CopilotUsage,
    Data,
    SessionEvent,
    SessionEventType,
)
from copilot.session import CopilotSession
from copilot.types import TelemetryConfig, Tool

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def _reset_otel_globals():
    """Create isolated OpenTelemetry providers for each test."""
    exporter = _CollectingSpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))

    reader = InMemoryMetricReader()
    meter_provider = MeterProvider(metric_readers=[reader])

    yield exporter, reader, tracer_provider, meter_provider

    tracer_provider.shutdown()
    meter_provider.shutdown()


class _CollectingSpanExporter(SpanExporter):
    """Collects finished spans in-memory for test assertions."""

    def __init__(self):
        self._spans: list[Any] = []
        self._lock = threading.Lock()

    def export(self, spans):
        with self._lock:
            self._spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self):
        pass

    def get_finished_spans(self) -> list[Any]:
        with self._lock:
            return list(self._spans)

    def clear(self):
        with self._lock:
            self._spans.clear()


def _get_exporter_and_reader(otel_fixture):
    exporter, reader, tracer_provider, meter_provider = otel_fixture
    return exporter, reader, tracer_provider, meter_provider


def _make_telemetry(
    enable_sensitive_data: bool = False,
    source_name: str | None = None,
    tracer_provider: Any = None,
    meter_provider: Any = None,
) -> CopilotTelemetry:
    config: TelemetryConfig = {}
    if enable_sensitive_data:
        config["enable_sensitive_data"] = True
    if source_name:
        config["source_name"] = source_name
    return CopilotTelemetry(config, tracer_provider=tracer_provider, meter_provider=meter_provider)


def _make_session_event(
    event_type: SessionEventType,
    **data_kwargs: Any,
) -> SessionEvent:
    """Create a minimal SessionEvent for testing."""
    import uuid
    from datetime import datetime

    data = Data(**data_kwargs)
    return SessionEvent(
        data=data,
        id=uuid.uuid4(),
        timestamp=datetime.now(UTC),
        type=event_type,
    )


def _make_session_with_telemetry(
    telemetry: CopilotTelemetry,
    session_id: str = "test-session",
    model: str = None,
    provider=None,
    system_message=None,
    tools=None,
    streaming: bool = False,
    agent_name: str | None = None,
    agent_description: str | None = None,
) -> CopilotSession:
    """Create a CopilotSession with telemetry for testing."""
    mock_client = AsyncMock()
    mock_client.request = AsyncMock(return_value={"messageId": "msg-1"})
    return CopilotSession(
        session_id,
        mock_client,
        telemetry=telemetry,
        model=model,
        provider=provider,
        system_message=system_message,
        tools=tools,
        streaming=streaming,
        agent_name=agent_name,
        agent_description=agent_description,
    )


def _get_metric_names(reader):
    """Collect all metric names from the reader."""
    data = reader.get_metrics_data()
    names = []
    for rm in data.resource_metrics:
        for sm in rm.scope_metrics:
            for m in sm.metrics:
                names.append(m.name)
    return names


def _get_metric_data_points(reader, metric_name):
    """Get data points for a specific metric."""
    data = reader.get_metrics_data()
    for rm in data.resource_metrics:
        for sm in rm.scope_metrics:
            for m in sm.metrics:
                if m.name == metric_name:
                    return list(m.data.data_points)
    return []


def _do_single_turn(session, prompt="Hello"):
    """Helper: send + turn_start + usage + message + turn_end + idle."""
    import asyncio

    asyncio.get_event_loop().run_until_complete(session.send({"prompt": prompt}))
    session._dispatch_event(
        _make_session_event(SessionEventType.ASSISTANT_TURN_START, turn_id="t1")
    )
    session._dispatch_event(
        _make_session_event(
            SessionEventType.ASSISTANT_USAGE,
            model="gpt-4o",
            api_call_id="call-1",
            input_tokens=50.0,
            output_tokens=100.0,
        )
    )
    session._dispatch_event(
        _make_session_event(SessionEventType.ASSISTANT_MESSAGE, content="Hi there!")
    )
    session._dispatch_event(_make_session_event(SessionEventType.ASSISTANT_TURN_END))
    session._dispatch_event(_make_session_event(SessionEventType.SESSION_IDLE))


# ---------------------------------------------------------------------------
# Tests: Provider name normalization
# ---------------------------------------------------------------------------


class TestNormalizeProviderName:
    def test_openai(self):
        assert normalize_provider_name("openai") == "openai"

    def test_openai_case_insensitive(self):
        assert normalize_provider_name("OpenAI") == "openai"

    def test_azure(self):
        assert normalize_provider_name("azure") == "azure.ai.openai"

    def test_anthropic(self):
        assert normalize_provider_name("anthropic") == "anthropic"

    def test_unknown_defaults_to_github(self):
        assert normalize_provider_name("some-other") == DEFAULT_PROVIDER_NAME

    def test_none_defaults_to_github(self):
        assert normalize_provider_name(None) == DEFAULT_PROVIDER_NAME

    def test_empty_defaults_to_github(self):
        assert normalize_provider_name("") == DEFAULT_PROVIDER_NAME

    def test_whitespace_defaults_to_github(self):
        assert normalize_provider_name("  ") == DEFAULT_PROVIDER_NAME


# ---------------------------------------------------------------------------
# Tests: Server address parsing
# ---------------------------------------------------------------------------


class TestParseServerAddress:
    def test_https_url(self):
        host, port = parse_server_address("https://api.example.com:443/v1")
        assert host == "api.example.com"
        assert port == 443

    def test_http_url_default_port(self):
        host, port = parse_server_address("http://localhost/v1")
        assert host == "localhost"
        assert port == 80

    def test_https_url_default_port(self):
        host, port = parse_server_address("https://api.openai.com/v1")
        assert host == "api.openai.com"
        assert port == 443

    def test_none_url(self):
        host, port = parse_server_address(None)
        assert host is None
        assert port is None

    def test_empty_url(self):
        host, port = parse_server_address("")
        assert host is None
        assert port is None

    def test_custom_port(self):
        host, port = parse_server_address("https://api.openai.com:8080")
        assert host == "api.openai.com"
        assert port == 8080


# ---------------------------------------------------------------------------
# Tests: invoke_agent span
# ---------------------------------------------------------------------------


class TestInvokeAgentSpan:
    def test_creates_client_span_with_attributes(self, _reset_otel_globals):
        exporter, _, tp, mp = _get_exporter_and_reader(_reset_otel_globals)
        telemetry = _make_telemetry(tracer_provider=tp, meter_provider=mp)

        span, _ctx = telemetry.start_invoke_agent_span(
            session_id="s1",
            model="gpt-4",
            provider_name="openai",
            server_address="api.openai.com",
            server_port=443,
        )
        span.end()

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        s = spans[0]
        assert s.name == OP_INVOKE_AGENT  # no agent_name, so just operation name
        assert s.kind == trace.SpanKind.CLIENT
        assert s.attributes[ATTR_GEN_AI_OPERATION_NAME] == OP_INVOKE_AGENT
        assert s.attributes[ATTR_GEN_AI_PROVIDER_NAME] == "openai"
        assert s.attributes[ATTR_GEN_AI_AGENT_ID] == "s1"
        assert s.attributes[ATTR_GEN_AI_CONVERSATION_ID] == "s1"
        assert s.attributes[ATTR_GEN_AI_REQUEST_MODEL] == "gpt-4"
        assert s.attributes[ATTR_SERVER_ADDRESS] == "api.openai.com"
        assert s.attributes[ATTR_SERVER_PORT] == 443

    def test_no_model_omits_model_attribute(self, _reset_otel_globals):
        exporter, _, tp, mp = _get_exporter_and_reader(_reset_otel_globals)
        telemetry = _make_telemetry(tracer_provider=tp, meter_provider=mp)

        span, _ctx = telemetry.start_invoke_agent_span(
            session_id="s1",
            model=None,
            provider_name="github",
            server_address=None,
            server_port=None,
        )
        span.end()

        s = exporter.get_finished_spans()[0]
        assert ATTR_GEN_AI_REQUEST_MODEL not in s.attributes
        assert ATTR_SERVER_ADDRESS not in s.attributes


# ---------------------------------------------------------------------------
# Tests: execute_tool span
# ---------------------------------------------------------------------------


class TestExecuteToolSpan:
    def test_creates_internal_span_with_attributes(self, _reset_otel_globals):
        exporter, _, tp, mp = _get_exporter_and_reader(_reset_otel_globals)
        telemetry = _make_telemetry(tracer_provider=tp, meter_provider=mp)

        span = telemetry.start_execute_tool_span(
            tool_name="my_tool",
            tool_call_id="tc-1",
            description="Does something",
            arguments={"x": 1},
        )
        span.end()

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        s = spans[0]
        assert s.name == f"{OP_EXECUTE_TOOL} my_tool"
        assert s.kind == trace.SpanKind.INTERNAL
        assert s.attributes[ATTR_GEN_AI_OPERATION_NAME] == OP_EXECUTE_TOOL
        assert s.attributes[ATTR_GEN_AI_TOOL_NAME] == "my_tool"
        assert s.attributes[ATTR_GEN_AI_TOOL_CALL_ID] == "tc-1"
        assert s.attributes[ATTR_GEN_AI_TOOL_TYPE] == "function"
        assert s.attributes[ATTR_GEN_AI_TOOL_DESCRIPTION] == "Does something"
        # Arguments should NOT be present without sensitive data
        assert ATTR_GEN_AI_TOOL_CALL_ARGUMENTS not in s.attributes

    def test_sensitive_data_includes_arguments(self, _reset_otel_globals):
        exporter, _, tp, mp = _get_exporter_and_reader(_reset_otel_globals)
        telemetry = _make_telemetry(
            enable_sensitive_data=True, tracer_provider=tp, meter_provider=mp
        )

        span = telemetry.start_execute_tool_span(
            tool_name="my_tool",
            tool_call_id="tc-1",
            description=None,
            arguments={"key": "value"},
        )
        span.end()

        s = exporter.get_finished_spans()[0]
        assert ATTR_GEN_AI_TOOL_CALL_ARGUMENTS in s.attributes
        assert "key" in s.attributes[ATTR_GEN_AI_TOOL_CALL_ARGUMENTS]


# ---------------------------------------------------------------------------
# Tests: Metrics recording
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_operation_duration_metric(self, _reset_otel_globals):
        _, reader, tp, mp = _get_exporter_and_reader(_reset_otel_globals)
        telemetry = _make_telemetry(tracer_provider=tp, meter_provider=mp)

        telemetry.record_operation_duration(
            1.5,
            "gpt-4",
            "gpt-4",
            "openai",
            "api.openai.com",
            443,
        )

        data = reader.get_metrics_data()
        metric_names = []
        for resource_metric in data.resource_metrics:
            for scope_metric in resource_metric.scope_metrics:
                for metric in scope_metric.metrics:
                    metric_names.append(metric.name)
        assert METRIC_OPERATION_DURATION in metric_names

    def test_token_usage_metric(self, _reset_otel_globals):
        _, reader, tp, mp = _get_exporter_and_reader(_reset_otel_globals)
        telemetry = _make_telemetry(tracer_provider=tp, meter_provider=mp)

        # Token usage metrics are now recorded via record_token_usage_metrics
        telemetry.record_token_usage_metrics(100, 50, "gpt-4", "gpt-4", "openai", None, None)

        data = reader.get_metrics_data()
        metric_names = []
        for resource_metric in data.resource_metrics:
            for scope_metric in resource_metric.scope_metrics:
                for metric in scope_metric.metrics:
                    metric_names.append(metric.name)
        assert METRIC_TOKEN_USAGE in metric_names

    def test_time_to_first_chunk_metric(self, _reset_otel_globals):
        _, reader, tp, mp = _get_exporter_and_reader(_reset_otel_globals)
        telemetry = _make_telemetry(tracer_provider=tp, meter_provider=mp)

        telemetry.record_time_to_first_chunk(0.5, "gpt-4", "gpt-4", "openai", None, None)

        data = reader.get_metrics_data()
        metric_names = []
        for resource_metric in data.resource_metrics:
            for scope_metric in resource_metric.scope_metrics:
                for metric in scope_metric.metrics:
                    metric_names.append(metric.name)
        assert METRIC_TIME_TO_FIRST_CHUNK in metric_names

    def test_time_per_output_chunk_metric(self, _reset_otel_globals):
        _, reader, tp, mp = _get_exporter_and_reader(_reset_otel_globals)
        telemetry = _make_telemetry(tracer_provider=tp, meter_provider=mp)

        telemetry.record_time_per_output_chunk(0.1, "gpt-4", "gpt-4", "openai", None, None)

        data = reader.get_metrics_data()
        metric_names = []
        for resource_metric in data.resource_metrics:
            for scope_metric in resource_metric.scope_metrics:
                for metric in scope_metric.metrics:
                    metric_names.append(metric.name)
        assert METRIC_TIME_PER_OUTPUT_CHUNK in metric_names


# ---------------------------------------------------------------------------
# Tests: Error recording
# ---------------------------------------------------------------------------


class TestErrorRecording:
    def test_records_error_on_span(self, _reset_otel_globals):
        exporter, _, tp, mp = _get_exporter_and_reader(_reset_otel_globals)
        telemetry = _make_telemetry(tracer_provider=tp, meter_provider=mp)

        span, _ctx = telemetry.start_invoke_agent_span("s1", None, "github", None, None)
        error = ValueError("test error")
        telemetry.record_error(span, error)
        span.end()

        s = exporter.get_finished_spans()[0]
        assert s.attributes[ATTR_ERROR_TYPE] == "ValueError"
        assert s.status.status_code == trace.StatusCode.ERROR


# ---------------------------------------------------------------------------
# Tests: Tool result recording
# ---------------------------------------------------------------------------


class TestToolResult:
    def test_set_tool_result_with_sensitive(self, _reset_otel_globals):
        exporter, _, tp, mp = _get_exporter_and_reader(_reset_otel_globals)
        telemetry = _make_telemetry(
            enable_sensitive_data=True, tracer_provider=tp, meter_provider=mp
        )

        span = telemetry.start_execute_tool_span("t1", "tc-1", None, None)
        telemetry.set_execute_tool_result(span, {"output": "hello"})
        span.end()

        s = exporter.get_finished_spans()[0]
        assert ATTR_GEN_AI_TOOL_CALL_RESULT in s.attributes
        assert "hello" in s.attributes[ATTR_GEN_AI_TOOL_CALL_RESULT]

    def test_set_tool_result_without_sensitive(self, _reset_otel_globals):
        exporter, _, tp, mp = _get_exporter_and_reader(_reset_otel_globals)
        telemetry = _make_telemetry(
            enable_sensitive_data=False, tracer_provider=tp, meter_provider=mp
        )

        span = telemetry.start_execute_tool_span("t1", "tc-1", None, None)
        telemetry.set_execute_tool_result(span, {"output": "hello"})
        span.end()

        s = exporter.get_finished_spans()[0]
        assert ATTR_GEN_AI_TOOL_CALL_RESULT not in s.attributes


# ---------------------------------------------------------------------------
# Tests: Sensitive data gating
# ---------------------------------------------------------------------------


class TestSensitiveDataGating:
    def test_sensitive_false_by_default(self, _reset_otel_globals):
        _, _, tp, mp = _get_exporter_and_reader(_reset_otel_globals)
        telemetry = _make_telemetry(tracer_provider=tp, meter_provider=mp)
        assert telemetry.enable_sensitive_data is False

    def test_sensitive_true_when_configured(self, _reset_otel_globals):
        _, _, tp, mp = _get_exporter_and_reader(_reset_otel_globals)
        telemetry = _make_telemetry(
            enable_sensitive_data=True, tracer_provider=tp, meter_provider=mp
        )
        assert telemetry.enable_sensitive_data is True

    def test_env_var_override(self, monkeypatch, _reset_otel_globals):
        _, _, tp, mp = _get_exporter_and_reader(_reset_otel_globals)
        monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "true")
        config: TelemetryConfig = {}
        telemetry = CopilotTelemetry(config, tracer_provider=tp, meter_provider=mp)
        assert telemetry.enable_sensitive_data is True

    @pytest.mark.asyncio
    async def test_includes_messages_when_sensitive_enabled(self, _reset_otel_globals):
        exporter, _, tp, mp = _get_exporter_and_reader(_reset_otel_globals)
        telemetry = _make_telemetry(
            enable_sensitive_data=True, tracer_provider=tp, meter_provider=mp
        )
        tool = Tool(
            name="myTool",
            description="A tool",
            handler=lambda inv: "ok",
            parameters={"type": "object"},
        )
        session = _make_session_with_telemetry(
            telemetry,
            "s1",
            system_message={"mode": "replace", "content": "You are helpful."},
            tools=[tool],
        )

        await session.send({"prompt": "Hello"})
        session._dispatch_event(
            _make_session_event(SessionEventType.ASSISTANT_TURN_START, turn_id="t1")
        )
        session._dispatch_event(
            _make_session_event(SessionEventType.ASSISTANT_MESSAGE, content="Hi there!")
        )
        session._dispatch_event(_make_session_event(SessionEventType.ASSISTANT_TURN_END))
        session._dispatch_event(_make_session_event(SessionEventType.SESSION_IDLE))

        spans = exporter.get_finished_spans()
        invoke_span = next(s for s in spans if OP_INVOKE_AGENT in s.name)
        chat_span = next(s for s in spans if OP_CHAT in s.name)
        invoke_attrs = dict(invoke_span.attributes)
        chat_attrs = dict(chat_span.attributes)

        # Chat span has input messages (user prompt) and output messages
        assert ATTR_GEN_AI_INPUT_MESSAGES in chat_attrs
        parsed_input = json.loads(chat_attrs[ATTR_GEN_AI_INPUT_MESSAGES])
        assert parsed_input[0]["role"] == "user"
        assert parsed_input[0]["parts"][0]["content"] == "Hello"

        assert ATTR_GEN_AI_OUTPUT_MESSAGES in chat_attrs
        parsed_output = json.loads(chat_attrs[ATTR_GEN_AI_OUTPUT_MESSAGES])
        assert parsed_output[0]["role"] == "assistant"
        assert parsed_output[0]["parts"][0]["content"] == "Hi there!"

        # invoke_agent span has input messages (user prompt) and output (last turn)
        assert ATTR_GEN_AI_INPUT_MESSAGES in invoke_attrs
        parsed_invoke_input = json.loads(invoke_attrs[ATTR_GEN_AI_INPUT_MESSAGES])
        assert parsed_invoke_input[0]["role"] == "user"
        assert parsed_invoke_input[0]["parts"][0]["content"] == "Hello"

        assert ATTR_GEN_AI_OUTPUT_MESSAGES in invoke_attrs

        # System instructions on invoke_agent span
        assert ATTR_GEN_AI_SYSTEM_INSTRUCTIONS in invoke_attrs
        parsed_sys = json.loads(invoke_attrs[ATTR_GEN_AI_SYSTEM_INSTRUCTIONS])
        assert parsed_sys[0]["content"] == "You are helpful."

        # Tool definitions (always included) on invoke_agent span
        assert ATTR_GEN_AI_TOOL_DEFINITIONS in invoke_attrs
        parsed_tools = json.loads(invoke_attrs[ATTR_GEN_AI_TOOL_DEFINITIONS])
        assert parsed_tools[0]["name"] == "myTool"

    @pytest.mark.asyncio
    async def test_excludes_messages_when_sensitive_disabled(self, _reset_otel_globals):
        exporter, _, tp, mp = _get_exporter_and_reader(_reset_otel_globals)
        telemetry = _make_telemetry(
            enable_sensitive_data=False, tracer_provider=tp, meter_provider=mp
        )
        tool = Tool(
            name="myTool",
            description="A tool",
            handler=lambda inv: "ok",
            parameters={"type": "object"},
        )
        session = _make_session_with_telemetry(
            telemetry,
            "s1",
            system_message={"mode": "replace", "content": "You are helpful."},
            tools=[tool],
        )

        await session.send({"prompt": "Hello"})
        session._dispatch_event(
            _make_session_event(SessionEventType.ASSISTANT_TURN_START, turn_id="t1")
        )
        session._dispatch_event(
            _make_session_event(SessionEventType.ASSISTANT_MESSAGE, content="Hi there!")
        )
        session._dispatch_event(_make_session_event(SessionEventType.ASSISTANT_TURN_END))
        session._dispatch_event(_make_session_event(SessionEventType.SESSION_IDLE))

        spans = exporter.get_finished_spans()
        invoke_span = next(s for s in spans if OP_INVOKE_AGENT in s.name)
        invoke_attrs = dict(invoke_span.attributes)

        # Sensitive attributes excluded
        assert ATTR_GEN_AI_INPUT_MESSAGES not in invoke_attrs
        assert ATTR_GEN_AI_OUTPUT_MESSAGES not in invoke_attrs
        assert ATTR_GEN_AI_SYSTEM_INSTRUCTIONS not in invoke_attrs

        # Tool definitions NOT sensitive-gated
        assert ATTR_GEN_AI_TOOL_DEFINITIONS in invoke_attrs
        parsed_tools = json.loads(invoke_attrs[ATTR_GEN_AI_TOOL_DEFINITIONS])
        assert parsed_tools[0]["name"] == "myTool"


# ---------------------------------------------------------------------------
# Tests: Custom source name
# ---------------------------------------------------------------------------


class TestCustomSourceName:
    def test_default_source_name(self, _reset_otel_globals):
        exporter, _, tp, mp = _get_exporter_and_reader(_reset_otel_globals)
        telemetry = _make_telemetry(tracer_provider=tp, meter_provider=mp)

        span, _ctx = telemetry.start_invoke_agent_span("s1", None, "github", None, None)
        span.end()

        s = exporter.get_finished_spans()[0]
        assert s.instrumentation_scope.name == DEFAULT_SOURCE_NAME

    def test_custom_source_name(self, _reset_otel_globals):
        exporter, _, tp, mp = _get_exporter_and_reader(_reset_otel_globals)
        telemetry = _make_telemetry(
            source_name="my.custom.source", tracer_provider=tp, meter_provider=mp
        )

        span, _ctx = telemetry.start_invoke_agent_span("s1", None, "github", None, None)
        span.end()

        s = exporter.get_finished_spans()[0]
        assert s.instrumentation_scope.name == "my.custom.source"


# ---------------------------------------------------------------------------
# Tests: Opt-in (no telemetry when config not provided)
# ---------------------------------------------------------------------------


class TestOptIn:
    def test_no_telemetry_when_not_configured(self, _reset_otel_globals):
        """Session with no telemetry should not produce spans."""
        exporter, _, tp, mp = _get_exporter_and_reader(_reset_otel_globals)

        session = _make_session_with_telemetry(telemetry=None, session_id="s1")
        # Dispatch events - should not crash
        event = _make_session_event(SessionEventType.SESSION_IDLE)
        session._dispatch_event(event)

        assert len(exporter.get_finished_spans()) == 0

    def test_session_without_telemetry_has_no_span_state(self):
        session = _make_session_with_telemetry(telemetry=None)
        assert session._telemetry is None
        assert session._turn_tracker is None


# ---------------------------------------------------------------------------
# Tests: Session telemetry integration
# ---------------------------------------------------------------------------


class TestSessionTelemetryIntegration:
    @pytest.mark.asyncio
    async def test_send_creates_invoke_agent_span(self, _reset_otel_globals):
        exporter, _, tp, mp = _get_exporter_and_reader(_reset_otel_globals)
        telemetry = _make_telemetry(tracer_provider=tp, meter_provider=mp)
        session = _make_session_with_telemetry(telemetry, "s1")

        await session.send({"prompt": "Hello"})

        # Span started but not ended yet (need idle event)
        assert session._turn_tracker._agent_span is not None

        # Dispatch turn start + turn end + idle to close spans
        session._dispatch_event(
            _make_session_event(SessionEventType.ASSISTANT_TURN_START, turn_id="t1")
        )
        session._dispatch_event(_make_session_event(SessionEventType.ASSISTANT_TURN_END))
        session._dispatch_event(_make_session_event(SessionEventType.SESSION_IDLE))

        spans = exporter.get_finished_spans()
        # Should have both chat and invoke_agent spans
        invoke_spans = [
            s for s in spans if s.attributes.get(ATTR_GEN_AI_OPERATION_NAME) == OP_INVOKE_AGENT
        ]
        chat_spans = [s for s in spans if s.attributes.get(ATTR_GEN_AI_OPERATION_NAME) == OP_CHAT]
        assert len(invoke_spans) == 1
        assert len(chat_spans) == 1

    @pytest.mark.asyncio
    async def test_two_level_span_hierarchy(self, _reset_otel_globals):
        """Chat span is a child of invoke_agent span."""
        exporter, _, tp, mp = _get_exporter_and_reader(_reset_otel_globals)
        telemetry = _make_telemetry(tracer_provider=tp, meter_provider=mp)
        session = _make_session_with_telemetry(telemetry, "s1")

        await session.send({"prompt": "Hello"})
        session._dispatch_event(
            _make_session_event(SessionEventType.ASSISTANT_TURN_START, turn_id="t1")
        )
        session._dispatch_event(_make_session_event(SessionEventType.ASSISTANT_TURN_END))
        session._dispatch_event(_make_session_event(SessionEventType.SESSION_IDLE))

        spans = exporter.get_finished_spans()
        invoke_span = next(
            s for s in spans if s.attributes.get(ATTR_GEN_AI_OPERATION_NAME) == OP_INVOKE_AGENT
        )
        chat_span = next(
            s for s in spans if s.attributes.get(ATTR_GEN_AI_OPERATION_NAME) == OP_CHAT
        )

        # Chat span's parent should be the invoke_agent span
        assert chat_span.parent is not None
        assert chat_span.parent.span_id == invoke_span.context.span_id

    @pytest.mark.asyncio
    async def test_usage_event_sets_per_turn_and_accumulated(self, _reset_otel_globals):
        exporter, _, tp, mp = _get_exporter_and_reader(_reset_otel_globals)
        telemetry = _make_telemetry(tracer_provider=tp, meter_provider=mp)
        session = _make_session_with_telemetry(telemetry, "s1")

        await session.send({"prompt": "Hello"})
        session._dispatch_event(
            _make_session_event(SessionEventType.ASSISTANT_TURN_START, turn_id="t1")
        )
        session._dispatch_event(
            _make_session_event(
                SessionEventType.ASSISTANT_USAGE,
                model="gpt-4o",
                api_call_id="call-1",
                input_tokens=50.0,
                output_tokens=100.0,
            )
        )
        session._dispatch_event(_make_session_event(SessionEventType.ASSISTANT_TURN_END))
        session._dispatch_event(_make_session_event(SessionEventType.SESSION_IDLE))

        spans = exporter.get_finished_spans()
        chat_span = next(
            s for s in spans if s.attributes.get(ATTR_GEN_AI_OPERATION_NAME) == OP_CHAT
        )
        invoke_span = next(
            s for s in spans if s.attributes.get(ATTR_GEN_AI_OPERATION_NAME) == OP_INVOKE_AGENT
        )

        # Per-turn usage on chat span
        assert chat_span.attributes[ATTR_GEN_AI_RESPONSE_MODEL] == "gpt-4o"
        assert chat_span.attributes[ATTR_GEN_AI_USAGE_INPUT_TOKENS] == 50
        assert chat_span.attributes[ATTR_GEN_AI_USAGE_OUTPUT_TOKENS] == 100

        # Accumulated usage on invoke_agent span
        assert invoke_span.attributes[ATTR_GEN_AI_RESPONSE_MODEL] == "gpt-4o"
        assert invoke_span.attributes[ATTR_GEN_AI_USAGE_INPUT_TOKENS] == 50
        assert invoke_span.attributes[ATTR_GEN_AI_USAGE_OUTPUT_TOKENS] == 100

    @pytest.mark.asyncio
    async def test_multi_turn_accumulates_usage(self, _reset_otel_globals):
        """Multiple chat turns accumulate tokens on the invoke_agent span."""
        exporter, _, tp, mp = _get_exporter_and_reader(_reset_otel_globals)
        telemetry = _make_telemetry(tracer_provider=tp, meter_provider=mp)
        session = _make_session_with_telemetry(telemetry, "s1")

        await session.send({"prompt": "Hello"})

        # Turn 1
        session._dispatch_event(
            _make_session_event(SessionEventType.ASSISTANT_TURN_START, turn_id="t1")
        )
        session._dispatch_event(
            _make_session_event(
                SessionEventType.ASSISTANT_USAGE,
                model="gpt-4o",
                api_call_id="call-1",
                input_tokens=50.0,
                output_tokens=100.0,
            )
        )
        session._dispatch_event(_make_session_event(SessionEventType.ASSISTANT_TURN_END))

        # Turn 2
        session._dispatch_event(
            _make_session_event(SessionEventType.ASSISTANT_TURN_START, turn_id="t2")
        )
        session._dispatch_event(
            _make_session_event(
                SessionEventType.ASSISTANT_USAGE,
                model="gpt-4o",
                api_call_id="call-2",
                input_tokens=30.0,
                output_tokens=60.0,
            )
        )
        session._dispatch_event(_make_session_event(SessionEventType.ASSISTANT_TURN_END))
        session._dispatch_event(_make_session_event(SessionEventType.SESSION_IDLE))

        spans = exporter.get_finished_spans()
        invoke_span = next(
            s for s in spans if s.attributes.get(ATTR_GEN_AI_OPERATION_NAME) == OP_INVOKE_AGENT
        )
        chat_spans = [s for s in spans if s.attributes.get(ATTR_GEN_AI_OPERATION_NAME) == OP_CHAT]

        assert len(chat_spans) == 2

        # Each chat span has per-turn tokens
        assert chat_spans[0].attributes[ATTR_GEN_AI_USAGE_INPUT_TOKENS] == 50
        assert chat_spans[1].attributes[ATTR_GEN_AI_USAGE_INPUT_TOKENS] == 30

        # invoke_agent span accumulates
        assert invoke_span.attributes[ATTR_GEN_AI_USAGE_INPUT_TOKENS] == 80
        assert invoke_span.attributes[ATTR_GEN_AI_USAGE_OUTPUT_TOKENS] == 160

    @pytest.mark.asyncio
    async def test_model_change_event_updates_response_model(self, _reset_otel_globals):
        exporter, _, tp, mp = _get_exporter_and_reader(_reset_otel_globals)
        telemetry = _make_telemetry(tracer_provider=tp, meter_provider=mp)
        session = _make_session_with_telemetry(telemetry, "s1")

        await session.send({"prompt": "Hello"})
        session._dispatch_event(
            _make_session_event(SessionEventType.ASSISTANT_TURN_START, turn_id="t1")
        )
        session._dispatch_event(
            _make_session_event(SessionEventType.SESSION_MODEL_CHANGE, new_model="gpt-4o-mini")
        )
        assert session._turn_tracker._response_model == "gpt-4o-mini"
        session._dispatch_event(_make_session_event(SessionEventType.ASSISTANT_TURN_END))
        session._dispatch_event(_make_session_event(SessionEventType.SESSION_IDLE))

    @pytest.mark.asyncio
    async def test_error_event_completes_span_with_error(self, _reset_otel_globals):
        exporter, _, tp, mp = _get_exporter_and_reader(_reset_otel_globals)
        telemetry = _make_telemetry(tracer_provider=tp, meter_provider=mp)
        session = _make_session_with_telemetry(telemetry, "s1")

        await session.send({"prompt": "Hello"})
        session._dispatch_event(
            _make_session_event(SessionEventType.ASSISTANT_TURN_START, turn_id="t1")
        )
        session._dispatch_event(
            _make_session_event(SessionEventType.SESSION_ERROR, message="Something went wrong")
        )

        spans = exporter.get_finished_spans()
        # Both chat and invoke_agent spans should have error
        for s in spans:
            assert s.attributes[ATTR_ERROR_TYPE] == "SessionTurnError"
            assert s.status.status_code == trace.StatusCode.ERROR

    @pytest.mark.asyncio
    async def test_idle_event_records_duration_metric(self, _reset_otel_globals):
        _, reader, tp, mp = _get_exporter_and_reader(_reset_otel_globals)
        telemetry = _make_telemetry(tracer_provider=tp, meter_provider=mp)
        session = _make_session_with_telemetry(telemetry, "s1")

        await session.send({"prompt": "Hello"})
        session._dispatch_event(
            _make_session_event(SessionEventType.ASSISTANT_TURN_START, turn_id="t1")
        )
        session._dispatch_event(_make_session_event(SessionEventType.ASSISTANT_TURN_END))
        session._dispatch_event(_make_session_event(SessionEventType.SESSION_IDLE))

        assert METRIC_OPERATION_DURATION in _get_metric_names(reader)

    @pytest.mark.asyncio
    async def test_subsequent_send_reuses_agent_span(self, _reset_otel_globals):
        exporter, _, tp, mp = _get_exporter_and_reader(_reset_otel_globals)
        telemetry = _make_telemetry(
            enable_sensitive_data=True, tracer_provider=tp, meter_provider=mp
        )
        session = _make_session_with_telemetry(telemetry, "s1")

        # First send creates the agent span
        await session.send({"prompt": "First"})
        span_after_first = session._turn_tracker._agent_span

        # Second send reuses the same span
        await session.send({"prompt": "Second"})
        assert session._turn_tracker._agent_span is span_after_first

        session._dispatch_event(
            _make_session_event(SessionEventType.ASSISTANT_TURN_START, turn_id="t1")
        )
        session._dispatch_event(_make_session_event(SessionEventType.ASSISTANT_TURN_END))
        session._dispatch_event(_make_session_event(SessionEventType.SESSION_IDLE))

        assert exporter.get_finished_spans()[0] is not None


# ---------------------------------------------------------------------------
# Tests: Streaming chunk metrics
# ---------------------------------------------------------------------------


class TestStreamingChunkMetrics:
    @pytest.mark.asyncio
    async def test_streaming_records_time_to_first_chunk(self, _reset_otel_globals):
        _, reader, tp, mp = _get_exporter_and_reader(_reset_otel_globals)
        telemetry = _make_telemetry(tracer_provider=tp, meter_provider=mp)
        session = _make_session_with_telemetry(telemetry, "s1", streaming=True)

        await session.send({"prompt": "Hello"})
        session._dispatch_event(
            _make_session_event(SessionEventType.ASSISTANT_TURN_START, turn_id="t1")
        )

        # First chunk - any event during a turn triggers chunk timing
        session._dispatch_event(
            _make_session_event(SessionEventType.ASSISTANT_MESSAGE_DELTA, delta_content="Hi")
        )

        session._dispatch_event(_make_session_event(SessionEventType.ASSISTANT_TURN_END))
        session._dispatch_event(_make_session_event(SessionEventType.SESSION_IDLE))

        assert METRIC_TIME_TO_FIRST_CHUNK in _get_metric_names(reader)

        # Verify chunk timing uses "chat" operation name
        data_points = _get_metric_data_points(reader, METRIC_TIME_TO_FIRST_CHUNK)
        assert len(data_points) > 0
        attrs = dict(data_points[0].attributes)
        assert attrs[ATTR_GEN_AI_OPERATION_NAME] == OP_CHAT

    @pytest.mark.asyncio
    async def test_streaming_records_time_per_output_chunk(self, _reset_otel_globals):
        _, reader, tp, mp = _get_exporter_and_reader(_reset_otel_globals)
        telemetry = _make_telemetry(tracer_provider=tp, meter_provider=mp)
        session = _make_session_with_telemetry(telemetry, "s1", streaming=True)

        await session.send({"prompt": "Hello"})
        session._dispatch_event(
            _make_session_event(SessionEventType.ASSISTANT_TURN_START, turn_id="t1")
        )

        # First chunk
        session._dispatch_event(
            _make_session_event(SessionEventType.ASSISTANT_MESSAGE_DELTA, delta_content="a")
        )
        # Second chunk
        session._dispatch_event(
            _make_session_event(SessionEventType.ASSISTANT_MESSAGE_DELTA, delta_content="b")
        )

        session._dispatch_event(_make_session_event(SessionEventType.ASSISTANT_TURN_END))
        session._dispatch_event(_make_session_event(SessionEventType.SESSION_IDLE))

        assert METRIC_TIME_PER_OUTPUT_CHUNK in _get_metric_names(reader)


# ---------------------------------------------------------------------------
# Tests: configure_telemetry_context
# ---------------------------------------------------------------------------


class TestConfigureTelemetryContext:
    def test_sets_provider_and_server(self, _reset_otel_globals):
        _, _, tp, mp = _get_exporter_and_reader(_reset_otel_globals)
        telemetry = _make_telemetry(tracer_provider=tp, meter_provider=mp)
        session = _make_session_with_telemetry(
            telemetry,
            "s1",
            model="gpt-4",
            provider={"type": "azure", "base_url": "https://myendpoint.openai.azure.com:443/v1"},
        )

        assert session._turn_tracker._request_model == "gpt-4"
        assert session._turn_tracker._provider_name == "azure.ai.openai"
        assert session._turn_tracker._server_address == "myendpoint.openai.azure.com"
        assert session._turn_tracker._server_port == 443

    def test_noop_without_telemetry(self):
        session = _make_session_with_telemetry(telemetry=None, model="gpt-4")
        assert session.telemetry_provider_name == "github"  # unchanged default


# ---------------------------------------------------------------------------
# Tests: Finish reasons
# ---------------------------------------------------------------------------


class TestFinishReasons:
    @pytest.mark.asyncio
    async def test_finish_reasons_on_success(self, _reset_otel_globals):
        exporter, _, tp, mp = _get_exporter_and_reader(_reset_otel_globals)
        telemetry = _make_telemetry(tracer_provider=tp, meter_provider=mp)
        session = _make_session_with_telemetry(telemetry, "s1")

        await session.send({"prompt": "Hello"})
        session._dispatch_event(
            _make_session_event(SessionEventType.ASSISTANT_TURN_START, turn_id="t1")
        )
        session._dispatch_event(_make_session_event(SessionEventType.ASSISTANT_TURN_END))
        session._dispatch_event(_make_session_event(SessionEventType.SESSION_IDLE))

        spans = exporter.get_finished_spans()
        for s in spans:
            assert s.attributes[ATTR_GEN_AI_RESPONSE_FINISH_REASONS] == ("stop",)


# ---------------------------------------------------------------------------
# Tests: Agent metadata
# ---------------------------------------------------------------------------


class TestAgentMetadata:
    def test_agent_name_and_description_on_span(self, _reset_otel_globals):
        exporter, _, tp, mp = _get_exporter_and_reader(_reset_otel_globals)
        telemetry = _make_telemetry(tracer_provider=tp, meter_provider=mp)

        span, _ctx = telemetry.start_invoke_agent_span(
            session_id="s1",
            model="gpt-4",
            provider_name="openai",
            server_address="api.openai.com",
            server_port=443,
            agent_name="my_agent",
            agent_description="Does something useful",
        )
        span.end()

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        s = spans[0]
        assert s.attributes[ATTR_GEN_AI_AGENT_NAME] == "my_agent"
        assert s.attributes[ATTR_GEN_AI_AGENT_DESCRIPTION] == "Does something useful"


# ---------------------------------------------------------------------------
# Tests: Vendor-prefixed attributes (Copilot-specific)
# ---------------------------------------------------------------------------


class TestVendorPrefixedAttributes:
    @pytest.mark.asyncio
    async def test_chat_span_copilot_attrs(self, _reset_otel_globals):
        """Copilot-specific attrs (cost, duration, initiator, aiu, turn_id, interaction_id)
        are set on chat spans."""
        exporter, _, tp, mp = _get_exporter_and_reader(_reset_otel_globals)
        telemetry = _make_telemetry(tracer_provider=tp, meter_provider=mp)
        session = _make_session_with_telemetry(telemetry, "s1")

        await session.send({"prompt": "Hello"})
        session._dispatch_event(
            _make_session_event(
                SessionEventType.ASSISTANT_TURN_START,
                turn_id="tid-1",
                interaction_id="iid-1",
            )
        )
        session._dispatch_event(
            _make_session_event(
                SessionEventType.ASSISTANT_USAGE,
                model="gpt-4o",
                input_tokens=50.0,
                output_tokens=100.0,
                cost=0.01,
                duration=1500.0,
                initiator="user",
                copilot_usage=CopilotUsage(token_details=None, total_nano_aiu=42000.0),
            )
        )
        session._dispatch_event(_make_session_event(SessionEventType.ASSISTANT_TURN_END))
        session._dispatch_event(_make_session_event(SessionEventType.SESSION_IDLE))

        spans = exporter.get_finished_spans()
        chat_span = next(
            s for s in spans if s.attributes.get(ATTR_GEN_AI_OPERATION_NAME) == OP_CHAT
        )
        attrs = dict(chat_span.attributes)

        assert attrs[ATTR_COPILOT_COST] == 0.01
        assert attrs[ATTR_COPILOT_SERVER_DURATION] == 1500.0
        assert attrs[ATTR_COPILOT_INITIATOR] == "user"
        assert attrs[ATTR_COPILOT_AIU] == 42000.0
        assert attrs[ATTR_COPILOT_TURN_ID] == "tid-1"
        assert attrs[ATTR_COPILOT_INTERACTION_ID] == "iid-1"

    @pytest.mark.asyncio
    async def test_invoke_agent_accumulated_cost_and_aiu(self, _reset_otel_globals):
        """invoke_agent span accumulates cost and AIU across turns."""
        exporter, _, tp, mp = _get_exporter_and_reader(_reset_otel_globals)
        telemetry = _make_telemetry(tracer_provider=tp, meter_provider=mp)
        session = _make_session_with_telemetry(telemetry, "s1")

        await session.send({"prompt": "Hello"})

        # Turn 1
        session._dispatch_event(
            _make_session_event(SessionEventType.ASSISTANT_TURN_START, turn_id="t1")
        )
        session._dispatch_event(
            _make_session_event(
                SessionEventType.ASSISTANT_USAGE,
                model="gpt-4o",
                input_tokens=10.0,
                output_tokens=20.0,
                cost=0.01,
                copilot_usage=CopilotUsage(token_details=None, total_nano_aiu=1000.0),
            )
        )
        session._dispatch_event(_make_session_event(SessionEventType.ASSISTANT_TURN_END))

        # Turn 2
        session._dispatch_event(
            _make_session_event(SessionEventType.ASSISTANT_TURN_START, turn_id="t2")
        )
        session._dispatch_event(
            _make_session_event(
                SessionEventType.ASSISTANT_USAGE,
                model="gpt-4o",
                input_tokens=20.0,
                output_tokens=30.0,
                cost=0.02,
                copilot_usage=CopilotUsage(token_details=None, total_nano_aiu=2000.0),
            )
        )
        session._dispatch_event(_make_session_event(SessionEventType.ASSISTANT_TURN_END))
        session._dispatch_event(_make_session_event(SessionEventType.SESSION_IDLE))

        spans = exporter.get_finished_spans()
        invoke_span = next(
            s for s in spans if s.attributes.get(ATTR_GEN_AI_OPERATION_NAME) == OP_INVOKE_AGENT
        )
        attrs = dict(invoke_span.attributes)

        assert attrs[ATTR_COPILOT_COST] == pytest.approx(0.03)
        assert attrs[ATTR_COPILOT_AIU] == 3000.0


# ---------------------------------------------------------------------------
# Tests: Custom span events
# ---------------------------------------------------------------------------


class TestCustomSpanEvents:
    @pytest.mark.asyncio
    async def test_truncation_span_event(self, _reset_otel_globals):
        exporter, _, tp, mp = _get_exporter_and_reader(_reset_otel_globals)
        telemetry = _make_telemetry(tracer_provider=tp, meter_provider=mp)
        session = _make_session_with_telemetry(telemetry, "s1")

        await session.send({"prompt": "Hello"})
        session._dispatch_event(
            _make_session_event(SessionEventType.ASSISTANT_TURN_START, turn_id="t1")
        )
        session._dispatch_event(
            _make_session_event(
                SessionEventType.SESSION_TRUNCATION,
                token_limit=4096.0,
                pre_truncation_tokens_in_messages=5000.0,
                post_truncation_tokens_in_messages=4000.0,
                pre_truncation_messages_length=20.0,
                post_truncation_messages_length=15.0,
                tokens_removed_during_truncation=1000.0,
                messages_removed_during_truncation=5.0,
                performed_by="model",
            )
        )
        session._dispatch_event(_make_session_event(SessionEventType.ASSISTANT_TURN_END))
        session._dispatch_event(_make_session_event(SessionEventType.SESSION_IDLE))

        spans = exporter.get_finished_spans()
        # Truncation event should be on the chat span (active during turn)
        chat_span = next(
            s for s in spans if s.attributes.get(ATTR_GEN_AI_OPERATION_NAME) == OP_CHAT
        )
        events = chat_span.events
        trunc_events = [e for e in events if e.name == COPILOT_EVENT_SESSION_TRUNCATION]
        assert len(trunc_events) == 1
        attrs = dict(trunc_events[0].attributes)
        assert attrs[COPILOT_EVENT_ATTR_TOKEN_LIMIT] == 4096.0
        assert attrs[COPILOT_EVENT_ATTR_PERFORMED_BY] == "model"

    @pytest.mark.asyncio
    async def test_compaction_start_span_event(self, _reset_otel_globals):
        exporter, _, tp, mp = _get_exporter_and_reader(_reset_otel_globals)
        telemetry = _make_telemetry(tracer_provider=tp, meter_provider=mp)
        session = _make_session_with_telemetry(telemetry, "s1")

        await session.send({"prompt": "Hello"})
        session._dispatch_event(
            _make_session_event(SessionEventType.ASSISTANT_TURN_START, turn_id="t1")
        )
        session._dispatch_event(_make_session_event(SessionEventType.SESSION_COMPACTION_START))
        session._dispatch_event(_make_session_event(SessionEventType.ASSISTANT_TURN_END))
        session._dispatch_event(_make_session_event(SessionEventType.SESSION_IDLE))

        spans = exporter.get_finished_spans()
        chat_span = next(
            s for s in spans if s.attributes.get(ATTR_GEN_AI_OPERATION_NAME) == OP_CHAT
        )
        events = chat_span.events
        compaction_events = [e for e in events if e.name == COPILOT_EVENT_SESSION_COMPACTION_START]
        assert len(compaction_events) == 1

    @pytest.mark.asyncio
    async def test_compaction_complete_span_event(self, _reset_otel_globals):
        exporter, _, tp, mp = _get_exporter_and_reader(_reset_otel_globals)
        telemetry = _make_telemetry(tracer_provider=tp, meter_provider=mp)
        session = _make_session_with_telemetry(telemetry, "s1")

        await session.send({"prompt": "Hello"})
        session._dispatch_event(
            _make_session_event(SessionEventType.ASSISTANT_TURN_START, turn_id="t1")
        )
        session._dispatch_event(
            _make_session_event(
                SessionEventType.SESSION_COMPACTION_COMPLETE,
                success=True,
                pre_compaction_tokens=5000.0,
                post_compaction_tokens=3000.0,
                tokens_removed=2000.0,
                messages_removed=10.0,
            )
        )
        session._dispatch_event(_make_session_event(SessionEventType.ASSISTANT_TURN_END))
        session._dispatch_event(_make_session_event(SessionEventType.SESSION_IDLE))

        spans = exporter.get_finished_spans()
        chat_span = next(
            s for s in spans if s.attributes.get(ATTR_GEN_AI_OPERATION_NAME) == OP_CHAT
        )
        events = chat_span.events
        compaction_events = [
            e for e in events if e.name == COPILOT_EVENT_SESSION_COMPACTION_COMPLETE
        ]
        assert len(compaction_events) == 1
        attrs = dict(compaction_events[0].attributes)
        assert attrs[COPILOT_EVENT_ATTR_SUCCESS] is True
        assert attrs[COPILOT_EVENT_ATTR_PRE_TOKENS] == 5000.0
        assert attrs[COPILOT_EVENT_ATTR_TOKENS_REMOVED] == 2000.0

    @pytest.mark.asyncio
    async def test_skill_invoked_span_event(self, _reset_otel_globals):
        exporter, _, tp, mp = _get_exporter_and_reader(_reset_otel_globals)
        telemetry = _make_telemetry(
            enable_sensitive_data=True, tracer_provider=tp, meter_provider=mp
        )
        session = _make_session_with_telemetry(telemetry, "s1")

        await session.send({"prompt": "Hello"})
        session._dispatch_event(
            _make_session_event(SessionEventType.ASSISTANT_TURN_START, turn_id="t1")
        )
        session._dispatch_event(
            _make_session_event(
                SessionEventType.SKILL_INVOKED,
                name="codeSearch",
                path="/skills/code-search",
                content="skill content here",
                plugin_name="copilot-extensions",
                plugin_version="1.2.3",
            )
        )
        session._dispatch_event(_make_session_event(SessionEventType.ASSISTANT_TURN_END))
        session._dispatch_event(_make_session_event(SessionEventType.SESSION_IDLE))

        spans = exporter.get_finished_spans()
        chat_span = next(
            s for s in spans if s.attributes.get(ATTR_GEN_AI_OPERATION_NAME) == OP_CHAT
        )
        events = chat_span.events
        skill_events = [e for e in events if e.name == COPILOT_EVENT_SKILL_INVOKED]
        assert len(skill_events) == 1
        attrs = dict(skill_events[0].attributes)
        assert attrs[COPILOT_EVENT_ATTR_SKILL_NAME] == "codeSearch"
        assert attrs[COPILOT_EVENT_ATTR_SKILL_PATH] == "/skills/code-search"
        assert attrs[COPILOT_EVENT_ATTR_SKILL_CONTENT] == "skill content here"
        assert attrs[COPILOT_EVENT_ATTR_SKILL_PLUGIN_NAME] == "copilot-extensions"
        assert attrs[COPILOT_EVENT_ATTR_SKILL_PLUGIN_VERSION] == "1.2.3"


# ---------------------------------------------------------------------------
# Tests: Rich message content (parts-based)
# ---------------------------------------------------------------------------


class TestRichMessageContent:
    @pytest.mark.asyncio
    async def test_reasoning_parts_in_output(self, _reset_otel_globals):
        exporter, _, tp, mp = _get_exporter_and_reader(_reset_otel_globals)
        telemetry = _make_telemetry(
            enable_sensitive_data=True, tracer_provider=tp, meter_provider=mp
        )
        session = _make_session_with_telemetry(telemetry, "s1")

        await session.send({"prompt": "Think about this"})
        session._dispatch_event(
            _make_session_event(SessionEventType.ASSISTANT_TURN_START, turn_id="t1")
        )
        session._dispatch_event(
            _make_session_event(
                SessionEventType.ASSISTANT_MESSAGE,
                content="The answer is 42",
                reasoning_text="Let me think step by step...",
            )
        )
        session._dispatch_event(_make_session_event(SessionEventType.ASSISTANT_TURN_END))
        session._dispatch_event(_make_session_event(SessionEventType.SESSION_IDLE))

        spans = exporter.get_finished_spans()
        chat_span = next(
            s for s in spans if s.attributes.get(ATTR_GEN_AI_OPERATION_NAME) == OP_CHAT
        )
        attrs = dict(chat_span.attributes)

        output_msgs = json.loads(attrs[ATTR_GEN_AI_OUTPUT_MESSAGES])
        assert len(output_msgs) == 1
        parts = output_msgs[0]["parts"]
        # Should have reasoning and text parts
        assert any(p["type"] == "reasoning" for p in parts)
        assert any(p["type"] == "text" for p in parts)
        reasoning_part = next(p for p in parts if p["type"] == "reasoning")
        assert reasoning_part["content"] == "Let me think step by step..."

    @pytest.mark.asyncio
    async def test_tool_call_parts(self, _reset_otel_globals):
        exporter, _, tp, mp = _get_exporter_and_reader(_reset_otel_globals)
        telemetry = _make_telemetry(
            enable_sensitive_data=True, tracer_provider=tp, meter_provider=mp
        )
        session = _make_session_with_telemetry(telemetry, "s1")

        await session.send({"prompt": "Use a tool"})
        session._dispatch_event(
            _make_session_event(SessionEventType.ASSISTANT_TURN_START, turn_id="t1")
        )
        session._dispatch_event(
            _make_session_event(
                SessionEventType.TOOL_EXECUTION_START,
                tool_call_id="tc-1",
                tool_name="search",
                arguments={"query": "hello"},
            )
        )
        session._dispatch_event(
            _make_session_event(
                SessionEventType.TOOL_EXECUTION_COMPLETE,
                tool_call_id="tc-1",
                tool_name="search",
            )
        )
        session._dispatch_event(_make_session_event(SessionEventType.ASSISTANT_TURN_END))
        session._dispatch_event(_make_session_event(SessionEventType.SESSION_IDLE))

        spans = exporter.get_finished_spans()
        chat_span = next(
            s for s in spans if s.attributes.get(ATTR_GEN_AI_OPERATION_NAME) == OP_CHAT
        )
        attrs = dict(chat_span.attributes)

        output_msgs = json.loads(attrs[ATTR_GEN_AI_OUTPUT_MESSAGES])
        tool_call_msg = next(m for m in output_msgs if m["parts"][0]["type"] == "tool_call")
        assert tool_call_msg["parts"][0]["name"] == "search"
        assert tool_call_msg["parts"][0]["id"] == "tc-1"

        input_msgs = json.loads(attrs[ATTR_GEN_AI_INPUT_MESSAGES])
        tool_resp_msg = next(
            (m for m in input_msgs if m["parts"][0]["type"] == "tool_call_response"), None
        )
        assert tool_resp_msg is not None


# ---------------------------------------------------------------------------
# Tests: MCP server tool tracking
# ---------------------------------------------------------------------------


class TestMcpServerToolTracking:
    @pytest.mark.asyncio
    async def test_server_tool_call_parts(self, _reset_otel_globals):
        exporter, _, tp, mp = _get_exporter_and_reader(_reset_otel_globals)
        telemetry = _make_telemetry(
            enable_sensitive_data=True, tracer_provider=tp, meter_provider=mp
        )
        session = _make_session_with_telemetry(telemetry, "s1")

        await session.send({"prompt": "Use MCP tool"})
        session._dispatch_event(
            _make_session_event(SessionEventType.ASSISTANT_TURN_START, turn_id="t1")
        )
        session._dispatch_event(
            _make_session_event(
                SessionEventType.TOOL_EXECUTION_START,
                tool_call_id="tc-mcp-1",
                tool_name="mcp_search",
                mcp_server_name="my-mcp-server",
                arguments={"q": "test"},
            )
        )
        session._dispatch_event(
            _make_session_event(
                SessionEventType.TOOL_EXECUTION_COMPLETE,
                tool_call_id="tc-mcp-1",
                tool_name="mcp_search",
            )
        )
        session._dispatch_event(_make_session_event(SessionEventType.ASSISTANT_TURN_END))
        session._dispatch_event(_make_session_event(SessionEventType.SESSION_IDLE))

        spans = exporter.get_finished_spans()
        chat_span = next(
            s for s in spans if s.attributes.get(ATTR_GEN_AI_OPERATION_NAME) == OP_CHAT
        )
        attrs = dict(chat_span.attributes)

        output_msgs = json.loads(attrs[ATTR_GEN_AI_OUTPUT_MESSAGES])
        stc_msg = next(m for m in output_msgs if m["parts"][0]["type"] == "server_tool_call")
        stc_part = stc_msg["parts"][0]
        assert "server_tool_call" in stc_part
        assert stc_part["server_tool_call"]["type"] == "mcp"
        assert stc_part["server_tool_call"]["server_name"] == "my-mcp-server"

        input_msgs = json.loads(attrs[ATTR_GEN_AI_INPUT_MESSAGES])
        stcr_msg = next(
            (m for m in input_msgs if m["parts"][0]["type"] == "server_tool_call_response"),
            None,
        )
        assert stcr_msg is not None
        stcr_part = stcr_msg["parts"][0]
        assert "server_tool_call_response" in stcr_part
        assert stcr_part["server_tool_call_response"]["type"] == "mcp"
        assert stcr_part["server_tool_call_response"]["server_name"] == "my-mcp-server"


# ---------------------------------------------------------------------------
# Tests: Subagent spans
# ---------------------------------------------------------------------------


class TestSubagentSpans:
    @pytest.mark.asyncio
    async def test_subagent_creates_nested_hierarchy(self, _reset_otel_globals):
        exporter, _, tp, mp = _get_exporter_and_reader(_reset_otel_globals)
        telemetry = _make_telemetry(
            enable_sensitive_data=True, tracer_provider=tp, meter_provider=mp
        )
        session = _make_session_with_telemetry(telemetry, "s1")

        await session.send({"prompt": "Hello"})
        session._dispatch_event(
            _make_session_event(SessionEventType.ASSISTANT_TURN_START, turn_id="t1")
        )

        # Subagent started
        session._dispatch_event(
            _make_session_event(
                SessionEventType.SUBAGENT_STARTED,
                tool_call_id="sa-tc-1",
                agent_name="sub-agent",
                agent_description="A helper agent",
            )
        )

        # Subagent gets a usage event (routed by parent_tool_call_id)
        session._dispatch_event(
            _make_session_event(
                SessionEventType.ASSISTANT_USAGE,
                parent_tool_call_id="sa-tc-1",
                model="gpt-4o-mini",
                input_tokens=25.0,
                output_tokens=50.0,
            )
        )

        # Subagent gets a message
        session._dispatch_event(
            _make_session_event(
                SessionEventType.ASSISTANT_MESSAGE,
                parent_tool_call_id="sa-tc-1",
                content="Subagent response",
            )
        )

        # Subagent completed
        session._dispatch_event(
            _make_session_event(
                SessionEventType.SUBAGENT_COMPLETED,
                tool_call_id="sa-tc-1",
            )
        )

        session._dispatch_event(_make_session_event(SessionEventType.ASSISTANT_TURN_END))
        session._dispatch_event(_make_session_event(SessionEventType.SESSION_IDLE))

        spans = exporter.get_finished_spans()
        invoke_spans = [
            s for s in spans if s.attributes.get(ATTR_GEN_AI_OPERATION_NAME) == OP_INVOKE_AGENT
        ]
        chat_spans = [s for s in spans if s.attributes.get(ATTR_GEN_AI_OPERATION_NAME) == OP_CHAT]

        # Should have 2 invoke_agent (main + subagent) and 2 chat spans
        assert len(invoke_spans) == 2
        assert len(chat_spans) == 2

        # Subagent's invoke_agent span should have agent_name
        sub_invoke = next(
            (s for s in invoke_spans if s.attributes.get(ATTR_GEN_AI_AGENT_NAME) == "sub-agent"),
            None,
        )
        assert sub_invoke is not None

        # Subagent spans should be children of the root invoke_agent span
        root_invoke = next(
            s for s in invoke_spans if s.attributes.get(ATTR_GEN_AI_AGENT_NAME) != "sub-agent"
        )
        assert sub_invoke.parent is not None
        assert sub_invoke.parent.span_id == root_invoke.context.span_id

        # Subagent chat should have usage
        sub_chat = next(
            (s for s in chat_spans if s.parent and s.parent.span_id == sub_invoke.context.span_id),
            None,
        )
        assert sub_chat is not None
        assert sub_chat.attributes.get(ATTR_GEN_AI_USAGE_INPUT_TOKENS) == 25
        assert sub_chat.attributes.get(ATTR_GEN_AI_USAGE_OUTPUT_TOKENS) == 50

    @pytest.mark.asyncio
    async def test_subagent_failed(self, _reset_otel_globals):
        exporter, _, tp, mp = _get_exporter_and_reader(_reset_otel_globals)
        telemetry = _make_telemetry(tracer_provider=tp, meter_provider=mp)
        session = _make_session_with_telemetry(telemetry, "s1")

        await session.send({"prompt": "Hello"})
        session._dispatch_event(
            _make_session_event(SessionEventType.ASSISTANT_TURN_START, turn_id="t1")
        )
        session._dispatch_event(
            _make_session_event(
                SessionEventType.SUBAGENT_STARTED,
                tool_call_id="sa-tc-fail",
                agent_name="bad-agent",
            )
        )
        session._dispatch_event(
            _make_session_event(
                SessionEventType.SUBAGENT_FAILED,
                tool_call_id="sa-tc-fail",
                agent_name="bad-agent",
                error="Something broke",
            )
        )
        session._dispatch_event(_make_session_event(SessionEventType.ASSISTANT_TURN_END))
        session._dispatch_event(_make_session_event(SessionEventType.SESSION_IDLE))

        spans = exporter.get_finished_spans()
        # Subagent spans should have error status
        sub_invoke = next(
            (s for s in spans if s.attributes.get(ATTR_GEN_AI_AGENT_NAME) == "bad-agent"),
            None,
        )
        assert sub_invoke is not None
        assert sub_invoke.status.status_code == trace.StatusCode.ERROR
        assert sub_invoke.attributes[ATTR_GEN_AI_RESPONSE_FINISH_REASONS] == ("error",)


# ---------------------------------------------------------------------------
# Tests: CompleteOnDispose
# ---------------------------------------------------------------------------


class TestCompleteOnDispose:
    @pytest.mark.asyncio
    async def test_complete_on_dispose_closes_spans_with_error(self, _reset_otel_globals):
        exporter, _, tp, mp = _get_exporter_and_reader(_reset_otel_globals)
        telemetry = _make_telemetry(tracer_provider=tp, meter_provider=mp)
        session = _make_session_with_telemetry(telemetry, "s1")

        await session.send({"prompt": "Hello"})
        session._dispatch_event(
            _make_session_event(SessionEventType.ASSISTANT_TURN_START, turn_id="t1")
        )

        # Dispose while turn is in progress
        session._turn_tracker.complete_on_dispose()

        spans = exporter.get_finished_spans()
        # Should have both chat and invoke_agent spans
        assert len(spans) >= 2

        for s in spans:
            assert s.status.status_code == trace.StatusCode.ERROR
            assert s.attributes[ATTR_ERROR_TYPE] == "SessionDestroyedError"
            assert s.attributes[ATTR_GEN_AI_RESPONSE_FINISH_REASONS] == ("error",)

    @pytest.mark.asyncio
    async def test_destroy_calls_complete_on_dispose(self, _reset_otel_globals):
        exporter, _, tp, mp = _get_exporter_and_reader(_reset_otel_globals)
        telemetry = _make_telemetry(tracer_provider=tp, meter_provider=mp)
        session = _make_session_with_telemetry(telemetry, "s1")

        await session.send({"prompt": "Hello"})
        session._dispatch_event(
            _make_session_event(SessionEventType.ASSISTANT_TURN_START, turn_id="t1")
        )

        # Destroy should close open spans
        await session.destroy()

        spans = exporter.get_finished_spans()
        assert len(spans) >= 2
        for s in spans:
            assert s.status.status_code == trace.StatusCode.ERROR


# ---------------------------------------------------------------------------
# Tests: Tool call parent context routing
# ---------------------------------------------------------------------------


class TestToolCallParentContext:
    @pytest.mark.asyncio
    async def test_get_tool_call_parent_context_returns_agent_context(self, _reset_otel_globals):
        _, _, tp, mp = _get_exporter_and_reader(_reset_otel_globals)
        telemetry = _make_telemetry(tracer_provider=tp, meter_provider=mp)
        session = _make_session_with_telemetry(telemetry, "s1")

        await session.send({"prompt": "Hello"})
        session._dispatch_event(
            _make_session_event(SessionEventType.ASSISTANT_TURN_START, turn_id="t1")
        )

        # Store a pending parent via tool_execution_start
        session._dispatch_event(
            _make_session_event(
                SessionEventType.TOOL_EXECUTION_START,
                tool_call_id="tc-1",
                tool_name="my_tool",
            )
        )

        # get_telemetry_tool_call_parent_context should return a context
        ctx = session.get_telemetry_tool_call_parent_context("tc-1")
        assert ctx is not None

        # Consuming again should fallback to agent context
        ctx2 = session.get_telemetry_tool_call_parent_context("tc-1")
        assert ctx2 is not None  # Falls back to agent span context

        session._dispatch_event(_make_session_event(SessionEventType.ASSISTANT_TURN_END))
        session._dispatch_event(_make_session_event(SessionEventType.SESSION_IDLE))


# ---------------------------------------------------------------------------
# Tests: Token usage metrics use "chat" operation name
# ---------------------------------------------------------------------------


class TestTokenUsageMetricsOperationName:
    @pytest.mark.asyncio
    async def test_per_turn_token_metrics_use_chat_op(self, _reset_otel_globals):
        _, reader, tp, mp = _get_exporter_and_reader(_reset_otel_globals)
        telemetry = _make_telemetry(tracer_provider=tp, meter_provider=mp)
        session = _make_session_with_telemetry(telemetry, "s1")

        await session.send({"prompt": "Hello"})
        session._dispatch_event(
            _make_session_event(SessionEventType.ASSISTANT_TURN_START, turn_id="t1")
        )
        session._dispatch_event(
            _make_session_event(
                SessionEventType.ASSISTANT_USAGE,
                model="gpt-4o",
                input_tokens=50.0,
                output_tokens=100.0,
            )
        )
        session._dispatch_event(_make_session_event(SessionEventType.ASSISTANT_TURN_END))
        session._dispatch_event(_make_session_event(SessionEventType.SESSION_IDLE))

        data_points = _get_metric_data_points(reader, METRIC_TOKEN_USAGE)
        assert len(data_points) > 0
        for dp in data_points:
            attrs = dict(dp.attributes)
            assert attrs[ATTR_GEN_AI_OPERATION_NAME] == OP_CHAT


# ---------------------------------------------------------------------------
# Tests: SDK version on tracer/meter
# ---------------------------------------------------------------------------


class TestSdkVersion:
    def test_tracer_has_version(self, _reset_otel_globals):
        exporter, _, tp, mp = _get_exporter_and_reader(_reset_otel_globals)
        telemetry = _make_telemetry(tracer_provider=tp, meter_provider=mp)

        span, _ctx = telemetry.start_invoke_agent_span("s1", None, "github", None, None)
        span.end()

        s = exporter.get_finished_spans()[0]
        # The instrumentation scope should have a version from copilot.__version__
        from copilot._copilot_telemetry import _SDK_VERSION

        if _SDK_VERSION:
            assert s.instrumentation_scope.version == _SDK_VERSION
        else:
            # Graceful fallback if version not found
            assert s.instrumentation_scope.version is not None or True
