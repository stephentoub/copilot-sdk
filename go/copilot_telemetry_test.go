/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------------------------------------------*/

package copilot

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	sdkmetric "go.opentelemetry.io/otel/sdk/metric"
	"go.opentelemetry.io/otel/sdk/metric/metricdata"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
)

// setupTestTelemetry sets up in-memory trace and metric exporters for testing,
// installs them as the global OTel providers, and returns cleanup/read helpers.
func setupTestTelemetry(t *testing.T, config *TelemetryConfig) (
	ct *copilotTelemetry,
	spanExporter *tracetest.InMemoryExporter,
	metricReader *sdkmetric.ManualReader,
) {
	t.Helper()

	spanExporter = tracetest.NewInMemoryExporter()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSyncer(spanExporter))

	metricReader = sdkmetric.NewManualReader()
	mp := sdkmetric.NewMeterProvider(sdkmetric.WithReader(metricReader))

	otel.SetTracerProvider(tp)
	otel.SetMeterProvider(mp)

	t.Cleanup(func() {
		_ = tp.Shutdown(context.Background())
		_ = mp.Shutdown(context.Background())
	})

	ct = newCopilotTelemetry(config)
	return ct, spanExporter, metricReader
}

func collectMetrics(t *testing.T, reader *sdkmetric.ManualReader) metricdata.ResourceMetrics {
	t.Helper()
	var rm metricdata.ResourceMetrics
	if err := reader.Collect(context.Background(), &rm); err != nil {
		t.Fatalf("Failed to collect metrics: %v", err)
	}
	return rm
}

func findSpanAttr(span tracetest.SpanStub, key string) *attribute.KeyValue {
	for _, a := range span.Attributes {
		if string(a.Key) == key {
			return &a
		}
	}
	return nil
}

func findMetricByName(rm metricdata.ResourceMetrics, name string) *metricdata.Metrics {
	for _, sm := range rm.ScopeMetrics {
		for i, m := range sm.Metrics {
			if m.Name == name {
				return &sm.Metrics[i]
			}
		}
	}
	return nil
}

func ptrString(s string) *string  { return &s }
func ptrFloat(f float64) *float64 { return &f }
func ptrBool(b bool) *bool        { return &b }

func findEventByName(span tracetest.SpanStub, name string) *sdktrace.Event {
	for i, e := range span.Events {
		if e.Name == name {
			return &span.Events[i]
		}
	}
	return nil
}

func findEventAttr(event *sdktrace.Event, key string) *attribute.KeyValue {
	for _, a := range event.Attributes {
		if string(a.Key) == key {
			return &a
		}
	}
	return nil
}

// --- Tests ---

func TestOpenTelemetry_NormalizeProviderName(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"openai", "openai"},
		{"OpenAI", "openai"},
		{"azure", "azure.ai.openai"},
		{"Azure", "azure.ai.openai"},
		{"anthropic", "anthropic"},
		{"Anthropic", "anthropic"},
		{"unknown", "github"},
		{"", "github"},
		{"  openai  ", "openai"},
	}
	for _, tc := range tests {
		t.Run(tc.input, func(t *testing.T) {
			result := normalizeProviderName(tc.input)
			if result != tc.expected {
				t.Errorf("normalizeProviderName(%q) = %q, want %q", tc.input, result, tc.expected)
			}
		})
	}
}

func TestOpenTelemetry_ParseServerAddress(t *testing.T) {
	tests := []struct {
		input        string
		expectedHost string
		expectedPort int
	}{
		{"https://api.example.com:8080/v1", "api.example.com", 8080},
		{"http://localhost:3000", "localhost", 3000},
		{"https://api.openai.com/v1", "api.openai.com", 443},
		{"http://example.com/path", "example.com", 80},
		{"", "", 0},
		{"not-a-url", "", 0},
	}
	for _, tc := range tests {
		t.Run(tc.input, func(t *testing.T) {
			host, port := parseServerAddress(tc.input)
			if host != tc.expectedHost {
				t.Errorf("parseServerAddress(%q) host = %q, want %q", tc.input, host, tc.expectedHost)
			}
			if port != tc.expectedPort {
				t.Errorf("parseServerAddress(%q) port = %d, want %d", tc.input, port, tc.expectedPort)
			}
		})
	}
}

func TestOpenTelemetry_InvokeAgentSpan(t *testing.T) {
	ct, exporter, _ := setupTestTelemetry(t, &TelemetryConfig{})

	ctx, span := ct.startInvokeAgentSpan(
		context.Background(),
		"session-123",
		"gpt-4",
		"openai",
		"api.openai.com",
		443,
		"",
		"",
	)
	_ = ctx
	span.End()

	spans := exporter.GetSpans()
	if len(spans) != 1 {
		t.Fatalf("Expected 1 span, got %d", len(spans))
	}
	s := spans[0]
	if s.Name != otelInvokeAgent {
		t.Errorf("Span name = %q, want %q", s.Name, otelInvokeAgent)
	}
	if s.SpanKind != 3 { // trace.SpanKindClient = 3
		t.Errorf("SpanKind = %d, want Client(3)", s.SpanKind)
	}

	assertAttr := func(key, expected string) {
		for _, a := range s.Attributes {
			if string(a.Key) == key {
				if a.Value.AsString() != expected {
					t.Errorf("Attribute %s = %q, want %q", key, a.Value.AsString(), expected)
				}
				return
			}
		}
		t.Errorf("Attribute %s not found", key)
	}

	assertAttr(otelGenAIOperationName, otelInvokeAgent)
	assertAttr(otelGenAIProviderName, "openai")
	assertAttr(otelGenAIAgentID, "session-123")
	assertAttr(otelGenAIConversationID, "session-123")
	assertAttr(otelGenAIRequestModel, "gpt-4")
	assertAttr(otelServerAddress, "api.openai.com")
}

func TestOpenTelemetry_ChatSpan(t *testing.T) {
	ct, exporter, _ := setupTestTelemetry(t, &TelemetryConfig{})

	ctx, _ := ct.startInvokeAgentSpan(
		context.Background(), "s1", "gpt-4", "openai", "api.openai.com", 443, "", "",
	)
	_, chatSpan := ct.startChatSpan(ctx, "gpt-4", "openai", "api.openai.com", 443, "s1")
	chatSpan.End()

	spans := exporter.GetSpans()
	if len(spans) < 1 {
		t.Fatalf("Expected at least 1 span, got %d", len(spans))
	}
	s := spans[0] // chat span ends first
	if s.Name != "chat gpt-4" {
		t.Errorf("Span name = %q, want %q", s.Name, "chat gpt-4")
	}

	assertAttr := func(key, expected string) {
		a := findSpanAttr(s, key)
		if a == nil {
			t.Errorf("Attribute %s not found", key)
			return
		}
		if a.Value.AsString() != expected {
			t.Errorf("Attribute %s = %q, want %q", key, a.Value.AsString(), expected)
		}
	}

	assertAttr(otelGenAIOperationName, otelChat)
	assertAttr(otelGenAIProviderName, "openai")
	assertAttr(otelGenAIConversationID, "s1")
}

func TestOpenTelemetry_ExecuteToolSpan(t *testing.T) {
	ct, exporter, _ := setupTestTelemetry(t, &TelemetryConfig{})

	_, span := ct.startExecuteToolSpan(
		context.Background(),
		"get_weather",
		"call-456",
		"Gets current weather",
		map[string]any{"location": "Seattle"},
	)
	span.End()

	spans := exporter.GetSpans()
	if len(spans) != 1 {
		t.Fatalf("Expected 1 span, got %d", len(spans))
	}
	s := spans[0]
	if s.Name != "execute_tool get_weather" {
		t.Errorf("Span name = %q, want %q", s.Name, "execute_tool get_weather")
	}
	if s.SpanKind != 1 { // trace.SpanKindInternal = 1
		t.Errorf("SpanKind = %d, want Internal(1)", s.SpanKind)
	}

	assertAttr := func(key, expected string) {
		for _, a := range s.Attributes {
			if string(a.Key) == key {
				if a.Value.AsString() != expected {
					t.Errorf("Attribute %s = %q, want %q", key, a.Value.AsString(), expected)
				}
				return
			}
		}
		t.Errorf("Attribute %s not found", key)
	}

	assertAttr(otelGenAIOperationName, otelExecuteTool)
	assertAttr(otelGenAIToolName, "get_weather")
	assertAttr(otelGenAIToolCallID, "call-456")
	assertAttr(otelGenAIToolType, "function")
	assertAttr(otelGenAIToolDescription, "Gets current weather")
}

func TestOpenTelemetry_ExecuteToolSpanSensitiveData(t *testing.T) {
	ct, exporter, _ := setupTestTelemetry(t, &TelemetryConfig{
		EnableSensitiveData: Bool(true),
	})

	args := map[string]any{"location": "Seattle"}
	_, span := ct.startExecuteToolSpan(
		context.Background(),
		"get_weather",
		"call-456",
		"Gets weather",
		args,
	)
	ct.setExecuteToolResult(span, "Sunny, 72°F")
	span.End()

	spans := exporter.GetSpans()
	if len(spans) != 1 {
		t.Fatalf("Expected 1 span, got %d", len(spans))
	}
	s := spans[0]

	if findSpanAttr(s, otelGenAIToolCallArguments) == nil {
		t.Error("Expected tool call arguments attribute when sensitive data enabled")
	}
	if findSpanAttr(s, otelGenAIToolCallResult) == nil {
		t.Error("Expected tool call result attribute when sensitive data enabled")
	}
}

func TestOpenTelemetry_SensitiveDataGating(t *testing.T) {
	ct, exporter, _ := setupTestTelemetry(t, &TelemetryConfig{
		EnableSensitiveData: Bool(false),
	})

	args := map[string]any{"secret": "password123"}
	_, span := ct.startExecuteToolSpan(
		context.Background(),
		"tool",
		"call-1",
		"",
		args,
	)
	ct.setExecuteToolResult(span, "result")
	span.End()

	spans := exporter.GetSpans()
	s := spans[0]

	for _, a := range s.Attributes {
		if string(a.Key) == otelGenAIToolCallArguments {
			t.Error("Tool call arguments should NOT be present when sensitive data disabled")
		}
		if string(a.Key) == otelGenAIToolCallResult {
			t.Error("Tool call result should NOT be present when sensitive data disabled")
		}
	}
}

func TestOpenTelemetry_MetricRecording(t *testing.T) {
	ct, _, reader := setupTestTelemetry(t, &TelemetryConfig{})

	ctx := context.Background()

	ct.recordOperationDuration(ctx, 1.5, "gpt-4", "gpt-4", "openai", "api.openai.com", 443, nil, otelInvokeAgent)
	ct.tokenUsageHistogram.Record(ctx, 100)
	ct.recordTimeToFirstChunk(ctx, 0.5, "gpt-4", "gpt-4", "openai", "", 0)
	ct.recordTimePerOutputChunk(ctx, 0.1, "gpt-4", "gpt-4", "openai", "", 0)

	rm := collectMetrics(t, reader)

	if m := findMetricByName(rm, otelMetricOperationDuration); m == nil {
		t.Error("Expected operation duration metric")
	}
	if m := findMetricByName(rm, otelMetricTokenUsage); m == nil {
		t.Error("Expected token usage metric")
	}
	if m := findMetricByName(rm, otelMetricTimeToFirstChunk); m == nil {
		t.Error("Expected time to first chunk metric")
	}
	if m := findMetricByName(rm, otelMetricTimePerOutputChunk); m == nil {
		t.Error("Expected time per output chunk metric")
	}
}

func TestOpenTelemetry_NoTelemetryWhenNilConfig(t *testing.T) {
	session := &Session{
		SessionID:    "test-session",
		telemetry:    nil,
		toolHandlers: make(map[string]ToolHandler),
	}
	// This should not panic
	session.dispatchEvent(SessionEvent{Type: SessionIdle})
}

func TestOpenTelemetry_CustomSourceName(t *testing.T) {
	ct, exporter, _ := setupTestTelemetry(t, &TelemetryConfig{
		SourceName: "my.custom.source",
	})

	_, span := ct.startInvokeAgentSpan(context.Background(), "s1", "", "github", "", 0, "", "")
	span.End()

	spans := exporter.GetSpans()
	if len(spans) != 1 {
		t.Fatalf("Expected 1 span, got %d", len(spans))
	}

	s := spans[0]
	if s.InstrumentationScope.Name != "my.custom.source" {
		t.Errorf("InstrumentationScope.Name = %q, want %q", s.InstrumentationScope.Name, "my.custom.source")
	}
}

func TestOpenTelemetry_DefaultSourceName(t *testing.T) {
	ct, exporter, _ := setupTestTelemetry(t, &TelemetryConfig{})

	_, span := ct.startInvokeAgentSpan(context.Background(), "s1", "", "github", "", 0, "", "")
	span.End()

	spans := exporter.GetSpans()
	s := spans[0]
	if s.InstrumentationScope.Name != otelDefaultSourceName {
		t.Errorf("Default source name = %q, want %q", s.InstrumentationScope.Name, otelDefaultSourceName)
	}
}

func TestOpenTelemetry_AgentMetadata(t *testing.T) {
	ct, exporter, _ := setupTestTelemetry(t, &TelemetryConfig{})

	_, span := ct.startInvokeAgentSpan(
		context.Background(), "s1", "gpt-4", "github", "", 0, "my-agent", "A helpful agent",
	)
	span.End()

	spans := exporter.GetSpans()
	s := spans[0]

	if a := findSpanAttr(s, otelGenAIAgentName); a == nil || a.Value.AsString() != "my-agent" {
		t.Error("Expected gen_ai.agent.name = 'my-agent'")
	}
	if a := findSpanAttr(s, otelGenAIAgentDescription); a == nil || a.Value.AsString() != "A helpful agent" {
		t.Error("Expected gen_ai.agent.description = 'A helpful agent'")
	}
}

func TestOpenTelemetry_InvokeAgentSpan_ServerPort(t *testing.T) {
	ct, exporter, _ := setupTestTelemetry(t, &TelemetryConfig{})

	_, span := ct.startInvokeAgentSpan(
		context.Background(), "session-123", "gpt-4", "openai", "api.openai.com", 443, "", "",
	)
	span.End()

	spans := exporter.GetSpans()
	s := spans[0]

	a := findSpanAttr(s, otelServerPort)
	if a == nil {
		t.Error("Expected server.port attribute")
		return
	}
	if a.Value.AsInt64() != 443 {
		t.Errorf("server.port = %d, want 443", a.Value.AsInt64())
	}
}

func TestOpenTelemetry_EnvVarOverride(t *testing.T) {
	t.Setenv(otelCaptureMessageContentEnvVar, "true")

	ct, exporter, _ := setupTestTelemetry(t, &TelemetryConfig{})

	args := map[string]any{"secret": "password123"}
	_, span := ct.startExecuteToolSpan(
		context.Background(), "sensitive_tool", "call-1", "Does sensitive thing", args,
	)
	span.End()

	spans := exporter.GetSpans()
	s := spans[0]

	if findSpanAttr(s, otelGenAIToolCallArguments) == nil {
		t.Error("Expected tool call arguments attribute when OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true")
	}
}

// ============================================================================
// Two-level span hierarchy tests
// ============================================================================

func newTestSession(ct *copilotTelemetry, streaming bool) *Session {
	session := &Session{
		SessionID:        "session-test",
		telemetry:        ct,
		toolHandlers:     make(map[string]ToolHandler),
		toolDescriptions: make(map[string]string),
	}
	session.configureTelemetryContext("gpt-4", &ProviderConfig{
		Type:    "openai",
		BaseURL: "https://api.openai.com:443/v1",
	}, &SystemMessageConfig{Content: "Be helpful"}, nil, streaming, "", "")
	return session
}

func TestOpenTelemetry_TwoLevelSpanHierarchy(t *testing.T) {
	ct, exporter, reader := setupTestTelemetry(t, &TelemetryConfig{EnableSensitiveData: Bool(true)})

	session := newTestSession(ct, false)

	// Simulate send
	session.turnTracker.beginSend(context.Background(), "Hello")

	// AssistantTurnStart creates the chat child span
	session.dispatchEvent(SessionEvent{
		Type: AssistantTurnStart,
		Data: Data{TurnID: ptrString("turn-1"), InteractionID: ptrString("int-1")},
	})

	// Assistant responds
	session.dispatchEvent(SessionEvent{
		Type: AssistantMessage,
		Data: Data{Content: ptrString("Hi there!")},
	})

	// Usage
	session.dispatchEvent(SessionEvent{
		Type: AssistantUsage,
		Data: Data{
			Model:        ptrString("gpt-4-0613"),
			APICallID:    ptrString("api-1"),
			InputTokens:  ptrFloat(10),
			OutputTokens: ptrFloat(20),
		},
	})

	// Turn end
	session.dispatchEvent(SessionEvent{Type: AssistantTurnEnd})

	// Session idle completes the invoke_agent span
	session.dispatchEvent(SessionEvent{Type: SessionIdle})

	spans := exporter.GetSpans()
	if len(spans) != 2 {
		t.Fatalf("Expected 2 spans (chat + invoke_agent), got %d", len(spans))
	}

	// First ended span is the chat span
	chatSpan := spans[0]
	if a := findSpanAttr(chatSpan, otelGenAIOperationName); a == nil || a.Value.AsString() != otelChat {
		t.Error("Expected chat span operation name")
	}
	if a := findSpanAttr(chatSpan, otelGenAIUsageInputTokens); a == nil || a.Value.AsInt64() != 10 {
		t.Error("Expected chat span input tokens = 10")
	}
	if a := findSpanAttr(chatSpan, otelGenAIUsageOutputTokens); a == nil || a.Value.AsInt64() != 20 {
		t.Error("Expected chat span output tokens = 20")
	}
	if a := findSpanAttr(chatSpan, otelCopilotTurnID); a == nil || a.Value.AsString() != "turn-1" {
		t.Error("Expected chat span turn_id = turn-1")
	}
	if a := findSpanAttr(chatSpan, otelCopilotInteractionID); a == nil || a.Value.AsString() != "int-1" {
		t.Error("Expected chat span interaction_id = int-1")
	}

	// Second ended span is the invoke_agent span
	agentSpan := spans[1]
	if a := findSpanAttr(agentSpan, otelGenAIOperationName); a == nil || a.Value.AsString() != otelInvokeAgent {
		t.Error("Expected invoke_agent span operation name")
	}
	// Accumulated usage on invoke_agent
	if a := findSpanAttr(agentSpan, otelGenAIUsageInputTokens); a == nil || a.Value.AsInt64() != 10 {
		t.Error("Expected invoke_agent accumulated input tokens = 10")
	}

	// Verify metrics recorded
	rm := collectMetrics(t, reader)
	if m := findMetricByName(rm, otelMetricOperationDuration); m == nil {
		t.Error("Expected operation duration metric")
	}
	if m := findMetricByName(rm, otelMetricTokenUsage); m == nil {
		t.Error("Expected token usage metric")
	}
}

func TestOpenTelemetry_MultiTurnAccumulatedUsage(t *testing.T) {
	ct, exporter, _ := setupTestTelemetry(t, &TelemetryConfig{})

	session := newTestSession(ct, false)
	session.turnTracker.beginSend(context.Background(), "Hello")

	// Turn 1
	session.dispatchEvent(SessionEvent{Type: AssistantTurnStart, Data: Data{}})
	session.dispatchEvent(SessionEvent{Type: AssistantUsage, Data: Data{
		Model: ptrString("gpt-4"), InputTokens: ptrFloat(100), OutputTokens: ptrFloat(50),
	}})
	session.dispatchEvent(SessionEvent{Type: AssistantTurnEnd})

	// Turn 2
	session.dispatchEvent(SessionEvent{Type: AssistantTurnStart, Data: Data{}})
	session.dispatchEvent(SessionEvent{Type: AssistantUsage, Data: Data{
		Model: ptrString("gpt-4"), InputTokens: ptrFloat(200), OutputTokens: ptrFloat(100),
	}})
	session.dispatchEvent(SessionEvent{Type: AssistantTurnEnd})

	// Complete
	session.dispatchEvent(SessionEvent{Type: SessionIdle})

	spans := exporter.GetSpans()
	if len(spans) != 3 {
		t.Fatalf("Expected 3 spans (2 chat + 1 invoke_agent), got %d", len(spans))
	}

	// The invoke_agent span (last) should have accumulated tokens
	agentSpan := spans[2]
	if a := findSpanAttr(agentSpan, otelGenAIUsageInputTokens); a == nil || a.Value.AsInt64() != 300 {
		t.Errorf("Expected invoke_agent accumulated input tokens = 300, got %v", a)
	}
	if a := findSpanAttr(agentSpan, otelGenAIUsageOutputTokens); a == nil || a.Value.AsInt64() != 150 {
		t.Errorf("Expected invoke_agent accumulated output tokens = 150, got %v", a)
	}
}

func TestOpenTelemetry_VendorPrefixedAttributes(t *testing.T) {
	ct, exporter, _ := setupTestTelemetry(t, &TelemetryConfig{})

	session := newTestSession(ct, false)
	session.turnTracker.beginSend(context.Background(), "Hello")

	session.dispatchEvent(SessionEvent{Type: AssistantTurnStart, Data: Data{}})
	session.dispatchEvent(SessionEvent{Type: AssistantUsage, Data: Data{
		Model:        ptrString("gpt-4"),
		InputTokens:  ptrFloat(10),
		OutputTokens: ptrFloat(20),
		Cost:         ptrFloat(0.005),
		Duration:     ptrFloat(1.5),
		Initiator:    ptrString("user"),
		CopilotUsage: &CopilotUsage{TotalNanoAiu: 42.0},
	}})
	session.dispatchEvent(SessionEvent{Type: AssistantTurnEnd})
	session.dispatchEvent(SessionEvent{Type: SessionIdle})

	spans := exporter.GetSpans()
	chatSpan := spans[0]

	if a := findSpanAttr(chatSpan, otelCopilotCost); a == nil || a.Value.AsFloat64() != 0.005 {
		t.Errorf("Expected copilot.cost = 0.005, got %v", a)
	}
	if a := findSpanAttr(chatSpan, otelCopilotServerDuration); a == nil || a.Value.AsFloat64() != 1.5 {
		t.Errorf("Expected copilot.server_duration = 1.5, got %v", a)
	}
	if a := findSpanAttr(chatSpan, otelCopilotInitiator); a == nil || a.Value.AsString() != "user" {
		t.Errorf("Expected copilot.initiator = 'user'")
	}
	if a := findSpanAttr(chatSpan, otelCopilotAIU); a == nil || a.Value.AsFloat64() != 42.0 {
		t.Errorf("Expected copilot.aiu = 42.0, got %v", a)
	}

	// Check accumulated cost/aiu on invoke_agent
	agentSpan := spans[1]
	if a := findSpanAttr(agentSpan, otelCopilotCost); a == nil || a.Value.AsFloat64() != 0.005 {
		t.Errorf("Expected invoke_agent copilot.cost = 0.005")
	}
	if a := findSpanAttr(agentSpan, otelCopilotAIU); a == nil || a.Value.AsFloat64() != 42.0 {
		t.Errorf("Expected invoke_agent copilot.aiu = 42.0")
	}
}

func TestOpenTelemetry_RichMessageContent(t *testing.T) {
	ct, exporter, _ := setupTestTelemetry(t, &TelemetryConfig{EnableSensitiveData: Bool(true)})

	session := newTestSession(ct, false)
	session.turnTracker.beginSend(context.Background(), "Hello")

	session.dispatchEvent(SessionEvent{Type: AssistantTurnStart, Data: Data{}})

	// Message with reasoning
	session.dispatchEvent(SessionEvent{
		Type: AssistantMessage,
		Data: Data{
			ReasoningText: ptrString("Let me think..."),
			Content:       ptrString("The answer is 42"),
		},
	})
	session.dispatchEvent(SessionEvent{Type: AssistantTurnEnd})
	session.dispatchEvent(SessionEvent{Type: SessionIdle})

	spans := exporter.GetSpans()
	chatSpan := spans[0]

	a := findSpanAttr(chatSpan, otelGenAIOutputMessages)
	if a == nil {
		t.Fatal("Expected gen_ai.output.messages attribute")
	}

	// Verify it contains reasoning part
	var msgs []map[string]any
	if err := json.Unmarshal([]byte(a.Value.AsString()), &msgs); err != nil {
		t.Fatalf("Failed to parse output messages: %v", err)
	}
	if len(msgs) != 1 {
		t.Fatalf("Expected 1 message, got %d", len(msgs))
	}
	parts, ok := msgs[0]["parts"].([]any)
	if !ok || len(parts) != 2 {
		t.Fatalf("Expected 2 parts (reasoning + text), got %v", msgs[0]["parts"])
	}
	part0 := parts[0].(map[string]any)
	if part0["type"] != "reasoning" || part0["content"] != "Let me think..." {
		t.Errorf("First part should be reasoning, got %v", part0)
	}
	part1 := parts[1].(map[string]any)
	if part1["type"] != "text" || part1["content"] != "The answer is 42" {
		t.Errorf("Second part should be text, got %v", part1)
	}
}

func TestOpenTelemetry_ToolCallParts(t *testing.T) {
	ct, exporter, _ := setupTestTelemetry(t, &TelemetryConfig{EnableSensitiveData: Bool(true)})

	session := newTestSession(ct, false)
	session.turnTracker.beginSend(context.Background(), "Hello")

	session.dispatchEvent(SessionEvent{Type: AssistantTurnStart, Data: Data{}})

	// Tool execution start
	session.dispatchEvent(SessionEvent{
		Type: ToolExecutionStart,
		Data: Data{
			ToolCallID: ptrString("tc-1"),
			ToolName:   ptrString("read_file"),
			Arguments:  map[string]any{"path": "/foo.txt"},
		},
	})

	// Tool execution complete
	session.dispatchEvent(SessionEvent{
		Type: ToolExecutionComplete,
		Data: Data{
			ToolCallID: ptrString("tc-1"),
			Result:     &Result{Content: "file contents"},
		},
	})

	session.dispatchEvent(SessionEvent{Type: AssistantTurnEnd})
	session.dispatchEvent(SessionEvent{Type: SessionIdle})

	spans := exporter.GetSpans()
	chatSpan := spans[0]

	// Check output messages contain tool_call
	outAttr := findSpanAttr(chatSpan, otelGenAIOutputMessages)
	if outAttr == nil {
		t.Fatal("Expected gen_ai.output.messages")
	}
	var outMsgs []map[string]any
	json.Unmarshal([]byte(outAttr.Value.AsString()), &outMsgs)
	if len(outMsgs) < 1 {
		t.Fatal("Expected at least 1 output message")
	}
	parts := outMsgs[0]["parts"].([]any)
	p0 := parts[0].(map[string]any)
	if p0["type"] != "tool_call" {
		t.Errorf("Expected tool_call part type, got %v", p0["type"])
	}

	// Check input messages contain tool_call_response
	inAttr := findSpanAttr(chatSpan, otelGenAIInputMessages)
	if inAttr == nil {
		t.Fatal("Expected gen_ai.input.messages")
	}
	var inMsgs []map[string]any
	json.Unmarshal([]byte(inAttr.Value.AsString()), &inMsgs)
	// First input msg = user "Hello", second = tool response
	foundToolResponse := false
	for _, m := range inMsgs {
		if m["role"] == "tool" {
			foundToolResponse = true
			break
		}
	}
	if !foundToolResponse {
		t.Error("Expected tool role in input messages")
	}
}

func TestOpenTelemetry_MCPServerToolTracking(t *testing.T) {
	ct, exporter, _ := setupTestTelemetry(t, &TelemetryConfig{EnableSensitiveData: Bool(true)})

	session := newTestSession(ct, false)
	session.turnTracker.beginSend(context.Background(), "Hello")

	session.dispatchEvent(SessionEvent{Type: AssistantTurnStart, Data: Data{}})

	// Server tool execution start
	session.dispatchEvent(SessionEvent{
		Type: ToolExecutionStart,
		Data: Data{
			ToolCallID:    ptrString("tc-mcp-1"),
			ToolName:      ptrString("mcp_search"),
			MCPServerName: ptrString("github-server"),
			Arguments:     map[string]any{"query": "test"},
		},
	})

	// Server tool execution complete
	session.dispatchEvent(SessionEvent{
		Type: ToolExecutionComplete,
		Data: Data{
			ToolCallID: ptrString("tc-mcp-1"),
			Result:     &Result{Content: "results"},
		},
	})

	session.dispatchEvent(SessionEvent{Type: AssistantTurnEnd})
	session.dispatchEvent(SessionEvent{Type: SessionIdle})

	spans := exporter.GetSpans()
	chatSpan := spans[0]

	// Check output messages contain server_tool_call with nested mcp
	outAttr := findSpanAttr(chatSpan, otelGenAIOutputMessages)
	if outAttr == nil {
		t.Fatal("Expected gen_ai.output.messages")
	}
	var outMsgs []map[string]any
	json.Unmarshal([]byte(outAttr.Value.AsString()), &outMsgs)
	if len(outMsgs) < 1 {
		t.Fatal("Expected at least 1 output message")
	}
	parts := outMsgs[0]["parts"].([]any)
	p0 := parts[0].(map[string]any)
	if p0["type"] != "server_tool_call" {
		t.Errorf("Expected server_tool_call type, got %v", p0["type"])
	}
	stc, ok := p0["server_tool_call"].(map[string]any)
	if !ok {
		t.Fatal("Expected server_tool_call nested object")
	}
	if stc["type"] != "mcp" {
		t.Errorf("Expected server_tool_call.type = 'mcp', got %v", stc["type"])
	}
	if stc["server_name"] != "github-server" {
		t.Errorf("Expected server_name = 'github-server', got %v", stc["server_name"])
	}
}

func TestOpenTelemetry_SessionTruncationEvent(t *testing.T) {
	ct, exporter, _ := setupTestTelemetry(t, &TelemetryConfig{})

	session := newTestSession(ct, false)
	session.turnTracker.beginSend(context.Background(), "Hello")
	session.dispatchEvent(SessionEvent{Type: AssistantTurnStart, Data: Data{}})

	session.dispatchEvent(SessionEvent{
		Type: SessionTruncation,
		Data: Data{
			TokenLimit:                      ptrFloat(8000),
			PreTruncationTokensInMessages:   ptrFloat(10000),
			PostTruncationTokensInMessages:  ptrFloat(7500),
			PreTruncationMessagesLength:     ptrFloat(50),
			PostTruncationMessagesLength:    ptrFloat(40),
			TokensRemovedDuringTruncation:   ptrFloat(2500),
			MessagesRemovedDuringTruncation: ptrFloat(10),
			PerformedBy:                     ptrString("system"),
		},
	})

	session.dispatchEvent(SessionEvent{Type: AssistantTurnEnd})
	session.dispatchEvent(SessionEvent{Type: SessionIdle})

	spans := exporter.GetSpans()
	chatSpan := spans[0]

	evt := findEventByName(chatSpan, otelCopilotSessionTruncation)
	if evt == nil {
		t.Fatal("Expected session.truncation span event")
	}
	if a := findEventAttr(evt, otelCopilotTokenLimit); a == nil || a.Value.AsInt64() != 8000 {
		t.Errorf("Expected token_limit = 8000")
	}
	if a := findEventAttr(evt, otelCopilotPerformedBy); a == nil || a.Value.AsString() != "system" {
		t.Errorf("Expected performed_by = 'system'")
	}
}

func TestOpenTelemetry_CompactionEvents(t *testing.T) {
	ct, exporter, _ := setupTestTelemetry(t, &TelemetryConfig{})

	session := newTestSession(ct, false)
	session.turnTracker.beginSend(context.Background(), "Hello")
	session.dispatchEvent(SessionEvent{Type: AssistantTurnStart, Data: Data{}})

	session.dispatchEvent(SessionEvent{Type: SessionCompactionStart})
	session.dispatchEvent(SessionEvent{
		Type: SessionCompactionComplete,
		Data: Data{
			Success:              ptrBool(true),
			PreCompactionTokens:  ptrFloat(10000),
			PostCompactionTokens: ptrFloat(5000),
			TokensRemoved:        ptrFloat(5000),
			MessagesRemoved:      ptrFloat(20),
		},
	})

	session.dispatchEvent(SessionEvent{Type: AssistantTurnEnd})
	session.dispatchEvent(SessionEvent{Type: SessionIdle})

	spans := exporter.GetSpans()
	chatSpan := spans[0]

	startEvt := findEventByName(chatSpan, otelCopilotSessionCompactionStart)
	if startEvt == nil {
		t.Error("Expected compaction_start span event")
	}

	completeEvt := findEventByName(chatSpan, otelCopilotSessionCompactionComplete)
	if completeEvt == nil {
		t.Fatal("Expected compaction_complete span event")
	}
	if a := findEventAttr(completeEvt, otelCopilotSuccess); a == nil || !a.Value.AsBool() {
		t.Error("Expected success = true")
	}
}

func TestOpenTelemetry_SkillInvokedEvent(t *testing.T) {
	ct, exporter, _ := setupTestTelemetry(t, &TelemetryConfig{EnableSensitiveData: Bool(true)})

	session := newTestSession(ct, false)
	session.turnTracker.beginSend(context.Background(), "Hello")
	session.dispatchEvent(SessionEvent{Type: AssistantTurnStart, Data: Data{}})

	session.dispatchEvent(SessionEvent{
		Type: SkillInvoked,
		Data: Data{
			Name:          ptrString("code-review"),
			Path:          ptrString("/skills/code-review"),
			PluginName:    ptrString("my-plugin"),
			PluginVersion: ptrString("1.0.0"),
			Content:       ptrString("skill content"),
		},
	})

	session.dispatchEvent(SessionEvent{Type: AssistantTurnEnd})
	session.dispatchEvent(SessionEvent{Type: SessionIdle})

	spans := exporter.GetSpans()
	chatSpan := spans[0]

	evt := findEventByName(chatSpan, otelCopilotSkillInvoked)
	if evt == nil {
		t.Fatal("Expected skill.invoked span event")
	}
	if a := findEventAttr(evt, otelCopilotSkillName); a == nil || a.Value.AsString() != "code-review" {
		t.Error("Expected skill name = 'code-review'")
	}
	if a := findEventAttr(evt, otelCopilotSkillPath); a == nil || a.Value.AsString() != "/skills/code-review" {
		t.Error("Expected skill path")
	}
	if a := findEventAttr(evt, otelCopilotSkillPluginName); a == nil || a.Value.AsString() != "my-plugin" {
		t.Error("Expected plugin name")
	}
	if a := findEventAttr(evt, otelCopilotSkillContent); a == nil || a.Value.AsString() != "skill content" {
		t.Error("Expected skill content (sensitive)")
	}
}

func TestOpenTelemetry_SessionErrorCompletesTurn(t *testing.T) {
	ct, exporter, _ := setupTestTelemetry(t, &TelemetryConfig{})

	session := newTestSession(ct, false)
	session.turnTracker.beginSend(context.Background(), "")

	errMsg := "model rate limited"
	session.dispatchEvent(SessionEvent{
		Type: SessionError,
		Data: Data{Message: &errMsg},
	})

	spans := exporter.GetSpans()
	if len(spans) != 1 {
		t.Fatalf("Expected 1 span, got %d", len(spans))
	}

	s := spans[0]
	if a := findSpanAttr(s, otelErrorType); a == nil {
		t.Error("Expected error.type attribute on error span")
	} else if a.Value.AsString() != "SessionTurnError" {
		t.Errorf("Expected error.type = SessionTurnError, got %s", a.Value.AsString())
	}
}

func TestOpenTelemetry_CompleteOnDispose(t *testing.T) {
	ct, exporter, _ := setupTestTelemetry(t, &TelemetryConfig{})

	session := newTestSession(ct, false)
	session.turnTracker.beginSend(context.Background(), "Hello")
	session.dispatchEvent(SessionEvent{Type: AssistantTurnStart, Data: Data{}})

	// Simulate dispose while turn is active
	session.turnTracker.completeOnDispose()

	spans := exporter.GetSpans()
	// Should have chat span + invoke_agent span, both with error
	if len(spans) < 1 {
		t.Fatalf("Expected at least 1 span after dispose, got %d", len(spans))
	}

	// Check that spans have error status with correct error type
	for _, s := range spans {
		if a := findSpanAttr(s, otelErrorType); a == nil {
			t.Errorf("Span %q should have error.type after dispose", s.Name)
		} else if a.Value.AsString() != "SessionDestroyedError" {
			t.Errorf("Span %q expected error.type = SessionDestroyedError, got %s", s.Name, a.Value.AsString())
		}
	}
}

func TestOpenTelemetry_StreamingChunkMetrics(t *testing.T) {
	ct, _, reader := setupTestTelemetry(t, &TelemetryConfig{})

	session := newTestSession(ct, true) // streaming=true
	session.turnTracker.beginSend(context.Background(), "")

	// Start a chat turn
	session.dispatchEvent(SessionEvent{Type: AssistantTurnStart, Data: Data{}})

	// Simulate streaming deltas
	delta1 := "Hello"
	session.dispatchEvent(SessionEvent{
		Type: AssistantMessageDelta,
		Data: Data{DeltaContent: &delta1},
	})

	time.Sleep(10 * time.Millisecond)

	delta2 := " world"
	session.dispatchEvent(SessionEvent{
		Type: AssistantMessageDelta,
		Data: Data{DeltaContent: &delta2},
	})

	session.dispatchEvent(SessionEvent{Type: AssistantTurnEnd})
	session.dispatchEvent(SessionEvent{Type: SessionIdle})

	rm := collectMetrics(t, reader)

	if m := findMetricByName(rm, otelMetricTimeToFirstChunk); m == nil {
		t.Error("Expected time_to_first_chunk metric for streaming")
	}
	if m := findMetricByName(rm, otelMetricTimePerOutputChunk); m == nil {
		t.Error("Expected time_per_output_chunk metric for streaming")
	}
}

func TestOpenTelemetry_SubagentSpans(t *testing.T) {
	ct, exporter, _ := setupTestTelemetry(t, &TelemetryConfig{})

	session := newTestSession(ct, false)
	session.turnTracker.beginSend(context.Background(), "Hello")
	session.dispatchEvent(SessionEvent{Type: AssistantTurnStart, Data: Data{}})

	// Subagent started
	session.dispatchEvent(SessionEvent{
		Type: SubagentStarted,
		Data: Data{
			ToolCallID:       ptrString("sub-tc-1"),
			AgentName:        ptrString("code-reviewer"),
			AgentDescription: ptrString("Reviews code"),
		},
	})

	// Subagent receives usage (routed by parentToolCallId)
	session.dispatchEvent(SessionEvent{
		Type: AssistantUsage,
		Data: Data{
			ParentToolCallID: ptrString("sub-tc-1"),
			Model:            ptrString("gpt-4"),
			InputTokens:      ptrFloat(50),
			OutputTokens:     ptrFloat(30),
		},
	})

	// Subagent sends message
	session.dispatchEvent(SessionEvent{
		Type: AssistantMessage,
		Data: Data{
			ParentToolCallID: ptrString("sub-tc-1"),
			Content:          ptrString("LGTM!"),
		},
	})

	// Subagent completed
	session.dispatchEvent(SessionEvent{
		Type: SubagentCompleted,
		Data: Data{ToolCallID: ptrString("sub-tc-1")},
	})

	session.dispatchEvent(SessionEvent{Type: AssistantTurnEnd})
	session.dispatchEvent(SessionEvent{Type: SessionIdle})

	spans := exporter.GetSpans()
	// Should have: subagent chat, subagent invoke_agent, main chat, main invoke_agent
	if len(spans) < 4 {
		t.Fatalf("Expected at least 4 spans, got %d", len(spans))
	}

	// Find the subagent invoke_agent span (has agent name)
	var subInvokeSpan *tracetest.SpanStub
	for i, s := range spans {
		if a := findSpanAttr(s, otelGenAIAgentName); a != nil && a.Value.AsString() == "code-reviewer" {
			subInvokeSpan = &spans[i]
			break
		}
	}
	if subInvokeSpan == nil {
		t.Fatal("Expected subagent invoke_agent span with agent name 'code-reviewer'")
	}
	if a := findSpanAttr(*subInvokeSpan, otelGenAIUsageInputTokens); a == nil || a.Value.AsInt64() != 50 {
		t.Error("Expected subagent invoke_agent input tokens = 50")
	}
}

func TestOpenTelemetry_ToolCallParentContext(t *testing.T) {
	ct, _, _ := setupTestTelemetry(t, &TelemetryConfig{})

	session := newTestSession(ct, false)
	session.turnTracker.beginSend(context.Background(), "Hello")
	session.dispatchEvent(SessionEvent{Type: AssistantTurnStart, Data: Data{}})

	// Simulate tool execution start to register pending parent
	session.dispatchEvent(SessionEvent{
		Type: ToolExecutionStart,
		Data: Data{
			ToolCallID: ptrString("tc-1"),
			ToolName:   ptrString("read_file"),
		},
	})

	// getToolCallParentContext should return a non-nil context
	ctx := session.getToolCallParentContext("tc-1")
	if ctx == nil {
		t.Error("Expected non-nil tool call parent context")
	}

	// Second call should still return the agent context (one-time consumption)
	ctx2 := session.getToolCallParentContext("tc-1")
	if ctx2 == nil {
		t.Error("Expected fallback to agent span context")
	}

	session.dispatchEvent(SessionEvent{Type: AssistantTurnEnd})
	session.dispatchEvent(SessionEvent{Type: SessionIdle})
}

func TestOpenTelemetry_ModelChangeUpdatesResponseModel(t *testing.T) {
	ct, _, _ := setupTestTelemetry(t, &TelemetryConfig{})

	session := newTestSession(ct, false)
	session.turnTracker.beginSend(context.Background(), "")
	session.dispatchEvent(SessionEvent{Type: AssistantTurnStart, Data: Data{}})

	newModel := "gpt-4-turbo"
	session.dispatchEvent(SessionEvent{
		Type: SessionModelChange,
		Data: Data{NewModel: &newModel},
	})

	session.turnTracker.mu.Lock()
	if session.turnTracker.turnResponseModel != "gpt-4-turbo" {
		t.Errorf("Expected response model to be updated to gpt-4-turbo, got %q", session.turnTracker.turnResponseModel)
	}
	session.turnTracker.mu.Unlock()

	session.dispatchEvent(SessionEvent{Type: AssistantTurnEnd})
	session.dispatchEvent(SessionEvent{Type: SessionIdle})
}

func TestOpenTelemetry_SpanReuseWithinTurn(t *testing.T) {
	ct, exporter, _ := setupTestTelemetry(t, &TelemetryConfig{})

	session := newTestSession(ct, false)

	// First send
	session.turnTracker.beginSend(context.Background(), "First question")

	// Second send within the same turn (no SessionIdle yet)
	session.turnTracker.beginSend(context.Background(), "Another question")

	// Complete
	session.dispatchEvent(SessionEvent{Type: SessionIdle})

	// Verify only 1 invoke_agent span was created (reused)
	spans := exporter.GetSpans()
	agentSpanCount := 0
	for _, s := range spans {
		if a := findSpanAttr(s, otelGenAIOperationName); a != nil && a.Value.AsString() == otelInvokeAgent {
			agentSpanCount++
		}
	}
	if agentSpanCount != 1 {
		t.Errorf("Expected 1 invoke_agent span (reused within turn), got %d", agentSpanCount)
	}
}

func TestOpenTelemetry_FinishReasons(t *testing.T) {
	ct, exporter, _ := setupTestTelemetry(t, &TelemetryConfig{})

	session := newTestSession(ct, false)
	session.turnTracker.beginSend(context.Background(), "Hello")
	session.dispatchEvent(SessionEvent{Type: AssistantTurnStart, Data: Data{}})
	session.dispatchEvent(SessionEvent{Type: AssistantTurnEnd})
	session.dispatchEvent(SessionEvent{Type: SessionIdle})

	spans := exporter.GetSpans()
	for _, s := range spans {
		a := findSpanAttr(s, otelGenAIResponseFinishReasons)
		if a == nil {
			t.Errorf("Span %q missing finish_reasons", s.Name)
			continue
		}
		reasons := a.Value.AsStringSlice()
		if len(reasons) != 1 || reasons[0] != "stop" {
			t.Errorf("Span %q finish_reasons = %v, want [\"stop\"]", s.Name, reasons)
		}
	}
}

func TestOpenTelemetry_MessageContentExcludedWhenSensitiveDisabled(t *testing.T) {
	ct, exporter, _ := setupTestTelemetry(t, &TelemetryConfig{EnableSensitiveData: Bool(false)})

	session := &Session{
		SessionID:        "session-no-sensitive",
		telemetry:        ct,
		toolHandlers:     make(map[string]ToolHandler),
		toolDescriptions: make(map[string]string),
	}
	sysMsg := &SystemMessageConfig{Content: "You are helpful."}
	tools := []Tool{{Name: "myTool", Description: "A tool", Parameters: map[string]any{"type": "object"}}}
	session.configureTelemetryContext("gpt-4", nil, sysMsg, tools, false, "", "")

	session.turnTracker.beginSend(context.Background(), "Hello")
	session.dispatchEvent(SessionEvent{Type: AssistantTurnStart, Data: Data{}})
	session.dispatchEvent(SessionEvent{
		Type: AssistantMessage,
		Data: Data{Content: ptrString("Hi there!")},
	})
	session.dispatchEvent(SessionEvent{Type: AssistantTurnEnd})
	session.dispatchEvent(SessionEvent{Type: SessionIdle})

	spans := exporter.GetSpans()
	for _, span := range spans {
		if findSpanAttr(span, otelGenAIInputMessages) != nil {
			t.Errorf("Span %q: gen_ai.input.messages should be excluded when sensitive disabled", span.Name)
		}
		if findSpanAttr(span, otelGenAIOutputMessages) != nil {
			t.Errorf("Span %q: gen_ai.output.messages should be excluded when sensitive disabled", span.Name)
		}
		if findSpanAttr(span, otelGenAISystemInstructions) != nil {
			t.Errorf("Span %q: gen_ai.system_instructions should be excluded when sensitive disabled", span.Name)
		}
	}

	// Tool definitions NOT sensitive-gated — should be on invoke_agent span
	agentSpan := spans[len(spans)-1]
	if findSpanAttr(agentSpan, otelGenAIToolDefinitions) == nil {
		t.Error("gen_ai.tool.definitions should always be included")
	}
}

func TestOpenTelemetry_CacheTokenAttributes(t *testing.T) {
	ct, exporter, _ := setupTestTelemetry(t, &TelemetryConfig{})

	session := newTestSession(ct, false)
	session.turnTracker.beginSend(context.Background(), "Hello")

	session.dispatchEvent(SessionEvent{Type: AssistantTurnStart, Data: Data{}})
	session.dispatchEvent(SessionEvent{
		Type: AssistantUsage,
		Data: Data{
			InputTokens:      ptrFloat(100),
			OutputTokens:     ptrFloat(50),
			CacheReadTokens:  ptrFloat(50),
			CacheWriteTokens: ptrFloat(25),
		},
	})
	session.dispatchEvent(SessionEvent{Type: AssistantTurnEnd})
	session.dispatchEvent(SessionEvent{Type: SessionIdle})

	spans := exporter.GetSpans()
	chatSpan := spans[0]

	if a := findSpanAttr(chatSpan, otelGenAIUsageCacheReadInputTokens); a == nil || a.Value.AsInt64() != 50 {
		t.Error("Expected gen_ai.usage.cache_read.input_tokens = 50")
	}
	if a := findSpanAttr(chatSpan, otelGenAIUsageCacheCreationInputTokens); a == nil || a.Value.AsInt64() != 25 {
		t.Error("Expected gen_ai.usage.cache_creation.input_tokens = 25")
	}
}

func TestOpenTelemetry_ChunkTimingUsesChat(t *testing.T) {
	ct, _, reader := setupTestTelemetry(t, &TelemetryConfig{})

	session := newTestSession(ct, true) // streaming=true
	session.turnTracker.beginSend(context.Background(), "")
	session.dispatchEvent(SessionEvent{Type: AssistantTurnStart, Data: Data{}})

	session.dispatchEvent(SessionEvent{
		Type: AssistantMessageDelta,
		Data: Data{DeltaContent: ptrString("Hello")},
	})

	session.dispatchEvent(SessionEvent{Type: AssistantTurnEnd})
	session.dispatchEvent(SessionEvent{Type: SessionIdle})

	rm := collectMetrics(t, reader)

	// Verify chunk timing metrics use "chat" operation name
	m := findMetricByName(rm, otelMetricTimeToFirstChunk)
	if m == nil {
		t.Fatal("Expected time_to_first_chunk metric")
	}
	histo := m.Data.(metricdata.Histogram[float64])
	for _, dp := range histo.DataPoints {
		for _, a := range dp.Attributes.ToSlice() {
			if string(a.Key) == otelGenAIOperationName {
				if a.Value.AsString() != otelChat {
					t.Errorf("chunk timing operation name = %q, want %q", a.Value.AsString(), otelChat)
				}
			}
		}
	}
}

func TestOpenTelemetry_TokenUsageMetricsUseChat(t *testing.T) {
	ct, _, reader := setupTestTelemetry(t, &TelemetryConfig{})

	session := newTestSession(ct, false)
	session.turnTracker.beginSend(context.Background(), "Hello")
	session.dispatchEvent(SessionEvent{Type: AssistantTurnStart, Data: Data{}})
	session.dispatchEvent(SessionEvent{
		Type: AssistantUsage,
		Data: Data{
			Model:        ptrString("gpt-4"),
			InputTokens:  ptrFloat(100),
			OutputTokens: ptrFloat(50),
		},
	})
	session.dispatchEvent(SessionEvent{Type: AssistantTurnEnd})
	session.dispatchEvent(SessionEvent{Type: SessionIdle})

	rm := collectMetrics(t, reader)

	m := findMetricByName(rm, otelMetricTokenUsage)
	if m == nil {
		t.Fatal("Expected token usage metric")
	}
	histo := m.Data.(metricdata.Histogram[int64])
	for _, dp := range histo.DataPoints {
		for _, a := range dp.Attributes.ToSlice() {
			if string(a.Key) == otelGenAIOperationName {
				if a.Value.AsString() != otelChat {
					t.Errorf("token usage operation name = %q, want %q", a.Value.AsString(), otelChat)
				}
			}
		}
	}
}
