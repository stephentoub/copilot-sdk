/* eslint-disable @typescript-eslint/no-explicit-any */
import { describe, expect, it, beforeEach, afterEach } from "vitest";
import { trace, metrics, SpanKind, SpanStatusCode } from "@opentelemetry/api";
import {
    InMemorySpanExporter,
    SimpleSpanProcessor,
    BasicTracerProvider,
} from "@opentelemetry/sdk-trace-base";
import {
    MeterProvider,
    InMemoryMetricExporter,
    PeriodicExportingMetricReader,
} from "@opentelemetry/sdk-metrics";
import { CopilotTelemetry } from "../src/copilot-telemetry.js";
import { CopilotSession } from "../src/session.js";
import type { SessionEvent } from "../src/types.js";

// ============================================================================
// Test infrastructure
// ============================================================================

let spanExporter: InMemorySpanExporter;
let tracerProvider: BasicTracerProvider;
let metricExporter: InMemoryMetricExporter;
let metricReader: PeriodicExportingMetricReader;
let meterProvider: MeterProvider;

function setupOtel(_sourceName = "github.copilot.sdk") {
    spanExporter = new InMemorySpanExporter();
    tracerProvider = new BasicTracerProvider({
        spanProcessors: [new SimpleSpanProcessor(spanExporter)],
    });
    trace.setGlobalTracerProvider(tracerProvider);

    metricExporter = new InMemoryMetricExporter();
    metricReader = new PeriodicExportingMetricReader({
        exporter: metricExporter,
        exportIntervalMillis: 60_000,
    });
    meterProvider = new MeterProvider({ readers: [metricReader] });
    metrics.setGlobalMeterProvider(meterProvider);
}

async function teardownOtel() {
    if (metricReader) await metricReader.shutdown();
    if (tracerProvider) await tracerProvider.shutdown();
    if (meterProvider) await meterProvider.shutdown();
    trace.disable();
    metrics.disable();
}

function getSpans() {
    return spanExporter.getFinishedSpans();
}

async function getMetrics() {
    await metricReader.forceFlush();
    const collected = metricExporter.getMetrics();
    if (collected.length === 0) return [];
    // Get the most recent collection
    const latest = collected[collected.length - 1];
    return latest.scopeMetrics;
}

function findMetric(scopeMetrics: any[], name: string) {
    for (const scope of scopeMetrics) {
        for (const metric of scope.metrics) {
            if (metric.descriptor.name === name) {
                return metric;
            }
        }
    }
    return undefined;
}

/** Create a mock MessageConnection for CopilotSession */
function createMockConnection(): any {
    return {
        sendRequest: async () => ({ messageId: "msg-1" }),
        onNotification: () => {},
        onRequest: () => {},
        listen: () => {},
        dispose: () => {},
    };
}

/** Create a session with telemetry for testing */
function createTestSession(
    telemetry: CopilotTelemetry,
    opts: {
        model?: string;
        providerType?: string;
        baseUrl?: string;
        streaming?: boolean;
        systemContent?: string;
        tools?: any[];
    } = {}
): CopilotSession {
    const conn = createMockConnection();
    const session = new CopilotSession(
        "test-session-id",
        conn,
        undefined,
        telemetry,
        opts.model ?? "gpt-4o",
        opts.providerType || opts.baseUrl
            ? {
                  type: opts.providerType as any,
                  baseUrl: opts.baseUrl ?? "https://api.openai.com/v1",
              }
            : undefined,
        opts.systemContent ? { mode: "replace", content: opts.systemContent } : undefined,
        opts.tools,
        opts.streaming ?? false
    );
    return session;
}

/** Dispatch an event to the session */
function dispatchEvent(session: CopilotSession, event: SessionEvent) {
    session._dispatchEvent(event);
}

/** Send a message and dispatch the matching user.message event (unit tests
 *  use mock connections that don't produce real events). */
async function sendWithEvent(
    session: CopilotSession,
    prompt: string,
    extra: Record<string, unknown> = {}
) {
    await session.send({ prompt, ...extra });
    dispatchEvent(session, makeEvent("user.message", { content: prompt }));
}

function makeEvent(type: string, data: Record<string, unknown> = {}): SessionEvent {
    return {
        id: "evt-1",
        timestamp: new Date().toISOString(),
        parentId: null,
        type,
        data,
    } as unknown as SessionEvent;
}

// ============================================================================
// Tests
// ============================================================================

describe("CopilotTelemetry", () => {
    beforeEach(() => {
        setupOtel();
    });

    afterEach(async () => {
        await teardownOtel();
    });

    // ========================================================================
    // Provider name normalization
    // ========================================================================

    describe("normalizeProviderName", () => {
        it("maps openai to openai", () => {
            expect(CopilotTelemetry.normalizeProviderName("openai")).toBe("openai");
        });

        it("maps azure to azure.ai.openai", () => {
            expect(CopilotTelemetry.normalizeProviderName("azure")).toBe("azure.ai.openai");
        });

        it("maps anthropic to anthropic", () => {
            expect(CopilotTelemetry.normalizeProviderName("anthropic")).toBe("anthropic");
        });

        it("maps unknown to github", () => {
            expect(CopilotTelemetry.normalizeProviderName("something")).toBe("github");
        });

        it("maps undefined to github", () => {
            expect(CopilotTelemetry.normalizeProviderName(undefined)).toBe("github");
        });

        it("maps empty string to github", () => {
            expect(CopilotTelemetry.normalizeProviderName("")).toBe("github");
        });

        it("handles case-insensitive matching", () => {
            expect(CopilotTelemetry.normalizeProviderName("OpenAI")).toBe("openai");
            expect(CopilotTelemetry.normalizeProviderName("AZURE")).toBe("azure.ai.openai");
        });
    });

    // ========================================================================
    // Server address parsing
    // ========================================================================

    describe("parseServerAddress", () => {
        it("parses standard http URL", () => {
            const result = CopilotTelemetry.parseServerAddress("http://api.example.com:8080/v1");
            expect(result.address).toBe("api.example.com");
            expect(result.port).toBe(8080);
        });

        it("parses https URL without explicit port", () => {
            const result = CopilotTelemetry.parseServerAddress("https://api.openai.com/v1");
            expect(result.address).toBe("api.openai.com");
            expect(result.port).toBe(443);
        });

        it("parses http URL without explicit port", () => {
            const result = CopilotTelemetry.parseServerAddress("http://example.com/path");
            expect(result.address).toBe("example.com");
            expect(result.port).toBe(80);
        });

        it("returns undefined for empty/undefined input", () => {
            expect(CopilotTelemetry.parseServerAddress(undefined)).toEqual({
                address: undefined,
                port: undefined,
            });
            expect(CopilotTelemetry.parseServerAddress("")).toEqual({
                address: undefined,
                port: undefined,
            });
        });

        it("returns undefined for invalid URL", () => {
            expect(CopilotTelemetry.parseServerAddress("not-a-url")).toEqual({
                address: undefined,
                port: undefined,
            });
        });
    });

    // ========================================================================
    // invoke_agent span creation
    // ========================================================================

    describe("invoke_agent span", () => {
        it("creates span with correct attributes on send", async () => {
            const telemetry = new CopilotTelemetry({});
            const session = createTestSession(telemetry, {
                model: "gpt-4o",
                baseUrl: "https://api.openai.com:8080/v1",
            });

            await sendWithEvent(session, "Hello");
            dispatchEvent(session, makeEvent("session.idle"));

            const spans = getSpans();
            expect(spans.length).toBe(1);

            const span = spans[0];
            expect(span.kind).toBe(SpanKind.CLIENT);
            expect(span.attributes["gen_ai.operation.name"]).toBe("invoke_agent");
            expect(span.attributes["gen_ai.provider.name"]).toBe("github");
            expect(span.attributes["gen_ai.agent.id"]).toBe("test-session-id");
            expect(span.attributes["gen_ai.conversation.id"]).toBe("test-session-id");
            expect(span.attributes["gen_ai.request.model"]).toBe("gpt-4o");
            expect(span.attributes["server.address"]).toBe("api.openai.com");
            expect(span.attributes["server.port"]).toBe(8080);
        });

        it("subsequent sends within same turn do not create new span", async () => {
            const telemetry = new CopilotTelemetry({});
            const session = createTestSession(telemetry);

            await sendWithEvent(session, "First");
            await sendWithEvent(session, "Second");
            dispatchEvent(session, makeEvent("session.idle"));

            const spans = getSpans();
            expect(spans.length).toBe(1);
        });

        it("new span after idle", async () => {
            const telemetry = new CopilotTelemetry({});
            const session = createTestSession(telemetry);

            // First turn
            await sendWithEvent(session, "Turn 1");
            dispatchEvent(session, makeEvent("session.idle"));

            // Second turn
            await sendWithEvent(session, "Turn 2");
            dispatchEvent(session, makeEvent("session.idle"));

            const spans = getSpans();
            expect(spans.length).toBe(2);
        });
    });

    // ========================================================================
    // execute_tool span creation
    // ========================================================================

    describe("execute_tool span", () => {
        it("creates span with correct attributes", () => {
            const telemetry = new CopilotTelemetry({});
            const span = telemetry.startExecuteToolSpan(
                "get_weather",
                "call-123",
                "Gets weather info",
                { location: "Seattle" }
            );

            expect(span).toBeDefined();

            span.end();

            const spans = getSpans();
            expect(spans.length).toBe(1);
            expect(spans[0].kind).toBe(SpanKind.INTERNAL);
            expect(spans[0].attributes["gen_ai.operation.name"]).toBe("execute_tool");
            expect(spans[0].attributes["gen_ai.tool.name"]).toBe("get_weather");
            expect(spans[0].attributes["gen_ai.tool.call.id"]).toBe("call-123");
            expect(spans[0].attributes["gen_ai.tool.type"]).toBe("function");
            expect(spans[0].attributes["gen_ai.tool.description"]).toBe("Gets weather info");
        });

        it("does not include arguments when sensitive data is disabled", () => {
            const telemetry = new CopilotTelemetry({ enableSensitiveData: false });
            const span = telemetry.startExecuteToolSpan("get_weather", "call-123", undefined, {
                location: "Seattle",
            });
            span.end();

            const spans = getSpans();
            expect(spans[0].attributes["gen_ai.tool.call.arguments"]).toBeUndefined();
        });

        it("includes arguments when sensitive data is enabled", () => {
            const telemetry = new CopilotTelemetry({ enableSensitiveData: true });
            const span = telemetry.startExecuteToolSpan("get_weather", "call-123", undefined, {
                location: "Seattle",
            });
            span.end();

            const spans = getSpans();
            expect(spans[0].attributes["gen_ai.tool.call.arguments"]).toBe(
                JSON.stringify({ location: "Seattle" })
            );
        });
    });

    // ========================================================================
    // Metric recording
    // ========================================================================

    describe("metric recording", () => {
        it("records operation duration metric on session.idle", async () => {
            const telemetry = new CopilotTelemetry({});
            const session = createTestSession(telemetry, { model: "gpt-4o" });

            await sendWithEvent(session, "Hello");
            dispatchEvent(session, makeEvent("session.idle"));

            const scopeMetrics = await getMetrics();
            const durationMetric = findMetric(scopeMetrics, "gen_ai.client.operation.duration");
            expect(durationMetric).toBeDefined();
            expect(durationMetric.dataPoints.length).toBeGreaterThan(0);
        });

        it("records token usage metrics from assistant.usage event", async () => {
            const telemetry = new CopilotTelemetry({});
            const session = createTestSession(telemetry, { model: "gpt-4o" });

            await sendWithEvent(session, "Hello");
            dispatchEvent(session, makeEvent("assistant.turn_start", { turnId: "turn-1" }));
            dispatchEvent(
                session,
                makeEvent("assistant.usage", {
                    model: "gpt-4o-2024-05-13",
                    inputTokens: 100,
                    outputTokens: 50,
                })
            );
            dispatchEvent(session, makeEvent("session.idle"));

            const spans = getSpans();
            // Chat span has per-turn usage
            const chatSpan = spans.find((s) => s.name.startsWith("chat"));
            expect(chatSpan).toBeDefined();
            expect(chatSpan!.attributes["gen_ai.usage.input_tokens"]).toBe(100);
            expect(chatSpan!.attributes["gen_ai.usage.output_tokens"]).toBe(50);
            expect(chatSpan!.attributes["gen_ai.response.model"]).toBe("gpt-4o-2024-05-13");
            // invoke_agent span has accumulated totals
            const agentSpan = spans.find((s) => s.name.startsWith("invoke_agent"));
            expect(agentSpan).toBeDefined();
            expect(agentSpan!.attributes["gen_ai.usage.input_tokens"]).toBe(100);
            expect(agentSpan!.attributes["gen_ai.usage.output_tokens"]).toBe(50);

            const scopeMetrics = await getMetrics();
            const tokenMetric = findMetric(scopeMetrics, "gen_ai.client.token.usage");
            expect(tokenMetric).toBeDefined();
            expect(tokenMetric.dataPoints.length).toBe(2); // input + output
        });

        it("records streaming chunk metrics", async () => {
            const telemetry = new CopilotTelemetry({});
            const session = createTestSession(telemetry, {
                model: "gpt-4o",
                streaming: true,
            });

            await sendWithEvent(session, "Hello");
            dispatchEvent(session, makeEvent("assistant.turn_start", { turnId: "turn-1" }));

            // Simulate streaming chunks
            dispatchEvent(
                session,
                makeEvent("assistant.message_delta", {
                    messageId: "msg-1",
                    deltaContent: "First ",
                })
            );
            dispatchEvent(
                session,
                makeEvent("assistant.message_delta", {
                    messageId: "msg-1",
                    deltaContent: "second ",
                })
            );
            dispatchEvent(
                session,
                makeEvent("assistant.message", {
                    messageId: "msg-1",
                    content: "First second chunk",
                })
            );
            dispatchEvent(session, makeEvent("session.idle"));

            const scopeMetrics = await getMetrics();
            const ttfcMetric = findMetric(
                scopeMetrics,
                "gen_ai.client.operation.time_to_first_chunk"
            );
            const tpocMetric = findMetric(
                scopeMetrics,
                "gen_ai.client.operation.time_per_output_chunk"
            );
            expect(ttfcMetric).toBeDefined();
            expect(ttfcMetric.dataPoints.length).toBe(1); // one time-to-first-chunk
            // Subsequent chunks record time-per-output-chunk
            expect(tpocMetric).toBeDefined();
            expect(tpocMetric.dataPoints.length).toBeGreaterThanOrEqual(1);
        });
    });

    // ========================================================================
    // Sensitive data gating
    // ========================================================================

    describe("sensitive data gating", () => {
        it("includes tool call result when sensitive enabled", () => {
            const telemetry = new CopilotTelemetry({ enableSensitiveData: true });
            const span = telemetry.startExecuteToolSpan("tool", "call-1", undefined, {});
            telemetry.setExecuteToolResult(span, { result: "data" });
            span.end();

            const spans = getSpans();
            expect(spans[0].attributes["gen_ai.tool.call.result"]).toBe(
                JSON.stringify({ result: "data" })
            );
        });

        it("excludes tool call result when sensitive disabled", () => {
            const telemetry = new CopilotTelemetry({ enableSensitiveData: false });
            const span = telemetry.startExecuteToolSpan("tool", "call-1", undefined, {});
            telemetry.setExecuteToolResult(span, { result: "data" });
            span.end();

            const spans = getSpans();
            expect(spans[0].attributes["gen_ai.tool.call.result"]).toBeUndefined();
        });

        it("includes input/output messages when sensitive enabled", async () => {
            const telemetry = new CopilotTelemetry({ enableSensitiveData: true });
            const session = createTestSession(telemetry, {
                systemContent: "You are a helpful assistant.",
                tools: [
                    {
                        name: "myTool",
                        description: "A tool",
                        parameters: { type: "object" },
                        handler: () => {},
                    },
                ],
            });

            await sendWithEvent(session, "Hello");
            dispatchEvent(session, makeEvent("assistant.turn_start", { turnId: "turn-1" }));
            dispatchEvent(session, makeEvent("assistant.message", { content: "Hi there!" }));
            dispatchEvent(session, makeEvent("session.idle"));

            const spans = getSpans();
            const invokeSpan = spans.find((s) => s.name.startsWith("invoke_agent"));
            expect(invokeSpan).toBeDefined();

            // Input messages on invoke_agent span (agent-level = user prompts)
            const agentInputMessages = invokeSpan!.attributes["gen_ai.input.messages"] as string;
            expect(agentInputMessages).toBeDefined();
            const parsedAgentInput = JSON.parse(agentInputMessages);
            expect(parsedAgentInput[0].role).toBe("user");
            expect(parsedAgentInput[0].parts[0].content).toBe("Hello");

            // Chat span has per-turn input/output messages
            const chatSpan = spans.find((s) => s.name.startsWith("chat"));
            expect(chatSpan).toBeDefined();

            const chatInputMessages = chatSpan!.attributes["gen_ai.input.messages"] as string;
            expect(chatInputMessages).toBeDefined();
            const parsedInput = JSON.parse(chatInputMessages);
            expect(parsedInput[0].role).toBe("user");
            expect(parsedInput[0].parts[0].content).toBe("Hello");

            // Output messages on chat span
            const outputMessages = chatSpan!.attributes["gen_ai.output.messages"] as string;
            expect(outputMessages).toBeDefined();
            const parsedOutput = JSON.parse(outputMessages);
            expect(parsedOutput[0].role).toBe("assistant");
            expect(parsedOutput[0].parts[0].content).toBe("Hi there!");
            expect(parsedOutput[0].finish_reason).toBe("stop");

            // System instructions on invoke_agent span
            const sysInstructions = invokeSpan!.attributes["gen_ai.system_instructions"] as string;
            expect(sysInstructions).toBeDefined();
            const parsedSys = JSON.parse(sysInstructions);
            expect(parsedSys[0].content).toBe("You are a helpful assistant.");

            // Tool definitions (always included, not sensitive-gated)
            const toolDefs = invokeSpan!.attributes["gen_ai.tool.definitions"] as string;
            expect(toolDefs).toBeDefined();
            const parsedTools = JSON.parse(toolDefs);
            expect(parsedTools[0].name).toBe("myTool");
        });

        it("excludes input/output messages when sensitive disabled", async () => {
            const telemetry = new CopilotTelemetry({ enableSensitiveData: false });
            const session = createTestSession(telemetry, {
                systemContent: "You are a helpful assistant.",
                tools: [
                    {
                        name: "myTool",
                        description: "A tool",
                        parameters: { type: "object" },
                        handler: () => {},
                    },
                ],
            });

            await sendWithEvent(session, "Hello");
            dispatchEvent(session, makeEvent("assistant.turn_start", { turnId: "turn-1" }));
            dispatchEvent(session, makeEvent("assistant.message", { content: "Hi there!" }));
            dispatchEvent(session, makeEvent("session.idle"));

            const spans = getSpans();
            const invokeSpan = spans.find((s) => s.name.startsWith("invoke_agent"));
            expect(invokeSpan).toBeDefined();

            // Sensitive attributes should be excluded
            expect(invokeSpan!.attributes["gen_ai.input.messages"]).toBeUndefined();
            expect(invokeSpan!.attributes["gen_ai.output.messages"]).toBeUndefined();
            expect(invokeSpan!.attributes["gen_ai.system_instructions"]).toBeUndefined();

            // Tool definitions are NOT sensitive-gated, should still be present
            const toolDefs = invokeSpan!.attributes["gen_ai.tool.definitions"] as string;
            expect(toolDefs).toBeDefined();
            const parsedTools = JSON.parse(toolDefs);
            expect(parsedTools[0].name).toBe("myTool");
        });
    });

    // ========================================================================
    // Opt-in behavior
    // ========================================================================

    describe("opt-in behavior", () => {
        it("no telemetry when config not provided", async () => {
            // Create session without telemetry
            const conn = createMockConnection();
            const session = new CopilotSession("test-id", conn);

            await sendWithEvent(session, "Hello");
            dispatchEvent(session, makeEvent("session.idle"));

            const spans = getSpans();
            expect(spans.length).toBe(0);
        });
    });

    // ========================================================================
    // Custom source name
    // ========================================================================

    describe("custom source name", () => {
        it("uses custom source name for tracer and meter", async () => {
            const telemetry = new CopilotTelemetry({
                sourceName: "my.custom.source",
            });
            const session = createTestSession(telemetry);

            await sendWithEvent(session, "Hello");
            dispatchEvent(session, makeEvent("session.idle"));

            const spans = getSpans();
            expect(spans.length).toBe(1);
            expect((spans[0] as any).instrumentationScope.name).toBe("my.custom.source");

            const scopeMetrics = await getMetrics();
            // Check that metrics were recorded under custom source name
            const customScope = scopeMetrics.find((sm) => sm.scope.name === "my.custom.source");
            expect(customScope).toBeDefined();
        });
    });

    // ========================================================================
    // Error handling
    // ========================================================================

    describe("error handling", () => {
        it("records error on session.error event", async () => {
            const telemetry = new CopilotTelemetry({});
            const session = createTestSession(telemetry);

            await sendWithEvent(session, "Hello");
            dispatchEvent(
                session,
                makeEvent("session.error", {
                    errorType: "internal",
                    message: "Something went wrong",
                })
            );

            const spans = getSpans();
            expect(spans.length).toBe(1);
            expect(spans[0].status.code).toBe(SpanStatusCode.ERROR);
            expect(spans[0].attributes["error.type"]).toBe("SessionTurnError");
        });

        it("sets response.id from apiCallId", async () => {
            const telemetry = new CopilotTelemetry({});
            const session = createTestSession(telemetry);

            await sendWithEvent(session, "Hello");
            dispatchEvent(session, makeEvent("assistant.turn_start", { turnId: "turn-1" }));
            dispatchEvent(
                session,
                makeEvent("assistant.usage", {
                    model: "gpt-4o",
                    inputTokens: 10,
                    outputTokens: 5,
                    apiCallId: "api-call-123",
                })
            );
            dispatchEvent(session, makeEvent("session.idle"));

            const spans = getSpans();
            const chatSpan = spans.find((s) => s.name.startsWith("chat"));
            expect(chatSpan).toBeDefined();
            expect(chatSpan!.attributes["gen_ai.response.id"]).toBe("api-call-123");
        });

        it("sets response.id from providerCallId when apiCallId is missing", async () => {
            const telemetry = new CopilotTelemetry({});
            const session = createTestSession(telemetry);

            await sendWithEvent(session, "Hello");
            dispatchEvent(session, makeEvent("assistant.turn_start", { turnId: "turn-1" }));
            dispatchEvent(
                session,
                makeEvent("assistant.usage", {
                    model: "gpt-4o",
                    inputTokens: 10,
                    outputTokens: 5,
                    providerCallId: "provider-call-456",
                })
            );
            dispatchEvent(session, makeEvent("session.idle"));

            const spans = getSpans();
            const chatSpan = spans.find((s) => s.name.startsWith("chat"));
            expect(chatSpan).toBeDefined();
            expect(chatSpan!.attributes["gen_ai.response.id"]).toBe("provider-call-456");
        });
    });

    // ========================================================================
    // Model change event
    // ========================================================================

    describe("session.model_change event", () => {
        it("updates response model from model_change event", async () => {
            const telemetry = new CopilotTelemetry({});
            const session = createTestSession(telemetry, { model: "gpt-4o" });

            await sendWithEvent(session, "Hello");
            dispatchEvent(session, makeEvent("assistant.turn_start", { turnId: "turn-1" }));
            dispatchEvent(
                session,
                makeEvent("session.model_change", {
                    previousModel: "gpt-4o",
                    newModel: "gpt-4o-mini",
                })
            );
            dispatchEvent(
                session,
                makeEvent("assistant.usage", {
                    model: "gpt-4o-mini",
                    inputTokens: 10,
                    outputTokens: 5,
                })
            );
            dispatchEvent(session, makeEvent("session.idle"));

            const spans = getSpans();
            const chatSpan = spans.find((s) => s.name.startsWith("chat"));
            expect(chatSpan).toBeDefined();
            expect(chatSpan!.attributes["gen_ai.response.model"]).toBe("gpt-4o-mini");
        });
    });

    // ========================================================================
    // Cache token attributes
    // ========================================================================

    describe("cache token attributes", () => {
        it("records cache read and write tokens", async () => {
            const telemetry = new CopilotTelemetry({});
            const session = createTestSession(telemetry);

            await sendWithEvent(session, "Hello");
            dispatchEvent(session, makeEvent("assistant.turn_start", { turnId: "turn-1" }));
            dispatchEvent(
                session,
                makeEvent("assistant.usage", {
                    model: "gpt-4o",
                    inputTokens: 100,
                    outputTokens: 50,
                    cacheReadTokens: 80,
                    cacheWriteTokens: 20,
                })
            );
            dispatchEvent(session, makeEvent("session.idle"));

            const spans = getSpans();
            const chatSpan = spans.find((s) => s.name.startsWith("chat"));
            expect(chatSpan).toBeDefined();
            expect(chatSpan!.attributes["gen_ai.usage.cache_read.input_tokens"]).toBe(80);
            expect(chatSpan!.attributes["gen_ai.usage.cache_creation.input_tokens"]).toBe(20);
        });
    });

    // ========================================================================
    // Static recordError
    // ========================================================================

    describe("static recordError", () => {
        it("sets error attributes and status on span", () => {
            const telemetry = new CopilotTelemetry({});
            const span = telemetry.startInvokeAgentSpan(
                "sid",
                "model",
                "github",
                undefined,
                undefined
            );
            CopilotTelemetry.recordError(span, new TypeError("test error"));
            span.end();

            const spans = getSpans();
            expect(spans[0].attributes["error.type"]).toBe("TypeError");
            expect(spans[0].status.code).toBe(SpanStatusCode.ERROR);
            expect(spans[0].status.message).toBe("test error");
        });
    });

    // ========================================================================
    // Finish reasons
    // ========================================================================

    describe("finish reasons", () => {
        it("sets gen_ai.response.finish_reasons to ['stop'] on successful turn completion", async () => {
            const telemetry = new CopilotTelemetry({});
            const session = createTestSession(telemetry);

            await sendWithEvent(session, "Hello");
            dispatchEvent(session, makeEvent("session.idle"));

            const spans = getSpans();
            expect(spans.length).toBe(1);
            expect(spans[0].attributes["gen_ai.response.finish_reasons"]).toEqual(["stop"]);
        });
    });

    // ========================================================================
    // Agent metadata
    // ========================================================================

    describe("agent metadata", () => {
        it("includes gen_ai.agent.name and gen_ai.agent.description on invoke_agent span", async () => {
            const telemetry = new CopilotTelemetry({});
            const conn = createMockConnection();
            const session = new CopilotSession(
                "test-agent-session",
                conn,
                undefined, // workspacePath
                telemetry, // telemetry
                "gpt-4o", // model
                undefined, // provider
                undefined, // systemMessage
                undefined, // tools
                false, // streaming
                "my-agent", // agentName
                "A helpful agent" // agentDescription
            );

            await sendWithEvent(session, "Hello");
            dispatchEvent(session, makeEvent("session.idle"));

            const spans = getSpans();
            expect(spans.length).toBe(1);
            expect(spans[0].attributes["gen_ai.agent.name"]).toBe("my-agent");
            expect(spans[0].attributes["gen_ai.agent.description"]).toBe("A helpful agent");
        });
    });

    // ========================================================================
    // Environment variable override
    // ========================================================================

    describe("env var override", () => {
        it("enables sensitive data when OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true", () => {
            const originalEnv = process.env["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"];
            try {
                process.env["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = "true";
                const telemetry = new CopilotTelemetry({});
                expect(telemetry.enableSensitiveData).toBe(true);
            } finally {
                if (originalEnv !== undefined) {
                    process.env["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = originalEnv;
                } else {
                    delete process.env["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"];
                }
            }
        });
    });

    // ========================================================================
    // Two-level span hierarchy (invoke_agent → chat)
    // ========================================================================

    describe("two-level span hierarchy", () => {
        it("assistant.turn_start creates a chat child span under invoke_agent", async () => {
            const telemetry = new CopilotTelemetry({});
            const session = createTestSession(telemetry);

            await sendWithEvent(session, "Hello");
            dispatchEvent(session, makeEvent("assistant.turn_start", { turnId: "turn-1" }));
            dispatchEvent(session, makeEvent("session.idle"));

            const spans = getSpans();
            const invokeSpan = spans.find((s) => s.name.startsWith("invoke_agent"));
            const chatSpan = spans.find((s) => s.name.startsWith("chat"));
            expect(invokeSpan).toBeDefined();
            expect(chatSpan).toBeDefined();
            expect(chatSpan!.attributes["gen_ai.operation.name"]).toBe("chat");
            expect(chatSpan!.attributes["gen_ai.conversation.id"]).toBe("test-session-id");
            // Chat span is in the same trace as invoke_agent
            expect(chatSpan!.spanContext().traceId).toBe(invokeSpan!.spanContext().traceId);
        });

        it("multi-turn: two turn_start/turn_end cycles produce 1 invoke_agent + 2 chat spans", async () => {
            const telemetry = new CopilotTelemetry({});
            const session = createTestSession(telemetry);

            await sendWithEvent(session, "Hello");

            // Turn 1
            dispatchEvent(session, makeEvent("assistant.turn_start", { turnId: "turn-1" }));
            dispatchEvent(
                session,
                makeEvent("assistant.usage", {
                    model: "gpt-4o",
                    inputTokens: 50,
                    outputTokens: 20,
                })
            );
            dispatchEvent(session, makeEvent("assistant.turn_end", { turnId: "turn-1" }));

            // Turn 2
            dispatchEvent(session, makeEvent("assistant.turn_start", { turnId: "turn-2" }));
            dispatchEvent(
                session,
                makeEvent("assistant.usage", {
                    model: "gpt-4o",
                    inputTokens: 60,
                    outputTokens: 30,
                })
            );
            dispatchEvent(session, makeEvent("session.idle"));

            const spans = getSpans();
            const invokeSpans = spans.filter((s) => s.name.startsWith("invoke_agent"));
            const chatSpans = spans.filter((s) => s.name.startsWith("chat"));
            expect(invokeSpans.length).toBe(1);
            expect(chatSpans.length).toBe(2);
        });

        it("per-turn usage goes on chat span, accumulated usage goes on invoke_agent", async () => {
            const telemetry = new CopilotTelemetry({});
            const session = createTestSession(telemetry);

            await sendWithEvent(session, "Hello");

            // Turn 1
            dispatchEvent(session, makeEvent("assistant.turn_start", { turnId: "turn-1" }));
            dispatchEvent(
                session,
                makeEvent("assistant.usage", {
                    model: "gpt-4o",
                    inputTokens: 50,
                    outputTokens: 20,
                })
            );
            dispatchEvent(session, makeEvent("assistant.turn_end", { turnId: "turn-1" }));

            // Turn 2
            dispatchEvent(session, makeEvent("assistant.turn_start", { turnId: "turn-2" }));
            dispatchEvent(
                session,
                makeEvent("assistant.usage", {
                    model: "gpt-4o",
                    inputTokens: 60,
                    outputTokens: 30,
                })
            );
            dispatchEvent(session, makeEvent("session.idle"));

            const spans = getSpans();
            const chatSpans = spans
                .filter((s) => s.name.startsWith("chat"))
                .sort((a, b) => a.startTime[1] - b.startTime[1]);
            const invokeSpan = spans.find((s) => s.name.startsWith("invoke_agent"))!;

            // Per-turn usage on chat spans
            expect(chatSpans[0].attributes["gen_ai.usage.input_tokens"]).toBe(50);
            expect(chatSpans[0].attributes["gen_ai.usage.output_tokens"]).toBe(20);
            expect(chatSpans[1].attributes["gen_ai.usage.input_tokens"]).toBe(60);
            expect(chatSpans[1].attributes["gen_ai.usage.output_tokens"]).toBe(30);

            // Accumulated totals on invoke_agent
            expect(invokeSpan.attributes["gen_ai.usage.input_tokens"]).toBe(110);
            expect(invokeSpan.attributes["gen_ai.usage.output_tokens"]).toBe(50);
        });
    });

    // ========================================================================
    // Rich message content (parts-based)
    // ========================================================================

    describe("rich message content", () => {
        it("output messages include reasoning parts when reasoningText is present", async () => {
            const telemetry = new CopilotTelemetry({ enableSensitiveData: true });
            const session = createTestSession(telemetry);

            await sendWithEvent(session, "Think step by step");
            dispatchEvent(session, makeEvent("assistant.turn_start", { turnId: "turn-1" }));
            dispatchEvent(
                session,
                makeEvent("assistant.message", {
                    messageId: "msg-1",
                    content: "The answer is 42.",
                    reasoningText: "Let me think about this...",
                })
            );
            dispatchEvent(session, makeEvent("session.idle"));

            const spans = getSpans();
            const chatSpan = spans.find((s) => s.name.startsWith("chat"))!;
            const outputMessages = JSON.parse(
                chatSpan.attributes["gen_ai.output.messages"] as string
            );
            expect(outputMessages[0].role).toBe("assistant");
            // Should have reasoning part first, then text part
            const parts = outputMessages[0].parts;
            expect(parts[0].type).toBe("reasoning");
            expect(parts[0].content).toBe("Let me think about this...");
            expect(parts[1].type).toBe("text");
            expect(parts[1].content).toBe("The answer is 42.");
        });

        it("tool_call and tool_call_response parts appear for tool execution events", async () => {
            const telemetry = new CopilotTelemetry({ enableSensitiveData: true });
            const session = createTestSession(telemetry);

            await sendWithEvent(session, "Get weather");
            dispatchEvent(session, makeEvent("assistant.turn_start", { turnId: "turn-1" }));

            // Tool execution start -> output message with tool_call part
            dispatchEvent(
                session,
                makeEvent("tool.execution_start", {
                    toolCallId: "call-1",
                    toolName: "get_weather",
                    arguments: { city: "Seattle" },
                })
            );

            // Tool execution complete -> input message with tool_call_response part
            dispatchEvent(
                session,
                makeEvent("tool.execution_complete", {
                    toolCallId: "call-1",
                    success: true,
                    result: { content: "Sunny, 72°F" },
                })
            );

            dispatchEvent(session, makeEvent("session.idle"));

            const spans = getSpans();
            const chatSpan = spans.find((s) => s.name.startsWith("chat"))!;

            // Output messages should include tool_call
            const outputMessages = JSON.parse(
                chatSpan.attributes["gen_ai.output.messages"] as string
            );
            const toolCallMsg = outputMessages.find((m: any) => m.parts[0]?.type === "tool_call");
            expect(toolCallMsg).toBeDefined();
            expect(toolCallMsg.parts[0].name).toBe("get_weather");
            expect(toolCallMsg.parts[0].id).toBe("call-1");

            // Input messages should include tool_call_response
            const inputMessages = JSON.parse(
                chatSpan.attributes["gen_ai.input.messages"] as string
            );
            const toolResponseMsg = inputMessages.find((m: any) => m.role === "tool");
            expect(toolResponseMsg).toBeDefined();
            expect(toolResponseMsg.parts[0].type).toBe("tool_call_response");
            expect(toolResponseMsg.parts[0].response).toBe("Sunny, 72°F");
        });

        it("user prompt appears as input message with text type", async () => {
            const telemetry = new CopilotTelemetry({ enableSensitiveData: true });
            const session = createTestSession(telemetry);

            await sendWithEvent(session, "Hello world");
            dispatchEvent(session, makeEvent("assistant.turn_start", { turnId: "turn-1" }));
            dispatchEvent(session, makeEvent("session.idle"));

            const spans = getSpans();
            const chatSpan = spans.find((s) => s.name.startsWith("chat"))!;
            const inputMessages = JSON.parse(
                chatSpan.attributes["gen_ai.input.messages"] as string
            );
            expect(inputMessages[0].role).toBe("user");
            expect(inputMessages[0].parts[0].type).toBe("text");
            expect(inputMessages[0].parts[0].content).toBe("Hello world");
        });
    });

    // ========================================================================
    // Subagent spans
    // ========================================================================

    describe("subagent spans", () => {
        it("subagent.started creates a nested invoke_agent → chat hierarchy", async () => {
            const telemetry = new CopilotTelemetry({});
            const session = createTestSession(telemetry);

            await sendWithEvent(session, "Hello");
            dispatchEvent(session, makeEvent("assistant.turn_start", { turnId: "turn-1" }));
            dispatchEvent(
                session,
                makeEvent("subagent.started", {
                    toolCallId: "sub-call-1",
                    agentName: "code-reviewer",
                    agentDisplayName: "Code Reviewer",
                    agentDescription: "Reviews code changes",
                })
            );
            dispatchEvent(
                session,
                makeEvent("subagent.completed", {
                    toolCallId: "sub-call-1",
                    agentName: "code-reviewer",
                    agentDisplayName: "Code Reviewer",
                })
            );
            dispatchEvent(session, makeEvent("session.idle"));

            const spans = getSpans();
            // Main invoke_agent + main chat + subagent invoke_agent + subagent chat
            const invokeSpans = spans.filter(
                (s) => s.attributes["gen_ai.operation.name"] === "invoke_agent"
            );
            const chatSpans = spans.filter((s) => s.attributes["gen_ai.operation.name"] === "chat");
            expect(invokeSpans.length).toBe(2); // main + subagent
            expect(chatSpans.length).toBe(2); // main + subagent
        });

        it("subagent spans are parented under the main invoke_agent span", async () => {
            const telemetry = new CopilotTelemetry({});
            const session = createTestSession(telemetry);

            await sendWithEvent(session, "Hello");
            dispatchEvent(session, makeEvent("assistant.turn_start", { turnId: "turn-1" }));
            dispatchEvent(
                session,
                makeEvent("subagent.started", {
                    toolCallId: "sub-call-1",
                    agentName: "helper",
                    agentDisplayName: "Helper",
                    agentDescription: "Helps with tasks",
                })
            );
            dispatchEvent(
                session,
                makeEvent("subagent.completed", {
                    toolCallId: "sub-call-1",
                    agentName: "helper",
                    agentDisplayName: "Helper",
                })
            );
            dispatchEvent(session, makeEvent("session.idle"));

            const spans = getSpans();
            const mainInvokeSpan = spans.find(
                (s) =>
                    s.attributes["gen_ai.operation.name"] === "invoke_agent" &&
                    !s.attributes["gen_ai.agent.name"]
            )!;
            const subInvokeSpan = spans.find(
                (s) =>
                    s.attributes["gen_ai.operation.name"] === "invoke_agent" &&
                    s.attributes["gen_ai.agent.name"] === "helper"
            )!;

            // Subagent invoke_agent is in the same trace as main invoke_agent
            expect(subInvokeSpan.spanContext().traceId).toBe(mainInvokeSpan.spanContext().traceId);

            // There should be 2 chat spans (main + subagent) in the same trace
            const chatSpans = spans.filter((s) => s.attributes["gen_ai.operation.name"] === "chat");
            expect(chatSpans.length).toBe(2);
        });

        it("subagent attributes include agent name and description", async () => {
            const telemetry = new CopilotTelemetry({});
            const session = createTestSession(telemetry);

            await sendWithEvent(session, "Hello");
            dispatchEvent(session, makeEvent("assistant.turn_start", { turnId: "turn-1" }));
            dispatchEvent(
                session,
                makeEvent("subagent.started", {
                    toolCallId: "sub-call-1",
                    agentName: "code-reviewer",
                    agentDisplayName: "Code Reviewer",
                    agentDescription: "Reviews code changes",
                })
            );
            dispatchEvent(
                session,
                makeEvent("subagent.completed", {
                    toolCallId: "sub-call-1",
                    agentName: "code-reviewer",
                    agentDisplayName: "Code Reviewer",
                })
            );
            dispatchEvent(session, makeEvent("session.idle"));

            const spans = getSpans();
            const subInvokeSpan = spans.find(
                (s) =>
                    s.attributes["gen_ai.operation.name"] === "invoke_agent" &&
                    s.attributes["gen_ai.agent.name"] === "code-reviewer"
            )!;
            expect(subInvokeSpan).toBeDefined();
            expect(subInvokeSpan.attributes["gen_ai.agent.name"]).toBe("code-reviewer");
            expect(subInvokeSpan.attributes["gen_ai.agent.description"]).toBe(
                "Reviews code changes"
            );
        });

        it("subagent.completed completes the subagent spans", async () => {
            const telemetry = new CopilotTelemetry({});
            const session = createTestSession(telemetry);

            await sendWithEvent(session, "Hello");
            dispatchEvent(session, makeEvent("assistant.turn_start", { turnId: "turn-1" }));
            dispatchEvent(
                session,
                makeEvent("subagent.started", {
                    toolCallId: "sub-call-1",
                    agentName: "helper",
                    agentDisplayName: "Helper",
                    agentDescription: "Helps",
                })
            );

            // Before completion, subagent spans should NOT be finished yet
            const spansBefore = getSpans();
            const subSpansBefore = spansBefore.filter(
                (s) => s.attributes["gen_ai.agent.name"] === "helper"
            );
            expect(subSpansBefore.length).toBe(0); // not yet finished

            dispatchEvent(
                session,
                makeEvent("subagent.completed", {
                    toolCallId: "sub-call-1",
                    agentName: "helper",
                    agentDisplayName: "Helper",
                })
            );

            // After completion, subagent spans should be finished
            const spansAfter = getSpans();
            const subSpansAfter = spansAfter.filter(
                (s) => s.attributes["gen_ai.agent.name"] === "helper"
            );
            expect(subSpansAfter.length).toBe(1); // invoke_agent span for subagent

            dispatchEvent(session, makeEvent("session.idle"));
        });
    });

    // ========================================================================
    // MCP server tool tracking
    // ========================================================================

    describe("MCP server tool tracking", () => {
        it("tool execution with mcpServerName produces server_tool_call / server_tool_call_response parts", async () => {
            const telemetry = new CopilotTelemetry({ enableSensitiveData: true });
            const session = createTestSession(telemetry);

            await sendWithEvent(session, "Use MCP tool");
            dispatchEvent(session, makeEvent("assistant.turn_start", { turnId: "turn-1" }));

            dispatchEvent(
                session,
                makeEvent("tool.execution_start", {
                    toolCallId: "mcp-call-1",
                    toolName: "query_database",
                    arguments: { sql: "SELECT 1" },
                    mcpServerName: "postgres-server",
                })
            );

            dispatchEvent(
                session,
                makeEvent("tool.execution_complete", {
                    toolCallId: "mcp-call-1",
                    success: true,
                    result: { content: "1 row returned" },
                })
            );

            dispatchEvent(session, makeEvent("session.idle"));

            const spans = getSpans();
            const chatSpan = spans.find((s) => s.name.startsWith("chat"))!;
            const outputMessages = JSON.parse(
                chatSpan.attributes["gen_ai.output.messages"] as string
            );
            const inputMessages = JSON.parse(
                chatSpan.attributes["gen_ai.input.messages"] as string
            );

            // Output should have server_tool_call type
            const serverToolCallMsg = outputMessages.find(
                (m: any) => m.parts[0]?.type === "server_tool_call"
            );
            expect(serverToolCallMsg).toBeDefined();
            expect(serverToolCallMsg.parts[0].server_tool_call.type).toBe("mcp");
            expect(serverToolCallMsg.parts[0].server_tool_call.server_name).toBe("postgres-server");
            expect(serverToolCallMsg.parts[0].server_tool_call.arguments).toEqual({
                sql: "SELECT 1",
            });

            // Input should have server_tool_call_response type
            const serverResponseMsg = inputMessages.find(
                (m: any) => m.parts[0]?.type === "server_tool_call_response"
            );
            expect(serverResponseMsg).toBeDefined();
            expect(serverResponseMsg.parts[0].server_tool_call_response.type).toBe("mcp");
            expect(serverResponseMsg.parts[0].server_tool_call_response.server_name).toBe(
                "postgres-server"
            );
        });
    });

    // ========================================================================
    // Vendor-prefixed attributes
    // ========================================================================

    describe("vendor-prefixed attributes", () => {
        it("chat spans get copilot vendor attributes from usage/turn_start events", async () => {
            const telemetry = new CopilotTelemetry({});
            const session = createTestSession(telemetry);

            await sendWithEvent(session, "Hello");
            dispatchEvent(
                session,
                makeEvent("assistant.turn_start", {
                    turnId: "turn-123",
                    interactionId: "interaction-456",
                })
            );
            dispatchEvent(
                session,
                makeEvent("assistant.usage", {
                    model: "gpt-4o",
                    inputTokens: 100,
                    outputTokens: 50,
                    cost: 0.005,
                    duration: 1200,
                    initiator: "user",
                    copilotUsage: {
                        tokenDetails: {
                            batchSize: 1,
                            costPerBatch: 1,
                            tokenCount: 100,
                            tokenType: "input",
                        },
                        totalNanoAiu: 42000,
                    },
                })
            );
            dispatchEvent(session, makeEvent("session.idle"));

            const spans = getSpans();
            const chatSpan = spans.find((s) => s.name.startsWith("chat"))!;
            expect(chatSpan.attributes["github.copilot.cost"]).toBe(0.005);
            expect(chatSpan.attributes["github.copilot.server_duration"]).toBe(1200);
            expect(chatSpan.attributes["github.copilot.initiator"]).toBe("user");
            expect(chatSpan.attributes["github.copilot.aiu"]).toBe(42000);
            expect(chatSpan.attributes["github.copilot.turn_id"]).toBe("turn-123");
            expect(chatSpan.attributes["github.copilot.interaction_id"]).toBe("interaction-456");
        });

        it("invoke_agent spans get accumulated copilot cost and aiu", async () => {
            const telemetry = new CopilotTelemetry({});
            const session = createTestSession(telemetry);

            await sendWithEvent(session, "Hello");

            // Turn 1 with cost and AIU
            dispatchEvent(session, makeEvent("assistant.turn_start", { turnId: "turn-1" }));
            dispatchEvent(
                session,
                makeEvent("assistant.usage", {
                    model: "gpt-4o",
                    inputTokens: 50,
                    outputTokens: 20,
                    cost: 0.003,
                    copilotUsage: {
                        tokenDetails: {
                            batchSize: 1,
                            costPerBatch: 1,
                            tokenCount: 50,
                            tokenType: "input",
                        },
                        totalNanoAiu: 10000,
                    },
                })
            );
            dispatchEvent(session, makeEvent("assistant.turn_end", { turnId: "turn-1" }));

            // Turn 2 with more cost and AIU
            dispatchEvent(session, makeEvent("assistant.turn_start", { turnId: "turn-2" }));
            dispatchEvent(
                session,
                makeEvent("assistant.usage", {
                    model: "gpt-4o",
                    inputTokens: 60,
                    outputTokens: 30,
                    cost: 0.004,
                    copilotUsage: {
                        tokenDetails: {
                            batchSize: 1,
                            costPerBatch: 1,
                            tokenCount: 60,
                            tokenType: "input",
                        },
                        totalNanoAiu: 15000,
                    },
                })
            );
            dispatchEvent(session, makeEvent("session.idle"));

            const spans = getSpans();
            const invokeSpan = spans.find((s) => s.name.startsWith("invoke_agent"))!;
            expect(invokeSpan.attributes["github.copilot.cost"]).toBeCloseTo(0.007, 6);
            expect(invokeSpan.attributes["github.copilot.aiu"]).toBe(25000);
        });
    });

    // ========================================================================
    // Custom span events
    // ========================================================================

    describe("custom span events", () => {
        it("session.truncation dispatches github.copilot.session.truncation span event", async () => {
            const telemetry = new CopilotTelemetry({});
            const session = createTestSession(telemetry);

            await sendWithEvent(session, "Hello");
            dispatchEvent(session, makeEvent("assistant.turn_start", { turnId: "turn-1" }));
            dispatchEvent(
                session,
                makeEvent("session.truncation", {
                    tokenLimit: 8000,
                    preTruncationTokensInMessages: 10000,
                    preTruncationMessagesLength: 20,
                    postTruncationTokensInMessages: 7500,
                    postTruncationMessagesLength: 15,
                    tokensRemovedDuringTruncation: 2500,
                    messagesRemovedDuringTruncation: 5,
                    performedBy: "system",
                })
            );
            dispatchEvent(session, makeEvent("session.idle"));

            const spans = getSpans();
            const chatSpan = spans.find((s) => s.name.startsWith("chat"))!;
            const truncationEvent = chatSpan.events.find(
                (e) => e.name === "github.copilot.session.truncation"
            );
            expect(truncationEvent).toBeDefined();
            expect(truncationEvent!.attributes!["github.copilot.token_limit"]).toBe(8000);
            expect(truncationEvent!.attributes!["github.copilot.pre_tokens"]).toBe(10000);
            expect(truncationEvent!.attributes!["github.copilot.post_tokens"]).toBe(7500);
            expect(truncationEvent!.attributes!["github.copilot.pre_messages"]).toBe(20);
            expect(truncationEvent!.attributes!["github.copilot.post_messages"]).toBe(15);
            expect(truncationEvent!.attributes!["github.copilot.tokens_removed"]).toBe(2500);
            expect(truncationEvent!.attributes!["github.copilot.messages_removed"]).toBe(5);
            expect(truncationEvent!.attributes!["github.copilot.performed_by"]).toBe("system");
        });

        it("session.compaction_start produces the right span event", async () => {
            const telemetry = new CopilotTelemetry({});
            const session = createTestSession(telemetry);

            await sendWithEvent(session, "Hello");
            dispatchEvent(session, makeEvent("assistant.turn_start", { turnId: "turn-1" }));
            dispatchEvent(session, makeEvent("session.compaction_start", {}));
            dispatchEvent(session, makeEvent("session.idle"));

            const spans = getSpans();
            const chatSpan = spans.find((s) => s.name.startsWith("chat"))!;
            const compactionStartEvent = chatSpan.events.find(
                (e) => e.name === "github.copilot.session.compaction_start"
            );
            expect(compactionStartEvent).toBeDefined();
        });

        it("session.compaction_complete produces span event with proper attributes", async () => {
            const telemetry = new CopilotTelemetry({});
            const session = createTestSession(telemetry);

            await sendWithEvent(session, "Hello");
            dispatchEvent(session, makeEvent("assistant.turn_start", { turnId: "turn-1" }));
            dispatchEvent(
                session,
                makeEvent("session.compaction_complete", {
                    success: true,
                    preCompactionTokens: 5000,
                    postCompactionTokens: 2000,
                    tokensRemoved: 3000,
                    messagesRemoved: 8,
                })
            );
            dispatchEvent(session, makeEvent("session.idle"));

            const spans = getSpans();
            const chatSpan = spans.find((s) => s.name.startsWith("chat"))!;
            const compactionEvent = chatSpan.events.find(
                (e) => e.name === "github.copilot.session.compaction_complete"
            );
            expect(compactionEvent).toBeDefined();
            expect(compactionEvent!.attributes!["github.copilot.success"]).toBe(true);
            expect(compactionEvent!.attributes!["github.copilot.pre_tokens"]).toBe(5000);
            expect(compactionEvent!.attributes!["github.copilot.post_tokens"]).toBe(2000);
            expect(compactionEvent!.attributes!["github.copilot.tokens_removed"]).toBe(3000);
            expect(compactionEvent!.attributes!["github.copilot.messages_removed"]).toBe(8);
        });
    });

    // ========================================================================
    // CompleteOnDispose
    // ========================================================================

    describe("completeOnDispose", () => {
        it("completes open spans with error status when session is destroyed", async () => {
            const telemetry = new CopilotTelemetry({});
            const session = createTestSession(telemetry);

            await sendWithEvent(session, "Hello");
            dispatchEvent(session, makeEvent("assistant.turn_start", { turnId: "turn-1" }));

            // Spans should not be finished yet
            expect(getSpans().length).toBe(0);

            // Simulate session destroy - this calls completeOnDispose
            try {
                await session.destroy();
            } catch {
                // sendRequest will fail since mock doesn't handle destroy, that's ok
            }

            const spans = getSpans();
            // Both invoke_agent and chat spans should now be finished
            const invokeSpan = spans.find((s) => s.name.startsWith("invoke_agent"));
            const chatSpan = spans.find((s) => s.name.startsWith("chat"));
            expect(invokeSpan).toBeDefined();
            expect(chatSpan).toBeDefined();
            expect(invokeSpan!.status.code).toBe(SpanStatusCode.ERROR);
            expect(chatSpan!.status.code).toBe(SpanStatusCode.ERROR);
        });
    });

    // ========================================================================
    // Chunk timing with "chat" operation name
    // ========================================================================

    describe("chunk timing operation name", () => {
        it("time_to_first_chunk metrics use chat as the operation name", async () => {
            const telemetry = new CopilotTelemetry({});
            const session = createTestSession(telemetry, {
                model: "gpt-4o",
                streaming: true,
            });

            await sendWithEvent(session, "Hello");
            dispatchEvent(session, makeEvent("assistant.turn_start", { turnId: "turn-1" }));
            dispatchEvent(
                session,
                makeEvent("assistant.message_delta", {
                    messageId: "msg-1",
                    deltaContent: "Hi ",
                })
            );
            dispatchEvent(
                session,
                makeEvent("assistant.message", {
                    messageId: "msg-1",
                    content: "Hi there",
                })
            );
            dispatchEvent(session, makeEvent("session.idle"));

            const scopeMetrics = await getMetrics();
            const ttfcMetric = findMetric(
                scopeMetrics,
                "gen_ai.client.operation.time_to_first_chunk"
            );
            expect(ttfcMetric).toBeDefined();

            // The metric data point should have gen_ai.operation.name = "chat"
            const dataPoint = ttfcMetric.dataPoints[0];
            expect(dataPoint.attributes["gen_ai.operation.name"]).toBe("chat");
        });
    });
});
