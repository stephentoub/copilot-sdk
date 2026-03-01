# OpenTelemetry Instrumentation

The Copilot SDK includes built-in OpenTelemetry instrumentation following the [OpenTelemetry Semantic Conventions for Generative AI systems (v1.40)](https://opentelemetry.io/docs/specs/semconv/gen-ai/). Telemetry is **opt-in** — enable it by providing a `TelemetryConfig` when creating a client. The SDK automatically creates spans, records metrics, and emits span events for agent invocations and tool executions.

## Quick Start

<details open>
<summary><strong>Node.js / TypeScript</strong></summary>

Install the OpenTelemetry SDK packages (the `@opentelemetry/api` peer dependency is included with the Copilot SDK):

```bash
npm install @opentelemetry/sdk-trace-node @opentelemetry/sdk-trace-base @opentelemetry/sdk-metrics
```

```typescript
import { CopilotClient } from "@github/copilot-sdk";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { SimpleSpanProcessor, ConsoleSpanExporter } from "@opentelemetry/sdk-trace-base";

// 1. Set up OpenTelemetry (your exporter of choice)
const provider = new NodeTracerProvider();
provider.addSpanProcessor(new SimpleSpanProcessor(new ConsoleSpanExporter()));
provider.register();

// 2. Enable built-in telemetry on the client
const client = new CopilotClient({
    telemetry: {},  // defaults are fine — or customize below
});
await client.start();

// 3. Use the SDK as usual — spans and metrics are emitted automatically
const session = await client.createSession({ model: "gpt-5" });
const response = await session.sendAndWait({ prompt: "Hello!" });

await session.destroy();
await client.stop();
```

</details>

<details>
<summary><strong>Python</strong></summary>

Install the OpenTelemetry SDK packages:

```bash
pip install opentelemetry-sdk opentelemetry-api
```

```python
import asyncio
from copilot import CopilotClient
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter

# 1. Set up OpenTelemetry
provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(provider)

# 2. Enable built-in telemetry on the client
client = CopilotClient({"telemetry": {}})
await client.start()

# 3. Use the SDK as usual
session = await client.create_session({"model": "gpt-5"})
response = await session.send_and_wait({"prompt": "Hello!"})

await session.destroy()
await client.stop()
```

</details>

<details>
<summary><strong>Go</strong></summary>

Install the OpenTelemetry SDK packages:

```bash
go get go.opentelemetry.io/otel
go get go.opentelemetry.io/otel/sdk/trace
go get go.opentelemetry.io/otel/exporters/stdout/stdouttrace
```

```go
package main

import (
    "context"
    "log"

    copilot "github.com/github/copilot-sdk/go"
    "go.opentelemetry.io/otel"
    sdktrace "go.opentelemetry.io/otel/sdk/trace"
    "go.opentelemetry.io/otel/exporters/stdout/stdouttrace"
)

func main() {
    // 1. Set up OpenTelemetry
    exporter, _ := stdouttrace.New()
    tp := sdktrace.NewTracerProvider(sdktrace.WithBatcher(exporter))
    otel.SetTracerProvider(tp)
    defer tp.Shutdown(context.Background())

    // 2. Enable built-in telemetry on the client
    client := copilot.NewClient(&copilot.ClientOptions{
        Telemetry: &copilot.TelemetryConfig{},
    })
    if err := client.Start(context.Background()); err != nil {
        log.Fatal(err)
    }
    defer client.Stop()

    // 3. Use the SDK as usual
    session, _ := client.CreateSession(context.Background(), &copilot.SessionConfig{
        Model: "gpt-5",
    })
    defer session.Destroy()

    session.SendAndWait(context.Background(), copilot.MessageOptions{
        Prompt: "Hello!",
    })
}
```

</details>

<details>
<summary><strong>.NET</strong></summary>

Install the OpenTelemetry SDK packages:

```bash
dotnet add package OpenTelemetry
dotnet add package OpenTelemetry.Exporter.Console
```

<!-- docs-validate: skip -->
```csharp
using GitHub.Copilot.SDK;
using OpenTelemetry;
using OpenTelemetry.Trace;

// 1. Set up OpenTelemetry — add the SDK's ActivitySource
using var tracerProvider = Sdk.CreateTracerProviderBuilder()
    .AddSource("github.copilot.sdk")   // matches the default source name
    .AddConsoleExporter()
    .Build();

// 2. Enable built-in telemetry on the client
await using var client = new CopilotClient(new CopilotClientOptions
{
    Telemetry = new TelemetryConfig()
});
await client.StartAsync();

// 3. Use the SDK as usual
await using var session = await client.CreateSessionAsync(new SessionConfig
{
    Model = "gpt-5"
});
await session.SendAndWaitAsync(new MessageOptions { Prompt = "Hello!" });
```

</details>

## Configuration

All languages accept the same two options:

| Option | Default | Description |
|--------|---------|-------------|
| `enableSensitiveData` | `false` | Include potentially sensitive data (message content, tool arguments/results, system instructions) in telemetry. Falls back to the `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` environment variable when not set. |
| `sourceName` | `"github.copilot.sdk"` | Name used for the tracer and meter. Use this to distinguish multiple SDK instances or match your OpenTelemetry pipeline filters. |

> **Language-specific option casing:**
> Node.js uses `enableSensitiveData` / `sourceName` (camelCase).
> Python uses `enable_sensitive_data` / `source_name` (snake_case).
> Go uses `EnableSensitiveData` / `SourceName` (PascalCase).
> .NET uses `EnableSensitiveData` / `SourceName` (PascalCase).

### Enabling Sensitive Data

By default, message content, tool arguments, tool results, and system instructions are **not** included in telemetry to protect potentially sensitive data. To include them:

**Option 1 — Per-client configuration:**

<!-- docs-validate: skip -->
```typescript
// Node.js
const client = new CopilotClient({
    telemetry: { enableSensitiveData: true },
});
```

<!-- docs-validate: skip -->
```python
# Python
client = CopilotClient({"telemetry": {"enable_sensitive_data": True}})
```

<!-- docs-validate: skip -->
```go
// Go
client := copilot.NewClient(&copilot.ClientOptions{
    Telemetry: &copilot.TelemetryConfig{
        EnableSensitiveData: copilot.Bool(true),
    },
})
```

<!-- docs-validate: skip -->
```csharp
// .NET
var client = new CopilotClient(new CopilotClientOptions
{
    Telemetry = new TelemetryConfig { EnableSensitiveData = true }
});
```

**Option 2 — Environment variable (applies to all clients):**

```bash
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true
```

## Agent Attribution

You can associate sessions with a named agent for telemetry attribution using `agentName` and `agentDescription` on the session config. When set, the `invoke_agent` span includes `gen_ai.agent.name` and `gen_ai.agent.description` attributes.

<!-- docs-validate: skip -->
```typescript
// Node.js
const session = await client.createSession({
    model: "gpt-5",
    agentName: "weather-bot",
    agentDescription: "An agent that provides weather forecasts",
});
```

<!-- docs-validate: skip -->
```python
# Python
session = await client.create_session({
    "model": "gpt-5",
    "agent_name": "weather-bot",
    "agent_description": "An agent that provides weather forecasts",
})
```

<!-- docs-validate: skip -->
```go
// Go
session, _ := client.CreateSession(ctx, &copilot.SessionConfig{
    Model:            "gpt-5",
    AgentName:        "weather-bot",
    AgentDescription: "An agent that provides weather forecasts",
})
```

<!-- docs-validate: skip -->
```csharp
// .NET
var session = await client.CreateSessionAsync(new SessionConfig
{
    Model = "gpt-5",
    AgentName = "weather-bot",
    AgentDescription = "An agent that provides weather forecasts",
});
```

## Emitted Telemetry

### Spans

The SDK automatically creates the following spans:

#### `invoke_agent` (Client span)

Created on the first `send` / `sendAndWait` call after the session becomes idle and reused across subsequent `send` / `sendAndWait` calls in the same turn. Ends when a turn-ending event is emitted (e.g., `session.idle` or `session.error`). Named `invoke_agent {model}` when a model is known, or just `invoke_agent`.

| Attribute | Description | Condition |
|-----------|-------------|-----------|
| `gen_ai.operation.name` | `"invoke_agent"` | Always |
| `gen_ai.provider.name` | Provider name (e.g., `"github"`, `"openai"`, `"azure.ai.openai"`, `"anthropic"`) | Always |
| `gen_ai.agent.id` | Session ID | Always |
| `gen_ai.conversation.id` | Session ID | Always |
| `gen_ai.request.model` | Requested model name | When model is set |
| `gen_ai.response.model` | Actual model used (from usage event) | When reported |
| `gen_ai.agent.name` | Agent name | When `agentName` is set |
| `gen_ai.agent.description` | Agent description | When `agentDescription` is set |
| `gen_ai.usage.input_tokens` | Input token count | When reported |
| `gen_ai.usage.output_tokens` | Output token count | When reported |
| `gen_ai.response.finish_reasons` | `["stop"]` or `["error"]` | At span end |
| `server.address` | Provider host | When using custom provider |
| `server.port` | Provider port | When using custom provider |
| `error.type` | Error type name | On error |
| `gen_ai.input.messages` | JSON input messages | When `enableSensitiveData` is true |
| `gen_ai.output.messages` | JSON output messages | When `enableSensitiveData` is true |
| `gen_ai.system_instructions` | System message content | When `enableSensitiveData` is true |
| `gen_ai.tool.definitions` | JSON tool definitions | Always (non-sensitive) |

#### `execute_tool` (Internal span)

Created as a child of `invoke_agent` for each custom tool call. Named `execute_tool {toolName}`.

| Attribute | Description | Condition |
|-----------|-------------|-----------|
| `gen_ai.operation.name` | `"execute_tool"` | Always |
| `gen_ai.tool.name` | Tool name | Always |
| `gen_ai.tool.call.id` | Unique call ID | Always |
| `gen_ai.tool.type` | `"function"` | Always |
| `gen_ai.tool.description` | Tool description | When available |
| `gen_ai.tool.call.arguments` | JSON arguments | When `enableSensitiveData` is true |
| `gen_ai.tool.call.result` | JSON result | When `enableSensitiveData` is true |
| `error.type` | Error type name | On error |

### Metrics

The SDK records the following metrics (all using the configured `sourceName` as the meter name):

| Metric | Type | Unit | Description |
|--------|------|------|-------------|
| `gen_ai.client.operation.duration` | Histogram (float) | `s` | Duration of `invoke_agent` and `execute_tool` operations |
| `gen_ai.client.token.usage` | Histogram (int) | `{token}` | Token usage per operation, with `gen_ai.token.type` attribute (`"input"` or `"output"`) |

## Exporter Setup

The SDK uses the standard OpenTelemetry API — configure any exporter compatible with your language's OpenTelemetry SDK.

### OTLP (Jaeger, Grafana, etc.)

<!-- docs-validate: skip -->
```bash
# Node.js
npm install @opentelemetry/exporter-trace-otlp-http

# Python
pip install opentelemetry-exporter-otlp

# Go
go get go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracehttp

# .NET
dotnet add package OpenTelemetry.Exporter.OpenTelemetryProtocol
```

### Azure Monitor

<!-- docs-validate: skip -->
```bash
# Python
pip install azure-monitor-opentelemetry

# .NET
dotnet add package Azure.Monitor.OpenTelemetry.Exporter
```

<!-- docs-validate: skip -->
```python
# Python — Azure Monitor
from azure.monitor.opentelemetry import configure_azure_monitor
configure_azure_monitor(connection_string="InstrumentationKey=...")
```

<!-- docs-validate: skip -->
```csharp
// .NET — Azure Monitor
using var tracerProvider = Sdk.CreateTracerProviderBuilder()
    .AddSource("github.copilot.sdk")
    .AddAzureMonitorTraceExporter(o => o.ConnectionString = "InstrumentationKey=...")
    .Build();
```

## Troubleshooting

### No spans appearing

1. Verify the OpenTelemetry provider is registered before creating the `CopilotClient`.
2. Ensure your exporter's source/activity filter includes the SDK's source name (default: `"github.copilot.sdk"`). For .NET, this means calling `.AddSource("github.copilot.sdk")` on the tracer provider builder.
3. Confirm `telemetry` is set on the client options — when omitted, no telemetry is emitted.

### Missing message content or tool arguments

Sensitive attributes are gated behind `enableSensitiveData`. Set it to `true` in the `TelemetryConfig` or set the `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true` environment variable.

## References

- [OpenTelemetry GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [OpenTelemetry MCP Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/mcp/)
- [Copilot SDK Documentation](https://github.com/github/copilot-sdk)
