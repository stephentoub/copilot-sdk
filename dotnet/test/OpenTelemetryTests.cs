/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------------------------------------------*/

using System.Diagnostics;
using System.Diagnostics.Metrics;
using System.Linq;
using GitHub.Copilot.SDK.Test.Harness;
using Microsoft.Extensions.AI;
using System.ComponentModel;
using Xunit;

namespace GitHub.Copilot.SDK.Test;

/// <summary>
/// E2E tests for OpenTelemetry instrumentation that validate spans, events,
/// metrics, sensitive data gating, and configuration through public APIs.
/// All tests reuse the "invokes_custom_tool" snapshot since telemetry is
/// purely client-side and does not affect the proxy traffic.
/// </summary>
public class OpenTelemetryE2ETests(E2ETestFixture fixture)
    : IClassFixture<E2ETestFixture>, IAsyncLifetime
{
    private E2ETestContext Ctx => fixture.Ctx;

    public Task InitializeAsync() =>
        Ctx.ConfigureForTestAsync("tools", "Invokes_Custom_Tool");

    public Task DisposeAsync() => Task.CompletedTask;

    [Fact]
    public async Task Records_Invoke_Agent_And_Execute_Tool_Spans()
    {
        var activities = new List<Activity>();
        using var activityListener = CreateActivityListener("github.copilot.sdk", activities);

        var tokenRecordings = new List<(int Value, KeyValuePair<string, object?>[] Tags)>();
        var durationRecordings = new List<(double Value, KeyValuePair<string, object?>[] Tags)>();
        using var meterListener = CreateMeterListener("github.copilot.sdk", tokenRecordings, durationRecordings);

        await using var client = Ctx.CreateClient(new TelemetryConfig());
        var session = await client.CreateSessionAsync(new SessionConfig
        {
            Model = "gpt-4o",
            Tools = [AIFunctionFactory.Create(EncryptString, "encrypt_string")],
            OnPermissionRequest = PermissionHandler.ApproveAll,
        });

        var response = await session.SendAndWaitAsync(new MessageOptions
        {
            Prompt = "Use encrypt_string to encrypt this string: Hello"
        });

        Assert.NotNull(response);
        Assert.Contains("HELLO", response!.Data.Content ?? string.Empty);

        // -- invoke_agent span --
        var invokeActivity = Assert.Single(activities, a =>
            a.GetTagItem("gen_ai.operation.name")?.ToString() == "invoke_agent");
        Assert.Equal(ActivityKind.Client, invokeActivity.Kind);
        Assert.Equal(session.SessionId, invokeActivity.GetTagItem("gen_ai.conversation.id")?.ToString());
        Assert.Equal(session.SessionId, invokeActivity.GetTagItem("gen_ai.agent.id")?.ToString());
        Assert.NotNull(invokeActivity.GetTagItem("gen_ai.provider.name"));

        // -- gen_ai.request.model --
        Assert.Equal("gpt-4o", invokeActivity.GetTagItem("gen_ai.request.model")?.ToString());

        // -- server.address (conditional — depends on provider config) --
        if (invokeActivity.GetTagItem("server.address") is not null)
        {
            Assert.IsType<string>(invokeActivity.GetTagItem("server.address"));
        }

        // -- chat child spans (one per LLM turn) --
        var chatActivities = activities.Where(a =>
            a.GetTagItem("gen_ai.operation.name")?.ToString() == "chat").ToList();
        Assert.NotEmpty(chatActivities);
        Assert.All(chatActivities, chatActivity =>
        {
            Assert.Equal(ActivityKind.Client, chatActivity.Kind);
            Assert.Equal(invokeActivity.TraceId, chatActivity.TraceId);
            Assert.Equal(invokeActivity.SpanId, chatActivity.ParentSpanId);
        });

        // -- gen_ai.response.model, gen_ai.response.id, token counts (on chat spans) --
        var firstChat = chatActivities[0];
        if (firstChat.GetTagItem("gen_ai.response.model") is not null)
        {
            Assert.IsType<string>(firstChat.GetTagItem("gen_ai.response.model"));
        }
        if (firstChat.GetTagItem("gen_ai.response.id") is not null)
        {
            Assert.IsType<string>(firstChat.GetTagItem("gen_ai.response.id"));
        }

        // -- execute_tool span --
        var toolActivity = Assert.Single(activities, a =>
            a.GetTagItem("gen_ai.operation.name")?.ToString() == "execute_tool");
        Assert.Equal(ActivityKind.Internal, toolActivity.Kind);
        Assert.Equal("encrypt_string", toolActivity.GetTagItem("gen_ai.tool.name")?.ToString());
        Assert.Equal("function", toolActivity.GetTagItem("gen_ai.tool.type")?.ToString());
        Assert.NotNull(toolActivity.GetTagItem("gen_ai.tool.call.id"));

        // -- execute_tool is a child of invoke_agent --
        Assert.Equal(invokeActivity.TraceId, toolActivity.TraceId);
        Assert.Equal(invokeActivity.SpanId, toolActivity.ParentSpanId);

        // -- gen_ai.tool.description --
        Assert.NotNull(toolActivity.GetTagItem("gen_ai.tool.description"));

        // -- duration metrics --
        Assert.NotEmpty(durationRecordings);
        Assert.Contains(durationRecordings, r =>
            r.Tags.Any(t => t.Key == "gen_ai.operation.name" && t.Value?.ToString() == "invoke_agent") &&
            r.Value > 0);
        Assert.Contains(durationRecordings, r =>
            r.Tags.Any(t => t.Key == "gen_ai.operation.name" && t.Value?.ToString() == "execute_tool") &&
            r.Value > 0);
        Assert.Contains(durationRecordings, r =>
            r.Tags.Any(t => t.Key == "gen_ai.operation.name" && t.Value?.ToString() == "chat") &&
            r.Value > 0);

        // Duration metrics should include provider tag
        var invokeMetric = durationRecordings.First(r =>
            r.Tags.Any(t => t.Key == "gen_ai.operation.name" && t.Value?.ToString() == "invoke_agent"));
        Assert.Contains(invokeMetric.Tags, t => t.Key == "gen_ai.provider.name" && t.Value is not null);

        // -- token usage metrics (conditional — depends on proxy response) --
        if (tokenRecordings.Count > 0)
        {
            Assert.All(tokenRecordings, r =>
            {
                var tokenType = r.Tags.FirstOrDefault(t => t.Key == "gen_ai.token.type").Value?.ToString();
                Assert.True(tokenType is "input" or "output",
                    $"Expected token type 'input' or 'output', got '{tokenType}'");
                Assert.Contains(r.Tags, t => t.Key == "gen_ai.operation.name" && t.Value?.ToString() == "chat");
                Assert.Contains(r.Tags, t => t.Key == "gen_ai.provider.name" && t.Value is not null);
            });
        }
    }

    [Fact]
    public async Task Sensitive_Data_Includes_Tool_Content()
    {
        var activities = new List<Activity>();
        using var activityListener = CreateActivityListener("github.copilot.sdk", activities);

        await using var client = Ctx.CreateClient(new TelemetryConfig { EnableSensitiveData = true });
        var session = await client.CreateSessionAsync(new SessionConfig
        {
            Tools = [AIFunctionFactory.Create(EncryptString, "encrypt_string")],
            OnPermissionRequest = PermissionHandler.ApproveAll,
        });

        var response = await session.SendAndWaitAsync(new MessageOptions
        {
            Prompt = "Use encrypt_string to encrypt this string: Hello"
        });

        Assert.NotNull(response);

        // Tool arguments are set on the execute_tool span
        var toolActivity = Assert.Single(activities, a =>
            a.GetTagItem("gen_ai.operation.name")?.ToString() == "execute_tool");
        var toolArgs = toolActivity.GetTagItem("gen_ai.tool.call.arguments")?.ToString();
        Assert.NotNull(toolArgs);
        Assert.Contains("Hello", toolArgs);

        // Tool result is set
        var toolResult = toolActivity.GetTagItem("gen_ai.tool.call.result")?.ToString();
        Assert.NotNull(toolResult);
        Assert.Contains("HELLO", toolResult);
    }

    [Fact]
    public async Task No_Telemetry_When_Config_Is_Null()
    {
        var activities = new List<Activity>();
        using var activityListener = CreateActivityListener("github.copilot.sdk", activities);

        var tokenRecordings = new List<(int Value, KeyValuePair<string, object?>[] Tags)>();
        var durationRecordings = new List<(double Value, KeyValuePair<string, object?>[] Tags)>();
        using var meterListener = CreateMeterListener("github.copilot.sdk", tokenRecordings, durationRecordings);

        await using var client = Ctx.CreateClient(telemetry: null);
        var session = await client.CreateSessionAsync(new SessionConfig
        {
            Tools = [AIFunctionFactory.Create(EncryptString, "encrypt_string")],
            OnPermissionRequest = PermissionHandler.ApproveAll,
        });

        var response = await session.SendAndWaitAsync(new MessageOptions
        {
            Prompt = "Use encrypt_string to encrypt this string: Hello"
        });

        Assert.NotNull(response);
        Assert.Contains("HELLO", response!.Data.Content ?? string.Empty);

        // No activities or metrics emitted
        Assert.Empty(activities);
        Assert.Empty(durationRecordings);
        Assert.Empty(tokenRecordings);
    }

    [Fact]
    public async Task Custom_Source_Name_Used()
    {
        const string customSourceName = "my.custom.source";

        var activities = new List<Activity>();
        using var activityListener = CreateActivityListener(customSourceName, activities);

        var durationRecordings = new List<(double Value, KeyValuePair<string, object?>[] Tags)>();
        using var meterListener = CreateMeterListener(customSourceName,
            new List<(int, KeyValuePair<string, object?>[])>(), durationRecordings);

        await using var client = Ctx.CreateClient(new TelemetryConfig { SourceName = customSourceName });
        var session = await client.CreateSessionAsync(new SessionConfig
        {
            Tools = [AIFunctionFactory.Create(EncryptString, "encrypt_string")],
            OnPermissionRequest = PermissionHandler.ApproveAll,
        });

        await session.SendAndWaitAsync(new MessageOptions
        {
            Prompt = "Use encrypt_string to encrypt this string: Hello"
        });

        Assert.NotEmpty(activities);
        Assert.All(activities, a => Assert.Equal(customSourceName, a.Source.Name));
        Assert.NotEmpty(durationRecordings);
    }

    [Fact]
    public async Task Default_Source_Name_Not_Captured_When_Custom_Name_Used()
    {
        // Verify that the default source name does NOT emit when a custom name is configured
        var defaultActivities = new List<Activity>();
        using var defaultListener = CreateActivityListener("github.copilot.sdk", defaultActivities);

        await using var client = Ctx.CreateClient(new TelemetryConfig { SourceName = "custom.telemetry" });
        var session = await client.CreateSessionAsync(new SessionConfig
        {
            Tools = [AIFunctionFactory.Create(EncryptString, "encrypt_string")],
            OnPermissionRequest = PermissionHandler.ApproveAll,
        });

        await session.SendAndWaitAsync(new MessageOptions
        {
            Prompt = "Use encrypt_string to encrypt this string: Hello"
        });

        Assert.Empty(defaultActivities);
    }

    [Fact]
    public async Task Successful_Turn_Has_Finish_Reasons()
    {
        var activities = new List<Activity>();
        using var activityListener = CreateActivityListener("github.copilot.sdk", activities);

        await using var client = Ctx.CreateClient(new TelemetryConfig());
        var session = await client.CreateSessionAsync(new SessionConfig
        {
            Tools = [AIFunctionFactory.Create(EncryptString, "encrypt_string")],
            OnPermissionRequest = PermissionHandler.ApproveAll,
        });

        var response = await session.SendAndWaitAsync(new MessageOptions
        {
            Prompt = "Use encrypt_string to encrypt this string: Hello"
        });

        Assert.NotNull(response);

        // Verify finish_reasons are set on the invoke_agent span
        var invokeActivity = Assert.Single(activities, a =>
            a.GetTagItem("gen_ai.operation.name")?.ToString() == "invoke_agent");
        var finishReasons = invokeActivity.GetTagItem("gen_ai.response.finish_reasons") as string[];
        Assert.NotNull(finishReasons);
        Assert.Contains("stop", finishReasons);
    }

    [Fact]
    public async Task Env_Var_Enables_Sensitive_Data()
    {
        try
        {
            Environment.SetEnvironmentVariable("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "true");

            var activities = new List<Activity>();
            using var activityListener = CreateActivityListener("github.copilot.sdk", activities);

            // Client created with default TelemetryConfig (no explicit EnableSensitiveData)
            // but the env var should enable it
            await using var client = Ctx.CreateClient(new TelemetryConfig());
            var session = await client.CreateSessionAsync(new SessionConfig
            {
                Tools = [AIFunctionFactory.Create(EncryptString, "encrypt_string")],
                OnPermissionRequest = PermissionHandler.ApproveAll,
            });

            var response = await session.SendAndWaitAsync(new MessageOptions
            {
                Prompt = "Use encrypt_string to encrypt this string: Hello"
            });

            Assert.NotNull(response);

            // Tool arguments should be present due to env var enabling sensitive data
            var toolActivity = Assert.Single(activities, a =>
                a.GetTagItem("gen_ai.operation.name")?.ToString() == "execute_tool");
            var toolArgs = toolActivity.GetTagItem("gen_ai.tool.call.arguments");
            Assert.NotNull(toolArgs);
        }
        finally
        {
            Environment.SetEnvironmentVariable("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", null);
        }
    }

    [Fact]
    public async Task Sensitive_Data_Disabled_Excludes_Tool_Content()
    {
        var activities = new List<Activity>();
        using var activityListener = CreateActivityListener("github.copilot.sdk", activities);

        await using var client = Ctx.CreateClient(new TelemetryConfig { EnableSensitiveData = false });
        var session = await client.CreateSessionAsync(new SessionConfig
        {
            Tools = [AIFunctionFactory.Create(EncryptString, "encrypt_string")],
            OnPermissionRequest = PermissionHandler.ApproveAll,
        });

        var response = await session.SendAndWaitAsync(new MessageOptions
        {
            Prompt = "Use encrypt_string to encrypt this string: Hello"
        });

        Assert.NotNull(response);

        // Tool arguments and results should NOT be present
        var toolActivity = Assert.Single(activities, a =>
            a.GetTagItem("gen_ai.operation.name")?.ToString() == "execute_tool");
        Assert.Null(toolActivity.GetTagItem("gen_ai.tool.call.arguments"));
        Assert.Null(toolActivity.GetTagItem("gen_ai.tool.call.result"));
    }

    [Fact]
    public async Task Sensitive_Data_Includes_Message_Content()
    {
        var activities = new List<Activity>();
        using var activityListener = CreateActivityListener("github.copilot.sdk", activities);

        await using var client = Ctx.CreateClient(new TelemetryConfig { EnableSensitiveData = true });
        var session = await client.CreateSessionAsync(new SessionConfig
        {
            Model = "gpt-4o",
            SystemMessage = new SystemMessageConfig { Content = "You are a helpful assistant." },
            Tools = [AIFunctionFactory.Create(EncryptString, "encrypt_string")],
            OnPermissionRequest = PermissionHandler.ApproveAll,
        });

        var response = await session.SendAndWaitAsync(new MessageOptions
        {
            Prompt = "Use encrypt_string to encrypt this string: Hello"
        });

        Assert.NotNull(response);

        var invokeActivity = Assert.Single(activities, a =>
            a.GetTagItem("gen_ai.operation.name")?.ToString() == "invoke_agent");

        // Messages are on chat child spans, not invoke_agent
        var chatActivities = activities.Where(a =>
            a.GetTagItem("gen_ai.operation.name")?.ToString() == "chat").ToList();
        Assert.NotEmpty(chatActivities);

        // Input messages should be present on at least one chat span
        var chatWithInput = chatActivities.FirstOrDefault(a =>
            a.GetTagItem("gen_ai.input.messages")?.ToString() is not null);
        Assert.NotNull(chatWithInput);
        var inputMessages = chatWithInput!.GetTagItem("gen_ai.input.messages")!.ToString()!;
        Assert.Contains("Hello", inputMessages);
        Assert.Contains("\"role\":\"user\"", inputMessages);

        // Tool results should appear as input messages with role "tool"
        var allInputMessages = string.Join(" ", chatActivities
            .Select(a => a.GetTagItem("gen_ai.input.messages")?.ToString() ?? ""));
        Assert.Contains("\"role\":\"tool\"", allInputMessages);
        Assert.Contains("\"type\":\"tool_call_response\"", allInputMessages);

        // Output messages should be present on at least one chat span
        var allOutputMessages = string.Join(" ", chatActivities
            .Select(a => a.GetTagItem("gen_ai.output.messages")?.ToString() ?? ""));
        Assert.Contains("\"role\":\"assistant\"", allOutputMessages);

        // Tool call requests should appear in output messages
        Assert.Contains("\"type\":\"tool_call\"", allOutputMessages);
        Assert.Contains("encrypt_string", allOutputMessages);

        // System instructions should be present on invoke_agent
        var sysInstructions = invokeActivity.GetTagItem("gen_ai.system_instructions")?.ToString();
        Assert.NotNull(sysInstructions);
        Assert.Contains("You are a helpful assistant.", sysInstructions);

        // Tool definitions should be present on invoke_agent
        var toolDefs = invokeActivity.GetTagItem("gen_ai.tool.definitions")?.ToString();
        Assert.NotNull(toolDefs);
        Assert.Contains("encrypt_string", toolDefs);
    }

    [Fact]
    public async Task Sensitive_Data_Disabled_Excludes_Message_Content()
    {
        var activities = new List<Activity>();
        using var activityListener = CreateActivityListener("github.copilot.sdk", activities);

        await using var client = Ctx.CreateClient(new TelemetryConfig { EnableSensitiveData = false });
        var session = await client.CreateSessionAsync(new SessionConfig
        {
            Model = "gpt-4o",
            SystemMessage = new SystemMessageConfig { Content = "You are a helpful assistant." },
            Tools = [AIFunctionFactory.Create(EncryptString, "encrypt_string")],
            OnPermissionRequest = PermissionHandler.ApproveAll,
        });

        var response = await session.SendAndWaitAsync(new MessageOptions
        {
            Prompt = "Use encrypt_string to encrypt this string: Hello"
        });

        Assert.NotNull(response);

        var invokeActivity = Assert.Single(activities, a =>
            a.GetTagItem("gen_ai.operation.name")?.ToString() == "invoke_agent");

        // Sensitive attributes should be excluded from invoke_agent
        Assert.Null(invokeActivity.GetTagItem("gen_ai.input.messages"));
        Assert.Null(invokeActivity.GetTagItem("gen_ai.output.messages"));
        Assert.Null(invokeActivity.GetTagItem("gen_ai.system_instructions"));

        // Sensitive attributes should be excluded from chat spans too
        var chatActivities = activities.Where(a =>
            a.GetTagItem("gen_ai.operation.name")?.ToString() == "chat").ToList();
        Assert.All(chatActivities, chatActivity =>
        {
            Assert.Null(chatActivity.GetTagItem("gen_ai.input.messages"));
            Assert.Null(chatActivity.GetTagItem("gen_ai.output.messages"));
        });

        // Tool definitions should still be present (not sensitive-gated)
        var toolDefs = invokeActivity.GetTagItem("gen_ai.tool.definitions")?.ToString();
        Assert.NotNull(toolDefs);
        Assert.Contains("encrypt_string", toolDefs);
    }

    [Fact]
    public async Task Records_Subagent_Invoke_Agent_And_Chat_Spans()
    {
        // Use a separate snapshot for the subagent scenario.
        await Ctx.ConfigureForTestAsync("otel", "Records_Subagent_Invoke_Agent_And_Chat_Spans");

        var activities = new List<Activity>();
        using var activityListener = CreateActivityListener("github.copilot.sdk", activities);

        var tokenRecordings = new List<(int Value, KeyValuePair<string, object?>[] Tags)>();
        var durationRecordings = new List<(double Value, KeyValuePair<string, object?>[] Tags)>();
        using var meterListener = CreateMeterListener("github.copilot.sdk", tokenRecordings, durationRecordings);

        int topicCallCount = 0;
        string GetTopic() => Interlocked.Increment(ref topicCallCount) == 1 ? "mountains" : "rivers";

        await using var client = Ctx.CreateClient(new TelemetryConfig { EnableSensitiveData = true });
        var session = await client.CreateSessionAsync(new SessionConfig
        {
            Tools = [AIFunctionFactory.Create(GetTopic, "get_topic", "Returns a poem topic")],
            OnPermissionRequest = PermissionHandler.ApproveAll,
        });

        var response = await session.SendAndWaitAsync(new MessageOptions
        {
            Prompt = "Call get_topic twice to get two different topics, then launch two background " +
                     "general-purpose subagents in parallel (one per topic) to each write a short " +
                     "four-line poem about their assigned topic. Use read_agent to wait for both " +
                     "results and show them to me."
        });

        Assert.NotNull(response);

        // =====================================================================
        // 1. Root invoke_agent span
        // =====================================================================
        var allInvokeActivities = activities.Where(a =>
            a.GetTagItem("gen_ai.operation.name")?.ToString() == "invoke_agent").ToList();
        Assert.True(allInvokeActivities.Count >= 3,
            $"Expected ≥3 invoke_agent spans (1 root + ≥2 subagents), got {allInvokeActivities.Count}");

        // The root invoke_agent has no parent invoke_agent
        var mainInvoke = allInvokeActivities.First(a =>
            !allInvokeActivities.Any(p => p.SpanId == a.ParentSpanId));
        Assert.Equal(ActivityKind.Client, mainInvoke.Kind);
        Assert.Equal("invoke_agent", mainInvoke.GetTagItem("gen_ai.operation.name")?.ToString());
        Assert.Equal(session.SessionId, mainInvoke.GetTagItem("gen_ai.conversation.id")?.ToString());
        Assert.Equal(session.SessionId, mainInvoke.GetTagItem("gen_ai.agent.id")?.ToString());
        Assert.NotNull(mainInvoke.GetTagItem("gen_ai.provider.name"));

        // Root invoke_agent finish_reasons should be "stop"
        var rootFinish = mainInvoke.GetTagItem("gen_ai.response.finish_reasons") as string[];
        Assert.NotNull(rootFinish);
        Assert.Contains("stop", rootFinish);

        // Tool definitions should be present (not sensitive-gated)
        var toolDefs = mainInvoke.GetTagItem("gen_ai.tool.definitions")?.ToString();
        Assert.NotNull(toolDefs);
        Assert.Contains("get_topic", toolDefs);

        // Root invoke_agent should have accumulated input messages (sensitive data on)
        var rootInputMessages = mainInvoke.GetTagItem("gen_ai.input.messages")?.ToString();
        Assert.NotNull(rootInputMessages);
        Assert.Contains("\"role\":\"user\"", rootInputMessages);

        // Root invoke_agent should have accumulated output messages
        var rootOutputMessages = mainInvoke.GetTagItem("gen_ai.output.messages")?.ToString();
        Assert.NotNull(rootOutputMessages);
        Assert.Contains("\"role\":\"assistant\"", rootOutputMessages);

        // =====================================================================
        // 2. Subagent invoke_agent spans (≥2, children of root)
        // =====================================================================
        var subagentInvokeActivities = allInvokeActivities.Where(a =>
            a.ParentSpanId == mainInvoke.SpanId).ToList();
        Assert.True(subagentInvokeActivities.Count >= 2,
            $"Expected ≥2 subagent invoke_agent spans, got {subagentInvokeActivities.Count}");

        Assert.All(subagentInvokeActivities, sub =>
        {
            Assert.Equal(ActivityKind.Client, sub.Kind);
            Assert.Equal(mainInvoke.TraceId, sub.TraceId);
            Assert.Equal("invoke_agent", sub.GetTagItem("gen_ai.operation.name")?.ToString());
            Assert.NotNull(sub.GetTagItem("gen_ai.provider.name"));

            // Subagent invoke_agent should have finish_reasons
            var subFinish = sub.GetTagItem("gen_ai.response.finish_reasons") as string[];
            Assert.NotNull(subFinish);
            Assert.Contains("stop", subFinish);

            // Subagent invoke_agent should have output messages (the poem)
            var subOutputMessages = sub.GetTagItem("gen_ai.output.messages")?.ToString();
            Assert.NotNull(subOutputMessages);
            Assert.Contains("\"role\":\"assistant\"", subOutputMessages);
        });

        // =====================================================================
        // 3. Main agent chat spans (children of root invoke_agent)
        // =====================================================================
        var mainChatActivities = activities.Where(a =>
            a.GetTagItem("gen_ai.operation.name")?.ToString() == "chat" &&
            a.ParentSpanId == mainInvoke.SpanId).ToList();
        Assert.NotEmpty(mainChatActivities);

        Assert.All(mainChatActivities, chat =>
        {
            Assert.Equal(ActivityKind.Client, chat.Kind);
            Assert.Equal(mainInvoke.TraceId, chat.TraceId);
            Assert.Equal("chat", chat.GetTagItem("gen_ai.operation.name")?.ToString());
            Assert.NotNull(chat.GetTagItem("gen_ai.provider.name"));
        });

        // At least one main chat span should have response model/id
        var mainChatWithModel = mainChatActivities.FirstOrDefault(a =>
            a.GetTagItem("gen_ai.response.model") is not null);
        if (mainChatWithModel is not null)
        {
            Assert.IsType<string>(mainChatWithModel.GetTagItem("gen_ai.response.model"));
            Assert.IsType<string>(mainChatWithModel.GetTagItem("gen_ai.response.id"));
        }

        // Main chat output messages should contain tool_call for "task"
        var mainChatOutputs = string.Join(" ", mainChatActivities
            .Select(a => a.GetTagItem("gen_ai.output.messages")?.ToString() ?? ""));
        Assert.Contains("\"type\":\"tool_call\"", mainChatOutputs);
        Assert.Contains("get_topic", mainChatOutputs);

        // Main chat input messages should contain user prompt and tool responses
        var mainChatInputs = string.Join(" ", mainChatActivities
            .Select(a => a.GetTagItem("gen_ai.input.messages")?.ToString() ?? ""));
        Assert.Contains("\"role\":\"user\"", mainChatInputs);
        Assert.Contains("\"role\":\"tool\"", mainChatInputs);
        Assert.Contains("\"type\":\"tool_call_response\"", mainChatInputs);

        // =====================================================================
        // 4. Subagent chat spans (each parented to its subagent invoke_agent)
        // =====================================================================
        foreach (var subInvoke in subagentInvokeActivities)
        {
            var subChatActivities = activities.Where(a =>
                a.GetTagItem("gen_ai.operation.name")?.ToString() == "chat" &&
                a.ParentSpanId == subInvoke.SpanId).ToList();
            Assert.NotEmpty(subChatActivities);

            Assert.All(subChatActivities, chat =>
            {
                Assert.Equal(ActivityKind.Client, chat.Kind);
                Assert.Equal(subInvoke.TraceId, chat.TraceId);
                Assert.Equal("chat", chat.GetTagItem("gen_ai.operation.name")?.ToString());
                Assert.NotNull(chat.GetTagItem("gen_ai.provider.name"));
            });

            // Subagent chat should have response model and response id
            var subChat = subChatActivities[0];
            if (subChat.GetTagItem("gen_ai.response.model") is not null)
            {
                Assert.IsType<string>(subChat.GetTagItem("gen_ai.response.model"));
            }
            if (subChat.GetTagItem("gen_ai.response.id") is not null)
            {
                Assert.IsType<string>(subChat.GetTagItem("gen_ai.response.id"));
            }

            // Subagent chat should have token counts
            if (subChat.GetTagItem("gen_ai.usage.input_tokens") is int subInputTokens)
            {
                Assert.True(subInputTokens > 0, "Subagent input tokens should be > 0");
            }
            if (subChat.GetTagItem("gen_ai.usage.output_tokens") is int subOutputTokens)
            {
                Assert.True(subOutputTokens > 0, "Subagent output tokens should be > 0");
            }

            // Subagent chat output messages should contain the poem
            var subChatOutput = subChat.GetTagItem("gen_ai.output.messages")?.ToString();
            Assert.NotNull(subChatOutput);
            Assert.Contains("\"role\":\"assistant\"", subChatOutput);
        }

        // =====================================================================
        // 5. execute_tool spans for get_topic
        // =====================================================================
        var getTopicTools = activities.Where(a =>
            a.GetTagItem("gen_ai.operation.name")?.ToString() == "execute_tool" &&
            a.GetTagItem("gen_ai.tool.name")?.ToString() == "get_topic").ToList();
        Assert.True(getTopicTools.Count >= 2,
            $"Expected ≥2 get_topic execute_tool spans, got {getTopicTools.Count}");

        Assert.All(getTopicTools, tool =>
        {
            Assert.Equal(ActivityKind.Internal, tool.Kind);
            Assert.Equal(mainInvoke.TraceId, tool.TraceId);
            Assert.Equal(mainInvoke.SpanId, tool.ParentSpanId);
            Assert.Equal("function", tool.GetTagItem("gen_ai.tool.type")?.ToString());
            Assert.NotNull(tool.GetTagItem("gen_ai.tool.call.id"));

            // Sensitive data enabled — tool result should be present
            var result = tool.GetTagItem("gen_ai.tool.call.result")?.ToString();
            Assert.NotNull(result);
            Assert.True(result!.Contains("mountains") || result.Contains("rivers"),
                $"Expected tool result containing 'mountains' or 'rivers', got '{result}'");
        });

        // =====================================================================
        // 6. Token isolation: main agent tokens are not inflated by subagent tokens
        // =====================================================================
        var allChatActivities = activities.Where(a =>
            a.GetTagItem("gen_ai.operation.name")?.ToString() == "chat").ToList();
        var mainTokenSum = mainChatActivities
            .Select(a => a.GetTagItem("gen_ai.usage.input_tokens")).OfType<int>().Sum();
        var subTokenSum = subagentInvokeActivities
            .SelectMany(sub => activities.Where(a =>
                a.GetTagItem("gen_ai.operation.name")?.ToString() == "chat" &&
                a.ParentSpanId == sub.SpanId))
            .Select(a => a.GetTagItem("gen_ai.usage.input_tokens")).OfType<int>().Sum();
        var totalTokenSum = allChatActivities
            .Select(a => a.GetTagItem("gen_ai.usage.input_tokens")).OfType<int>().Sum();

        if (mainTokenSum > 0 && subTokenSum > 0)
        {
            // All tokens should be accounted for without duplication
            Assert.Equal(mainTokenSum + subTokenSum, totalTokenSum);
        }

        // =====================================================================
        // 7. Duration metrics — with model tags
        // =====================================================================
        Assert.NotEmpty(durationRecordings);

        // All duration metrics should have provider.name
        Assert.All(durationRecordings, r =>
        {
            Assert.Contains(r.Tags, t => t.Key == "gen_ai.provider.name" && t.Value is not null);
        });

        // invoke_agent durations
        var invokeAgentDurations = durationRecordings.Where(r =>
            r.Tags.Any(t => t.Key == "gen_ai.operation.name" && t.Value?.ToString() == "invoke_agent")).ToList();
        Assert.True(invokeAgentDurations.Count >= 3,
            $"Expected ≥3 invoke_agent duration recordings (1 root + ≥2 subagents), got {invokeAgentDurations.Count}");
        Assert.All(invokeAgentDurations, r => Assert.True(r.Value > 0));

        // chat durations
        var chatDurations = durationRecordings.Where(r =>
            r.Tags.Any(t => t.Key == "gen_ai.operation.name" && t.Value?.ToString() == "chat")).ToList();
        Assert.NotEmpty(chatDurations);
        Assert.All(chatDurations, r => Assert.True(r.Value > 0));

        // execute_tool durations
        Assert.Contains(durationRecordings, r =>
            r.Tags.Any(t => t.Key == "gen_ai.operation.name" && t.Value?.ToString() == "execute_tool") &&
            r.Value > 0);

        // Duration metrics with response.model should carry the model used
        var durationsWithResponseModel = durationRecordings.Where(r =>
            r.Tags.Any(t => t.Key == "gen_ai.response.model" && t.Value is not null)).ToList();
        if (durationsWithResponseModel.Count > 0)
        {
            Assert.All(durationsWithResponseModel, r =>
            {
                var model = r.Tags.First(t => t.Key == "gen_ai.response.model").Value?.ToString();
                Assert.False(string.IsNullOrWhiteSpace(model), "gen_ai.response.model should not be blank");
            });
        }

        // =====================================================================
        // 8. Token usage metrics — with model tags
        // =====================================================================
        if (tokenRecordings.Count > 0)
        {
            Assert.All(tokenRecordings, r =>
            {
                // Token type must be input or output
                var tokenType = r.Tags.FirstOrDefault(t => t.Key == "gen_ai.token.type").Value?.ToString();
                Assert.True(tokenType is "input" or "output",
                    $"Expected token type 'input' or 'output', got '{tokenType}'");

                // Must be tagged with operation.name = chat
                Assert.Contains(r.Tags, t => t.Key == "gen_ai.operation.name" && t.Value?.ToString() == "chat");

                // Must carry provider.name
                Assert.Contains(r.Tags, t => t.Key == "gen_ai.provider.name" && t.Value is not null);

                // Must carry gen_ai.response.model (the model that produced the tokens)
                var responseModel = r.Tags.FirstOrDefault(t => t.Key == "gen_ai.response.model").Value?.ToString();
                Assert.False(string.IsNullOrWhiteSpace(responseModel),
                    "Token metric should carry gen_ai.response.model");
            });

            // Token recordings from subagent chat spans should exist and have their
            // own model attribution (not the main agent's model attribution)
            var subagentTokenRecordings = tokenRecordings.Where(r =>
                r.Tags.Any(t => t.Key == "gen_ai.response.model" && t.Value is not null)).ToList();
            Assert.NotEmpty(subagentTokenRecordings);
        }
    }

    // -- Helpers --

    [Description("Encrypts a string")]
    static string EncryptString([Description("String to encrypt")] string input)
        => input.ToUpperInvariant();

    private static ActivityListener CreateActivityListener(string sourceName, List<Activity> activities)
    {
        var listener = new ActivityListener
        {
            ShouldListenTo = source => source.Name == sourceName,
            Sample = (ref ActivityCreationOptions<ActivityContext> _) => ActivitySamplingResult.AllDataAndRecorded,
            ActivityStopped = activity => activities.Add(activity),
        };
        ActivitySource.AddActivityListener(listener);
        return listener;
    }

    private static MeterListener CreateMeterListener(
        string meterName,
        List<(int, KeyValuePair<string, object?>[])> tokenRecordings,
        List<(double, KeyValuePair<string, object?>[])> durationRecordings)
    {
        var listener = new MeterListener();
        listener.InstrumentPublished = (instrument, mListener) =>
        {
            if (instrument.Meter.Name == meterName)
            {
                mListener.EnableMeasurementEvents(instrument);
            }
        };
        listener.SetMeasurementEventCallback<int>((instrument, measurement, tags, _) =>
        {
            if (instrument.Name == "gen_ai.client.token.usage")
            {
                tokenRecordings.Add((measurement, tags.ToArray()));
            }
        });
        listener.SetMeasurementEventCallback<double>((instrument, measurement, tags, _) =>
        {
            if (instrument.Name == "gen_ai.client.operation.duration")
            {
                durationRecordings.Add((measurement, tags.ToArray()));
            }
        });
        listener.Start();
        return listener;
    }
}
