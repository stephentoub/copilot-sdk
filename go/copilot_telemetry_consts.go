/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------------------------------------------*/

package copilot

// String constants for OpenTelemetry Semantic Conventions for Generative AI systems.
// Based on the Semantic Conventions for Generative AI systems v1.40,
// defined at https://opentelemetry.io/docs/specs/semconv/gen-ai/
// The specification is still experimental and subject to change.

const (
	otelDefaultSourceName           = "github.copilot.sdk"
	otelDefaultProviderName         = "github"
	otelCaptureMessageContentEnvVar = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"
	otelSecondsUnit                 = "s"
	otelTokensUnit                  = "{token}"

	otelTokenTypeInput  = "input"
	otelTokenTypeOutput = "output"

	// Error attributes
	otelErrorType = "error.type"

	// Server attributes
	otelServerAddress = "server.address"
	otelServerPort    = "server.port"

	// GenAI operation names
	otelInvokeAgent = "invoke_agent"
	otelChat        = "chat"
	otelExecuteTool = "execute_tool"

	// GenAI attributes
	otelGenAIOperationName = "gen_ai.operation.name"
	otelGenAIProviderName  = "gen_ai.provider.name"

	otelGenAIAgentID = "gen_ai.agent.id"

	otelGenAIAgentName        = "gen_ai.agent.name"
	otelGenAIAgentDescription = "gen_ai.agent.description"

	otelGenAIConversationID = "gen_ai.conversation.id"

	otelGenAIRequestModel = "gen_ai.request.model"

	otelGenAIResponseID            = "gen_ai.response.id"
	otelGenAIResponseModel         = "gen_ai.response.model"
	otelGenAIResponseFinishReasons = "gen_ai.response.finish_reasons"

	otelGenAIUsageInputTokens              = "gen_ai.usage.input_tokens"
	otelGenAIUsageOutputTokens             = "gen_ai.usage.output_tokens"
	otelGenAIUsageCacheReadInputTokens     = "gen_ai.usage.cache_read.input_tokens"
	otelGenAIUsageCacheCreationInputTokens = "gen_ai.usage.cache_creation.input_tokens"

	otelGenAITokenType = "gen_ai.token.type"

	otelGenAIInputMessages      = "gen_ai.input.messages"
	otelGenAIOutputMessages     = "gen_ai.output.messages"
	otelGenAISystemInstructions = "gen_ai.system_instructions"

	otelGenAIToolDefinitions   = "gen_ai.tool.definitions"
	otelGenAIToolCallID        = "gen_ai.tool.call.id"
	otelGenAIToolCallArguments = "gen_ai.tool.call.arguments"
	otelGenAIToolCallResult    = "gen_ai.tool.call.result"
	otelGenAIToolName          = "gen_ai.tool.name"
	otelGenAIToolDescription   = "gen_ai.tool.description"
	otelGenAIToolType          = "gen_ai.tool.type"

	// Metric names
	otelMetricOperationDuration  = "gen_ai.client.operation.duration"
	otelMetricTokenUsage         = "gen_ai.client.token.usage"
	otelMetricTimeToFirstChunk   = "gen_ai.client.operation.time_to_first_chunk"
	otelMetricTimePerOutputChunk = "gen_ai.client.operation.time_per_output_chunk"

	// Metric descriptions
	otelDescOperationDuration  = "GenAI operation duration."
	otelDescTokenUsage         = "Number of input and output tokens used."
	otelDescTimeToFirstChunk   = "Time to receive the first chunk from a streaming response."
	otelDescTimePerOutputChunk = "Time elapsed between streamed output chunks after the first chunk."

	// Vendor-prefixed span attributes
	otelCopilotCost           = "github.copilot.cost"
	otelCopilotServerDuration = "github.copilot.server_duration"
	otelCopilotInitiator      = "github.copilot.initiator"
	otelCopilotAIU            = "github.copilot.aiu"
	otelCopilotTurnID         = "github.copilot.turn_id"
	otelCopilotInteractionID  = "github.copilot.interaction_id"

	// Vendor-prefixed span event names
	otelCopilotSessionTruncation         = "github.copilot.session.truncation"
	otelCopilotSessionCompactionStart    = "github.copilot.session.compaction_start"
	otelCopilotSessionCompactionComplete = "github.copilot.session.compaction_complete"
	otelCopilotSkillInvoked              = "github.copilot.skill.invoked"

	// Event attribute keys
	otelCopilotMessage            = "github.copilot.message"
	otelCopilotTokenLimit         = "github.copilot.token_limit"
	otelCopilotPreTokens          = "github.copilot.pre_tokens"
	otelCopilotPostTokens         = "github.copilot.post_tokens"
	otelCopilotPreMessages        = "github.copilot.pre_messages"
	otelCopilotPostMessages       = "github.copilot.post_messages"
	otelCopilotTokensRemoved      = "github.copilot.tokens_removed"
	otelCopilotMessagesRemoved    = "github.copilot.messages_removed"
	otelCopilotPerformedBy        = "github.copilot.performed_by"
	otelCopilotSuccess            = "github.copilot.success"
	otelCopilotSkillName          = "github.copilot.skill.name"
	otelCopilotSkillPath          = "github.copilot.skill.path"
	otelCopilotSkillContent       = "github.copilot.skill.content"
	otelCopilotSkillPluginName    = "github.copilot.skill.plugin_name"
	otelCopilotSkillPluginVersion = "github.copilot.skill.plugin_version"
)

// Explicit bucket boundaries for histograms, matching .NET implementation.
var (
	otelDurationBucketBoundaries = []float64{
		0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24, 20.48, 40.96, 81.92,
	}
	otelTokenBucketBoundaries = []float64{
		1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 67108864,
	}
)
