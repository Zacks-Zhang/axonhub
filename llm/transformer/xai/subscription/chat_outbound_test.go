package subscription

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/samber/lo"
	"github.com/stretchr/testify/require"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/oauth"
	"github.com/looplj/axonhub/llm/transformer"
)

func TestChatOutboundTransformer_TransformRequest_uses_xAI_CLI_chat_identity(t *testing.T) {
	const accessToken = "synthetic-access-token"
	tokens := oauth.NewStaticTokenProvider(&oauth.OAuthCredentials{
		AccessToken: accessToken,
		ExpiresAt:   time.Now().Add(time.Hour),
	})
	outbound, err := NewChatOutboundTransformer(tokens)
	require.NoError(t, err)
	request := &llm.Request{
		Model: "grok-4.5",
		Messages: []llm.Message{
			{Role: "user", Content: llm.MessageContent{Content: lo.ToPtr("hello")}},
		},
		Stream: lo.ToPtr(true),
	}

	httpRequest, err := outbound.TransformRequest(context.Background(), request)
	require.NoError(t, err)
	require.Equal(t, DefaultBaseURL+"/chat/completions", httpRequest.URL)
	require.Equal(t, accessToken, httpRequest.Auth.APIKey)
	require.Equal(t, CLITokenAuth, httpRequest.Headers.Get(CLITokenAuthHeader))
	require.Equal(t, CLIClientVersion, httpRequest.Headers.Get(CLIClientVersionHeader))
	require.Equal(t, CLIClientIdentifier, httpRequest.Headers.Get(CLIClientIdentifierHeader))
	require.Equal(t, CLIUserAgent, httpRequest.Headers.Get("User-Agent"))
}

func TestChatOutboundTransformer_APIFormat_returns_chat_completions(t *testing.T) {
	outbound, err := NewChatOutboundTransformer(oauth.NewStaticTokenProvider(&oauth.OAuthCredentials{AccessToken: "synthetic-token"}))
	require.NoError(t, err)
	require.Equal(t, llm.APIFormatOpenAIChatCompletion, outbound.APIFormat())
}

func TestChatOutboundTransformer_ResponsesRequestCapabilities_forwards_chat_lifecycle(t *testing.T) {
	outbound, err := NewChatOutboundTransformer(oauth.NewStaticTokenProvider(&oauth.OAuthCredentials{AccessToken: "synthetic-token"}))
	require.NoError(t, err)
	require.True(t, outbound.ResponsesRequestCapabilities(&llm.Request{APIFormat: llm.APIFormatOpenAIResponse}).ChatToolLifecycle)
	require.False(t, outbound.ResponsesRequestCapabilities(&llm.Request{RequestType: llm.RequestTypeCompact}).ChatToolLifecycle)
}

func TestChatOutboundTransformer_TransformRequest_rejects_non_chat_requests(t *testing.T) {
	for _, requestType := range []llm.RequestType{llm.RequestTypeImage, llm.RequestTypeCompact} {
		t.Run(string(requestType), func(t *testing.T) {
			outbound, err := NewChatOutboundTransformer(oauth.NewStaticTokenProvider(&oauth.OAuthCredentials{AccessToken: "synthetic-token"}))
			require.NoError(t, err)

			request, err := outbound.TransformRequest(t.Context(), &llm.Request{RequestType: requestType, Model: "synthetic-model"})
			require.Nil(t, request)
			require.ErrorIs(t, err, transformer.ErrInvalidRequest)
		})
	}
}

func TestChatOutboundTransformer_TransformRequest_inlinesRootUnionRefs(t *testing.T) {
	outbound, err := NewChatOutboundTransformer(oauth.NewStaticTokenProvider(&oauth.OAuthCredentials{AccessToken: "synthetic-token"}))
	require.NoError(t, err)

	request := &llm.Request{
		APIFormat: llm.APIFormatOpenAIResponse,
		Model:     "grok-4.6",
		Messages: []llm.Message{
			{Role: "user", Content: llm.MessageContent{Content: lo.ToPtr("hello")}},
		},
		Tools: []llm.Tool{{
			Type: llm.ToolTypeFunction,
			Function: llm.Function{
				Name:       "diag_tool",
				Parameters: json.RawMessage("{\"type\":\"object\",\"oneOf\":[{\"$ref\":\"#/$defs/create\"},{\"$ref\":\"#/$defs/view\"}],\"$defs\":{\"create\":{\"type\":\"object\",\"properties\":{\"name\":{\"$ref\":\"#/$defs/value\"}},\"required\":[\"name\"],\"additionalProperties\":false},\"view\":{\"type\":\"object\",\"properties\":{\"id\":{\"$ref\":\"#/$defs/value\"}},\"required\":[\"id\"],\"additionalProperties\":false},\"value\":{\"type\":\"string\"}}}"),
			},
		}},
	}

	originalParameters := string(request.Tools[0].Function.Parameters)

	httpRequest, err := outbound.TransformRequest(context.Background(), request)
	require.NoError(t, err)
	require.Equal(t, originalParameters, string(request.Tools[0].Function.Parameters))

	var body map[string]any
	require.NoError(t, json.Unmarshal(httpRequest.Body, &body))
	tools, ok := body["tools"].([]any)
	require.True(t, ok)
	require.Len(t, tools, 1)
	tool, ok := tools[0].(map[string]any)
	require.True(t, ok)
	function, ok := tool["function"].(map[string]any)
	require.True(t, ok)

	expected := map[string]any{
		"type": "object",
		"oneOf": []any{
			map[string]any{
				"type":                 "object",
				"properties":           map[string]any{"name": map[string]any{"type": "string"}},
				"required":             []any{"name"},
				"additionalProperties": false,
			},
			map[string]any{
				"type":                 "object",
				"properties":           map[string]any{"id": map[string]any{"type": "string"}},
				"required":             []any{"id"},
				"additionalProperties": false,
			},
		},
		"$defs": map[string]any{
			"create": map[string]any{
				"type":                 "object",
				"properties":           map[string]any{"name": map[string]any{"$ref": "#/$defs/value"}},
				"required":             []any{"name"},
				"additionalProperties": false,
			},
			"view": map[string]any{
				"type":                 "object",
				"properties":           map[string]any{"id": map[string]any{"$ref": "#/$defs/value"}},
				"required":             []any{"id"},
				"additionalProperties": false,
			},
			"value": map[string]any{"type": "string"},
		},
	}
	require.Equal(t, expected, function["parameters"])
}

func TestChatOutboundTransformer_TransformRequest_rejectsMissingUnionRef(t *testing.T) {
	outbound, err := NewChatOutboundTransformer(oauth.NewStaticTokenProvider(&oauth.OAuthCredentials{AccessToken: "synthetic-token"}))
	require.NoError(t, err)

	request := &llm.Request{
		APIFormat: llm.APIFormatOpenAIResponse,
		Model:     "grok-4.6",
		Messages: []llm.Message{
			{Role: "user", Content: llm.MessageContent{Content: lo.ToPtr("hello")}},
		},
		Tools: []llm.Tool{{
			Type: llm.ToolTypeFunction,
			Function: llm.Function{
				Name:       "diag_tool",
				Parameters: json.RawMessage("{\"type\":\"object\",\"oneOf\":[{\"$ref\":\"#/$defs/missing\"}]}"),
			},
		}},
	}

	httpRequest, err := outbound.TransformRequest(context.Background(), request)
	require.Nil(t, httpRequest)
	require.ErrorIs(t, err, transformer.ErrInvalidRequest)
	require.ErrorContains(t, err, "local definition \"missing\" is missing")
}

func TestChatOutboundTransformer_DoesNotNormalizeNonUnionRefs(t *testing.T) {
	parameters := json.RawMessage("{\"type\":\"object\",\"properties\":{\"name\":{\"$ref\":\"#/$defs/name\"}},\"$defs\":{\"name\":{\"type\":\"string\"}}}")

	normalized, err := normalizeChatFunctionParameters(parameters)
	require.NoError(t, err)
	require.Equal(t, string(parameters), string(normalized))
}

func TestChatOutboundTransformer_DoesNotNormalizeNativeChatRequests(t *testing.T) {
	request := &llm.Request{
		APIFormat: llm.APIFormatOpenAIChatCompletion,
		Tools: []llm.Tool{{
			Type: llm.ToolTypeFunction,
			Function: llm.Function{
				Name:       "diag_tool",
				Parameters: json.RawMessage(`{"type":"object","oneOf":[{"$ref":"#/$defs/value"}],"$defs":{"value":{"type":"string"}}}`),
			},
		}},
	}

	normalized, err := normalizeChatRequestToolSchemas(request)
	require.NoError(t, err)
	require.Same(t, request, normalized)
}

func TestChatOutboundTransformer_RejectsCyclicUnionRefs(t *testing.T) {
	parameters := json.RawMessage(`{"type":"object","oneOf":[{"$ref":"#/$defs/a"}],"$defs":{"a":{"$ref":"#/$defs/b"},"b":{"$ref":"#/$defs/a"}}}`)

	normalized, err := normalizeChatFunctionParameters(parameters)
	require.Nil(t, normalized)
	require.ErrorContains(t, err, "local definition \"a\" is cyclic")
}
