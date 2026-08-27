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

func TestChatOutboundTransformer_TransformRequest_flattensNestedRootUnionRefs(t *testing.T) {
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
				Name: "diag_tool",
				Parameters: json.RawMessage(`{
					"type": "object",
					"oneOf": [
						{"$ref": "#/$defs/view"},
						{"$ref": "#/$defs/create"}
					],
					"$defs": {
						"view": {
							"type": "object",
							"properties": {
								"id": {"type": "string"},
								"mode": {"enum": ["view"]}
							},
							"required": ["id", "mode"],
							"additionalProperties": false
						},
						"create": {
							"oneOf": [
								{
									"type": "object",
									"properties": {
										"kind": {"enum": ["cron"]},
										"mode": {"enum": ["create"]}
									},
									"required": ["kind", "mode"],
									"additionalProperties": false
								},
								{
									"type": "object",
									"properties": {
										"kind": {"enum": ["heartbeat"]},
										"mode": {"enum": ["create"]}
									},
									"required": ["kind", "mode"],
									"additionalProperties": false
								}
							]
						}
					}
				}`),
			},
		}},
	}

	originalParameters := string(request.Tools[0].Function.Parameters)

	httpRequest, err := outbound.TransformRequest(context.Background(), request)
	require.NoError(t, err)
	require.Equal(t, originalParameters, string(request.Tools[0].Function.Parameters))

	var body struct {
		Tools []struct {
			Function struct {
				Parameters map[string]any `json:"parameters"`
			} `json:"function"`
		} `json:"tools"`
	}
	require.NoError(t, json.Unmarshal(httpRequest.Body, &body))
	require.Len(t, body.Tools, 1)

	branches, ok := body.Tools[0].Function.Parameters["oneOf"].([]any)
	require.True(t, ok)
	require.Len(t, branches, 3)

	modes := make([]string, 0, 3)
	for _, branch := range branches {
		branchSchema, ok := branch.(map[string]any)
		require.True(t, ok)
		require.Equal(t, "object", branchSchema["type"])

		properties, ok := branchSchema["properties"].(map[string]any)
		require.True(t, ok)
		mode, ok := properties["mode"].(map[string]any)
		require.True(t, ok)
		values, ok := mode["enum"].([]any)
		require.True(t, ok)
		require.Len(t, values, 1)
		modes = append(modes, values[0].(string))
	}
	require.Equal(t, []string{"view", "create", "create"}, modes)
}

func TestChatOutboundTransformer_DoesNotFlattenSemanticNestedUnions(t *testing.T) {
	for name, tc := range map[string]struct {
		parameters json.RawMessage
		nestedKey  string
	}{
		"sibling": {
			parameters: json.RawMessage(`{"type":"object","oneOf":[{"$ref":"#/$defs/value"}],"$defs":{"value":{"oneOf":[{"type":"string"}],"description":"nested"}}}`),
			nestedKey:  "oneOf",
		},
		"mixed": {
			parameters: json.RawMessage(`{"type":"object","oneOf":[{"$ref":"#/$defs/value"}],"$defs":{"value":{"anyOf":[{"type":"string"},{"type":"number"}]}}}`),
			nestedKey:  "anyOf",
		},
	} {
		t.Run(name, func(t *testing.T) {
			normalized, err := normalizeChatFunctionParameters(tc.parameters)
			require.NoError(t, err)

			var schema map[string]any
			require.NoError(t, json.Unmarshal(normalized, &schema))
			branches, ok := schema["oneOf"].([]any)
			require.True(t, ok)
			require.Len(t, branches, 1)

			branch, ok := branches[0].(map[string]any)
			require.True(t, ok)
			require.Contains(t, branch, tc.nestedKey)
			if name == "sibling" {
				require.Equal(t, "nested", branch["description"])
			} else {
				require.Len(t, branch, 1)
			}
		})
	}
}

func TestChatOutboundTransformer_DoesNotFlattenOverlappingNestedOneOf(t *testing.T) {
	parameters := json.RawMessage(`{
		"type": "object",
		"oneOf": [
			{"$ref": "#/$defs/view"},
			{"$ref": "#/$defs/create"}
		],
		"$defs": {
			"view": {
				"type": "object",
				"properties": {"mode": {"enum": ["view"]}},
				"required": ["mode"],
				"additionalProperties": false
			},
			"create": {
				"oneOf": [
					{
						"type": "object",
						"properties": {
							"kind": {"enum": ["cron"]},
							"mode": {"enum": ["create"]}
						},
						"required": ["kind", "mode"],
						"additionalProperties": false
					},
					{
						"type": "object",
						"properties": {
							"kind": {"enum": ["cron"]},
							"mode": {"enum": ["create"]}
						},
						"required": ["kind", "mode"],
						"additionalProperties": false
					}
				]
			}
		}
	}`)

	normalized, err := normalizeChatFunctionParameters(parameters)
	require.NoError(t, err)

	var schema map[string]any
	require.NoError(t, json.Unmarshal(normalized, &schema))
	branches, ok := schema["oneOf"].([]any)
	require.True(t, ok)
	require.Len(t, branches, 2)

	nested, ok := branches[1].(map[string]any)
	require.True(t, ok)
	require.Contains(t, nested, "oneOf")
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
