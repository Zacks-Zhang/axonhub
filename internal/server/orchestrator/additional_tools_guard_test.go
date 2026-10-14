package orchestrator

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/samber/lo"
	"github.com/stretchr/testify/require"

	"github.com/looplj/axonhub/internal/ent"
	"github.com/looplj/axonhub/internal/server/biz"
	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/transformer"
	openaitransformer "github.com/looplj/axonhub/llm/transformer/openai"
)

func TestPersistentOutboundTransformer_TransformRequest_RejectsAdditionalToolsForFinalNonResponsesFormat(t *testing.T) {
	tests := []struct {
		name            string
		candidateFormat llm.APIFormat
		returnedFormat  llm.APIFormat
		chatToolAdapter bool
		wantUnsupported bool
	}{
		{name: "chat with tool adapter", candidateFormat: llm.APIFormatOpenAIChatCompletion, returnedFormat: "", chatToolAdapter: true, wantUnsupported: false},
		{name: "chat without tool adapter", candidateFormat: llm.APIFormatOpenAIChatCompletion, returnedFormat: "", chatToolAdapter: false, wantUnsupported: true},
		{name: "anthropic", candidateFormat: llm.APIFormatAnthropicMessage, returnedFormat: "", wantUnsupported: true},
		{name: "gemini", candidateFormat: llm.APIFormatGeminiContents, returnedFormat: "", wantUnsupported: true},
		{name: "returned format overrides responses candidate", candidateFormat: llm.APIFormatOpenAIResponse, returnedFormat: llm.APIFormatOpenAIChatCompletion, wantUnsupported: true},
		{name: "responses", candidateFormat: llm.APIFormatOpenAIResponse, returnedFormat: "", wantUnsupported: false},
		{name: "compact", candidateFormat: llm.APIFormatOpenAIResponseCompact, returnedFormat: "", wantUnsupported: false},
		{name: "returned responses overrides chat candidate", candidateFormat: llm.APIFormatOpenAIChatCompletion, returnedFormat: llm.APIFormatOpenAIResponse, wantUnsupported: false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outbound := &mockTransformer{apiFormat: tt.candidateFormat, requestAPIFormat: tt.returnedFormat, responsesChatTools: tt.chatToolAdapter}
			channel := &biz.Channel{
				Channel:  &ent.Channel{Name: tt.name},
				Outbound: outbound,
			}
			processor := &PersistentOutboundTransformer{
				wrapped: outbound,
				state: &PersistenceState{
					ChannelModelsCandidates: []*ChannelModelsCandidate{{
						Channel:   channel,
						Models:    []biz.ChannelModelEntry{{RequestModel: "model", ActualModel: "model"}},
						APIFormat: tt.candidateFormat.String(),
					}},
				},
			}
			request := &llm.Request{
				ProviderExtensions: &llm.ProviderExtensions{
					OpenAIResponses: &llm.OpenAIResponsesProviderExtensions{
						Request: &llm.OpenAIResponsesRequestExtensions{
							RawInputItems: []llm.OpenAIResponsesRawFragment{{
								Type: "additional_tools",
								Raw:  json.RawMessage(`{"type":"additional_tools"}`),
							}},
						},
					},
				},
			}

			_, err := processor.TransformRequest(context.Background(), request)
			if tt.wantUnsupported {
				require.ErrorIs(t, err, transformer.ErrUnsupportedConversion)
				require.ErrorIs(t, err, transformer.ErrInvalidRequest)
				return
			}

			require.NoError(t, err)
		})
	}
}

func TestContainsAdditionalToolsInputItems_RequiresExactRawFragmentType(t *testing.T) {
	require.False(t, containsAdditionalToolsInputItems(&llm.Request{}))
	messageRequest := &llm.Request{ProviderExtensions: &llm.ProviderExtensions{OpenAIResponses: &llm.OpenAIResponsesProviderExtensions{
		Request: &llm.OpenAIResponsesRequestExtensions{RawInputItems: []llm.OpenAIResponsesRawFragment{{Type: "message"}}},
	}}}
	require.False(t, containsAdditionalToolsInputItems(messageRequest))
	additionalToolsRequest := &llm.Request{ProviderExtensions: &llm.ProviderExtensions{OpenAIResponses: &llm.OpenAIResponsesProviderExtensions{
		Request: &llm.OpenAIResponsesRequestExtensions{RawInputItems: []llm.OpenAIResponsesRawFragment{{Type: "additional_tools"}}},
	}}}
	require.True(t, containsAdditionalToolsInputItems(additionalToolsRequest))
}

// The chat tool adapter promotes additional_tools into Request.Tools, so a
// real Chat Completions outbound must not be rejected by the guard.
func TestPersistentOutboundTransformer_TransformRequest_AllowsAdditionalToolsForChatToolAdapter(t *testing.T) {
	outbound, err := openaitransformer.NewOutboundTransformer("https://chat.example.com", "test-key")
	require.NoError(t, err)

	processor := &PersistentOutboundTransformer{
		wrapped: outbound,
		state: &PersistenceState{
			ChannelModelsCandidates: []*ChannelModelsCandidate{{
				Channel:   &biz.Channel{Channel: &ent.Channel{Name: "chat"}, Outbound: outbound},
				Models:    []biz.ChannelModelEntry{{RequestModel: "model", ActualModel: "model"}},
				APIFormat: llm.APIFormatOpenAIChatCompletion.String(),
			}},
		},
	}

	request := &llm.Request{
		APIFormat: llm.APIFormatOpenAIResponse,
		Model:     "model",
		Messages:  []llm.Message{{Role: "user", Content: llm.MessageContent{Content: lo.ToPtr("hi")}}},
		Tools: []llm.Tool{{
			Type:             llm.ToolTypeFunction,
			Function:         llm.Function{Name: "lookup", Parameters: []byte(`{"type":"object"}`)},
			ResponsesOrigin:  "additional_tools",
			ResponsesRawID:   "input:0",
		}},
		ProviderExtensions: &llm.ProviderExtensions{
			OpenAIResponses: &llm.OpenAIResponsesProviderExtensions{
				Request: &llm.OpenAIResponsesRequestExtensions{
					RawInputItems: []llm.OpenAIResponsesRawFragment{{
						Type: "additional_tools",
						Raw:  json.RawMessage(`{"type":"additional_tools","tools":[{"type":"function","name":"lookup"}]}`),
					}},
				},
			},
		},
	}

	httpRequest, err := processor.TransformRequest(context.Background(), request)
	require.NoError(t, err)
	require.NotEmpty(t, httpRequest.Body)
	require.NotContains(t, string(httpRequest.Body), "additional_tools")
	require.Contains(t, string(httpRequest.Body), `"lookup"`)
}
