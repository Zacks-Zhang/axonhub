package openai

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/samber/lo"
	"github.com/stretchr/testify/require"

	"github.com/looplj/axonhub/llm/httpclient"
	"github.com/looplj/axonhub/llm/streams"
	responsesapi "github.com/looplj/axonhub/llm/transformer/openai/responses"
)

// A selector may address a namespaced custom tool by its bare member name.
// Matching only the flattened form marks the mapping HistoryOnly and drops it
// from tools[], so the model still calls it from the system prompt but the
// client receives an unrestorable function_call and aborts it.
func TestResponsesToChatTools_NamespaceCustomToolAllowedToolsRoundTrip(t *testing.T) {
	for _, selectorName := range []string{"exec", "functions__exec"} {
		t.Run("selector_"+selectorName, func(t *testing.T) {
			ctx := context.Background()
			responsesInbound := responsesapi.NewInboundTransformer()
			llmRequest, err := responsesInbound.TransformRequest(ctx, &httpclient.Request{Body: []byte(`{
				"model":"gpt-5.5","stream":true,
				"tool_choice":{"type":"allowed_tools","mode":"auto","tools":[
					{"type":"custom","name":"` + selectorName + `"}
				]},
				"input":[
					{"type":"additional_tools","role":"developer","tools":[
						{"type":"namespace","name":"functions","tools":[
							{"type":"custom","name":"exec","description":"Run code"}
						]}
					]},
					{"role":"user","type":"message","content":[{"type":"input_text","text":"continue"}]}
				]
			}`)})
			require.NoError(t, err)

			chatOutbound, err := NewOutboundTransformer("https://chat.example.com", "test-key")
			require.NoError(t, err)
			chatRequest, err := chatOutbound.TransformRequest(ctx, llmRequest)
			require.NoError(t, err)

			var converted Request
			require.NoError(t, json.Unmarshal(chatRequest.Body, &converted))
			require.Equal(t, []string{"functions__exec"}, chatToolNames(converted.Tools))

			mappings := responsesChatToolMappings(chatRequest)
			require.False(t, mappings["functions__exec"].HistoryOnly)

			chatChunk := func(choice map[string]any) *httpclient.StreamEvent {
				return &httpclient.StreamEvent{Data: marshalResponsesChatTestJSON(t, map[string]any{
					"id": "chatcmpl_stream", "object": "chat.completion.chunk", "created": 1, "model": "gpt-5.5",
					"choices": []any{choice},
				})}
			}
			providerEvents := []*httpclient.StreamEvent{
				chatChunk(map[string]any{
					"index": 0, "delta": map[string]any{"tool_calls": []any{map[string]any{
						"index": 0, "id": "call_exec", "type": "function",
						"function": map[string]any{"name": "functions__exec", "arguments": `{"input":"text(\"hello\")"}`},
					}}},
				}),
				chatChunk(map[string]any{"index": 0, "delta": map[string]any{}, "finish_reason": "tool_calls"}),
				{Data: []byte("[DONE]")},
			}
			llmStream, err := chatOutbound.TransformStream(ctx, chatRequest, streams.SliceStream(providerEvents))
			require.NoError(t, err)
			responsesStream, err := responsesInbound.TransformStream(ctx, llmStream)
			require.NoError(t, err)

			var events []responsesapi.StreamEvent
			for responsesStream.Next() {
				var event responsesapi.StreamEvent
				require.NoError(t, json.Unmarshal(responsesStream.Current().Data, &event))
				events = append(events, event)
			}
			require.NoError(t, responsesStream.Err())

			var done *responsesapi.Item
			for i := range events {
				event := &events[i]
				if event.Type == responsesapi.StreamEventTypeOutputItemDone && event.Item != nil {
					done = event.Item
				}
			}

			require.NotNil(t, done)
			require.Equal(t, "custom_tool_call", done.Type)
			require.Equal(t, "exec", done.Name)
			require.Equal(t, "functions", done.Namespace)
			require.Equal(t, `text("hello")`, lo.FromPtr(done.Input))
		})
	}
}

func chatToolNames(tools []Tool) []string {
	names := make([]string, 0, len(tools))
	for _, tool := range tools {
		names = append(names, tool.Function.Name)
	}
	return names
}
