package openai

import (
	"context"
	"encoding/json"
	"strconv"
	"testing"

	"github.com/samber/lo"
	"github.com/stretchr/testify/require"

	"github.com/looplj/axonhub/llm/httpclient"
	"github.com/looplj/axonhub/llm/streams"
	responsesapi "github.com/looplj/axonhub/llm/transformer/openai/responses"
)

// Freeform/grammar custom tools describe themselves as raw input, which
// conflicts with the {"input":"..."} wrapper the Chat adapter adds. Models then
// emit the bare payload and repeat it inside the wrapper, producing
// raw+wrapped concatenation that must still deliver the raw payload once.
func TestResponsesToChatTools_FreeformConcatenatedInputUnwraps(t *testing.T) {
	const rawJS = "const r = await tools.exec_command({cmd:\"ls\"}); text(r.output);"

	wrapped, err := json.Marshal(map[string]string{"input": rawJS})
	require.NoError(t, err)
	concatenated := rawJS + string(wrapped)

	for _, arguments := range []string{string(wrapped), concatenated, rawJS} {
		t.Run("arguments_len_"+strconv.Itoa(len(arguments)), func(t *testing.T) {
			ctx := context.Background()
			responsesInbound := responsesapi.NewInboundTransformer()
			llmRequest, err := responsesInbound.TransformRequest(ctx, &httpclient.Request{Body: []byte(`{
				"model":"gpt-5.5","stream":true,
				"input":[
					{"type":"additional_tools","role":"developer","tools":[
						{"type":"namespace","name":"functions","tools":[
							{"type":"custom","name":"exec","description":"Run JavaScript code",
							 "format":{"type":"grammar","syntax":"lark","definition":"start: SOURCE"}}
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
						"function": map[string]any{"name": "functions__exec", "arguments": arguments},
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
			var deltas []string
			for i := range events {
				event := &events[i]
				switch event.Type {
				case responsesapi.StreamEventTypeCustomToolCallInputDelta:
					deltas = append(deltas, event.Delta)
				case responsesapi.StreamEventTypeOutputItemDone:
					if event.Item != nil && event.Item.Type == "custom_tool_call" {
						done = event.Item
					}
				}
			}

			require.NotNil(t, done)
			require.Equal(t, "custom_tool_call", done.Type)
			require.Equal(t, "exec", done.Name)
			require.Equal(t, "functions", done.Namespace)
			require.Equal(t, rawJS, lo.FromPtr(done.Input))
			require.NotContains(t, lo.FromPtr(done.Input), `{"input"`)
			for _, delta := range deltas {
				require.NotContains(t, delta, `{"input"`)
			}
		})
	}
}

func TestUnwrapChatCustomInput(t *testing.T) {
	const raw = "text(\"hello\");"
	wrapped := `{"input":"text(\"hello\");"}`

	got, warn := responsesapi.UnwrapChatCustomInput(wrapped)
	require.Equal(t, raw, got)
	require.Empty(t, warn)

	got, warn = responsesapi.UnwrapChatCustomInput(raw)
	require.Equal(t, raw, got)
	require.Empty(t, warn)

	got, warn = responsesapi.UnwrapChatCustomInput(raw + wrapped)
	require.Equal(t, raw, got)
	require.Empty(t, warn)

	got, warn = responsesapi.UnwrapChatCustomInput("plain")
	require.Equal(t, "plain", got)
	require.Empty(t, warn)

	_, warn = responsesapi.UnwrapChatCustomInput(`{"input":"patch"`)
	require.Contains(t, warn, "failed to unwrap Chat custom tool input")

	_, warn = responsesapi.UnwrapChatCustomInput(`{}`)
	require.Contains(t, warn, "failed to unwrap Chat custom tool input")
	require.Contains(t, warn, "missing input field")
}
