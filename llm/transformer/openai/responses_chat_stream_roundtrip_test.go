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

func TestResponsesToChatTools_NamespaceCustomToolStreamRoundTrip(t *testing.T) {
	ctx := context.Background()
	responsesInbound := responsesapi.NewInboundTransformer()
	llmRequest, err := responsesInbound.TransformRequest(ctx, &httpclient.Request{Body: []byte(`{
		"model":"gpt-5.5","stream":true,
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
	require.Len(t, converted.Tools, 1)
	require.Equal(t, "functions__exec", converted.Tools[0].Function.Name)

	chatChunk := func(choice map[string]any) *httpclient.StreamEvent {
		return &httpclient.StreamEvent{Data: marshalResponsesChatTestJSON(t, map[string]any{
			"id": "chatcmpl_stream", "object": "chat.completion.chunk", "created": 1, "model": "gpt-5.5",
			"choices": []any{choice},
		})}
	}
	providerEvents := []*httpclient.StreamEvent{
		chatChunk(map[string]any{
			"index": 0, "delta": map[string]any{"role": "assistant"},
		}),
		chatChunk(map[string]any{
			"index": 0, "delta": map[string]any{"tool_calls": []any{map[string]any{
				"index": 0, "id": "call_exec", "type": "function",
				"function": map[string]any{"name": "functions__exec", "arguments": `{"input":"`},
			}}},
		}),
		chatChunk(map[string]any{
			"index": 0, "delta": map[string]any{"tool_calls": []any{map[string]any{
				"index": 0, "function": map[string]any{"arguments": `text(\"hello\")"}`},
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

	var added, done *responsesapi.Item
	for i := range events {
		event := &events[i]
		switch event.Type {
		case responsesapi.StreamEventTypeOutputItemAdded:
			if event.Item != nil && event.Item.Type == "custom_tool_call" {
				added = event.Item
			}
		case responsesapi.StreamEventTypeOutputItemDone:
			if event.Item != nil && event.Item.Type == "custom_tool_call" {
				done = event.Item
			}
		}
	}

	require.NotNil(t, added)
	require.Equal(t, "call_exec", added.CallID)
	require.Equal(t, "exec", added.Name)
	require.Equal(t, "functions", added.Namespace)

	require.NotNil(t, done)
	require.Equal(t, "call_exec", done.CallID)
	require.Equal(t, "exec", done.Name)
	require.Equal(t, "functions", done.Namespace)
	require.Equal(t, `text("hello")`, lo.FromPtr(done.Input))
}
