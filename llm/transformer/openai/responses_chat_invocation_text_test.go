package openai

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/samber/lo"
	"github.com/stretchr/testify/require"

	"github.com/looplj/axonhub/llm/httpclient"
	"github.com/looplj/axonhub/llm/streams"
	responsesapi "github.com/looplj/axonhub/llm/transformer/openai/responses"
)

func TestResponsesToChatTools_SuppressesDuplicateInvocationText(t *testing.T) {
	const rawJS = "const r = await tools.exec_command({cmd:\"ls\"}); text(r.output);"
	openTag := "<" + "tool-use>"
	closeTag := "</parameter" + "></" + "function>"
	invocation := openTag + "\n" + rawJS + "\n" + closeTag

	ctx := context.Background()
	responsesInbound := responsesapi.NewInboundTransformer()
	llmRequest, err := responsesInbound.TransformRequest(ctx, &httpclient.Request{Body: []byte(`{
		"model":"gpt-5.5","stream":true,
		"input":[
			{"type":"additional_tools","role":"developer","tools":[
				{"type":"namespace","name":"functions","tools":[
					{"type":"custom","name":"exec","description":"Run JavaScript code"}
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
				"function": map[string]any{"name": "functions__exec", "arguments": ""},
			}}},
		}),
		chatChunk(map[string]any{
			"index": 0, "delta": map[string]any{"content": invocation},
		}),
		chatChunk(map[string]any{
			"index": 0, "delta": map[string]any{"tool_calls": []any{map[string]any{
				"index": 0, "function": map[string]any{"arguments": string(lo.Must(json.Marshal(map[string]string{"input": rawJS})))},
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

	var (
		done   *responsesapi.Item
		textD  []string
		anyMsg bool
	)
	for i := range events {
		event := &events[i]
		switch event.Type {
		case responsesapi.StreamEventTypeOutputTextDelta:
			textD = append(textD, event.Delta)
		case responsesapi.StreamEventTypeOutputItemAdded:
			if event.Item != nil && event.Item.Type == "message" {
				anyMsg = true
			}
		case responsesapi.StreamEventTypeOutputItemDone:
			if event.Item != nil && event.Item.Type == "message" {
				anyMsg = true
			}
			if event.Item != nil && event.Item.Type == "custom_tool_call" {
				done = event.Item
			}
		}
	}

	require.NotNil(t, done, "structured custom tool call must survive")
	require.Equal(t, rawJS, lo.FromPtr(done.Input))
	require.False(t, anyMsg, "duplicate invocation text must not emit a message item")
	require.Empty(t, textD, "duplicate invocation text must not emit text deltas")
	require.NotContains(t, strings.Join(textD, ""), "tool")
}

func TestIsToolInvocationText(t *testing.T) {
	openTag := "<" + "tool-use>"
	require.True(t, responsesapi.IsToolInvocationText(openTag+"abc"))
	require.True(t, responsesapi.IsToolInvocationText("  "+openTag+"abc"))
	require.False(t, responsesapi.IsToolInvocationText("I will run the command."))
	require.False(t, responsesapi.IsToolInvocationText("see tool docs"))
}
