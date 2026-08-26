package subscription

import (
	"context"
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
