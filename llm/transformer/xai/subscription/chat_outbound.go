package subscription

import (
	"context"
	"errors"
	"fmt"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/auth"
	"github.com/looplj/axonhub/llm/httpclient"
	"github.com/looplj/axonhub/llm/oauth"
	"github.com/looplj/axonhub/llm/streams"
	"github.com/looplj/axonhub/llm/transformer"
	"github.com/looplj/axonhub/llm/transformer/xai"
)

type ChatOutboundTransformer struct {
	tokens oauth.TokenGetter
	chat   transformer.Outbound
}

var (
	_ transformer.Outbound                             = (*ChatOutboundTransformer)(nil)
	_ transformer.ResponsesRequestCapabilitiesProvider = (*ChatOutboundTransformer)(nil)
)

func NewChatOutboundTransformer(tokenProvider oauth.TokenGetter) (*ChatOutboundTransformer, error) {
	if tokenProvider == nil {
		return nil, errors.New("token provider is required")
	}

	outbound, err := xai.NewOutboundTransformerWithConfig(&xai.Config{
		BaseURL:        DefaultBaseURL,
		APIKeyProvider: auth.NewStaticKeyProvider("unused"),
	})
	if err != nil {
		return nil, err
	}

	return &ChatOutboundTransformer{tokens: tokenProvider, chat: outbound}, nil
}

func (t *ChatOutboundTransformer) TokenProvider() oauth.TokenGetter {
	return t.tokens
}

func (t *ChatOutboundTransformer) APIFormat() llm.APIFormat {
	return llm.APIFormatOpenAIChatCompletion
}

func (t *ChatOutboundTransformer) ResponsesRequestCapabilities(req *llm.Request) transformer.ResponsesRequestCapabilities {
	return transformer.ResponsesRequestCapabilitiesOf(t.chat, req)
}

func (t *ChatOutboundTransformer) TransformRequest(ctx context.Context, request *llm.Request) (*httpclient.Request, error) {
	if request != nil {
		//nolint:exhaustive // The subscription proxy only exposes text chat and Responses.
		switch request.RequestType {
		case llm.RequestTypeChat, "":
		default:
			return nil, fmt.Errorf("%w: xAI subscription does not support %s requests", transformer.ErrInvalidRequest, request.RequestType)
		}
	}

	request, err := normalizeChatRequestToolSchemas(request)
	if err != nil {
		return nil, err
	}

	credentials, err := t.tokens.Get(ctx)
	if err != nil {
		return nil, err
	}

	httpRequest, err := t.chat.TransformRequest(ctx, request)
	if err != nil {
		return nil, err
	}

	applyCLIAuth(httpRequest, credentials.AccessToken)

	return httpRequest, nil
}

func (t *ChatOutboundTransformer) TransformResponse(ctx context.Context, response *httpclient.Response) (*llm.Response, error) {
	return t.chat.TransformResponse(ctx, response)
}

func (t *ChatOutboundTransformer) TransformStream(ctx context.Context, request *httpclient.Request, stream streams.Stream[*httpclient.StreamEvent]) (streams.Stream[*llm.Response], error) {
	return t.chat.TransformStream(ctx, request, stream)
}

func (t *ChatOutboundTransformer) TransformError(ctx context.Context, err *httpclient.Error) *llm.ResponseError {
	return t.chat.TransformError(ctx, err)
}

func (t *ChatOutboundTransformer) AggregateStreamChunks(ctx context.Context, request *httpclient.Request, chunks []*httpclient.StreamEvent) ([]byte, llm.ResponseMeta, error) {
	return t.chat.AggregateStreamChunks(ctx, request, chunks)
}
