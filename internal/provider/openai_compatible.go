package provider

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/anderson688/semantic-cache-proxy/internal/domain"
)

const (
	defaultBaseURL        = "https://api.openai.com/v1"
	defaultRequestTimeout = 60 * time.Second
)

// ErrRateLimited indicates the upstream API returned HTTP 429.
var ErrRateLimited = errors.New("rate limit exceeded")

// APIError describes a non-success response from the upstream API.
type APIError struct {
	StatusCode int
	Message    string
	Body       string
}

func (e *APIError) Error() string {
	if e.Message != "" {
		return fmt.Sprintf("upstream api error %d: %s", e.StatusCode, e.Message)
	}
	return fmt.Sprintf("upstream api error %d", e.StatusCode)
}

// OpenAICompatibleClient calls OpenAI-compatible /embeddings and /chat/completions endpoints.
type OpenAICompatibleClient struct {
	baseURL      string
	apiKey       string
	httpClient   *http.Client
	llmModel     string
	embeddingModel string
	embeddingDim int
}

// ClientOption configures optional behavior for OpenAICompatibleClient.
type ClientOption func(*OpenAICompatibleClient)

// WithHTTPClient replaces the default HTTP client (transport, timeouts).
func WithHTTPClient(client *http.Client) ClientOption {
	return func(c *OpenAICompatibleClient) {
		if client != nil {
			c.httpClient = client
		}
	}
}

// NewOpenAICompatibleClient constructs a provider for OpenAI, Ollama, OpenRouter, etc.
func NewOpenAICompatibleClient(baseURL, apiKey, llmModel, embeddingModel string, embeddingDim int, opts ...ClientOption) *OpenAICompatibleClient {
	base := strings.TrimRight(strings.TrimSpace(baseURL), "/")
	if base == "" {
		base = defaultBaseURL
	}

	c := &OpenAICompatibleClient{
		baseURL:        base,
		apiKey:         apiKey,
		llmModel:       llmModel,
		embeddingModel: embeddingModel,
		embeddingDim:   embeddingDim,
		httpClient: &http.Client{
			Timeout: defaultRequestTimeout,
			Transport: &http.Transport{
				MaxIdleConns:        100,
				MaxIdleConnsPerHost: 20,
				IdleConnTimeout:     90 * time.Second,
			},
		},
	}
	for _, opt := range opts {
		opt(c)
	}
	return c
}

var (
	_ domain.EmbeddingProvider = (*OpenAICompatibleClient)(nil)
	_ domain.LLMProvider       = (*OpenAICompatibleClient)(nil)
)

// Model returns the configured embedding model name.
func (c *OpenAICompatibleClient) Model() string {
	return c.embeddingModel
}

// Dimensions returns the expected embedding vector size.
func (c *OpenAICompatibleClient) Dimensions() int {
	return c.embeddingDim
}

// Embed requests vectors for one or more input strings.
func (c *OpenAICompatibleClient) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}

	type embeddingReq struct {
		Model string      `json:"model"`
		Input interface{} `json:"input"`
	}
	type embeddingData struct {
		Index     int       `json:"index"`
		Embedding []float64 `json:"embedding"`
	}
	type embeddingResp struct {
		Data []embeddingData `json:"data"`
	}

	payload := embeddingReq{
		Model: c.embeddingModel,
		Input: texts,
	}
	if len(texts) == 1 {
		payload.Input = texts[0]
	}

	var resp embeddingResp
	if err := c.postJSON(ctx, "/embeddings", payload, &resp); err != nil {
		return nil, fmt.Errorf("embed: %w", err)
	}
	if len(resp.Data) == 0 {
		return nil, fmt.Errorf("embed: empty data in response")
	}

	out := make([][]float32, len(texts))
	for _, item := range resp.Data {
		if item.Index < 0 || item.Index >= len(out) {
			return nil, fmt.Errorf("embed: invalid embedding index %d", item.Index)
		}
		out[item.Index] = float64SliceTo32(item.Embedding)
	}
	for i, vec := range out {
		if vec == nil {
			return nil, fmt.Errorf("embed: missing vector for input index %d", i)
		}
	}
	return out, nil
}

// GetEmbedding returns a single embedding vector for text.
func (c *OpenAICompatibleClient) GetEmbedding(ctx context.Context, text string) ([]float32, error) {
	vectors, err := c.Embed(ctx, []string{text})
	if err != nil {
		return nil, err
	}
	return vectors[0], nil
}

// CreateChatCompletion calls the upstream chat completions API.
func (c *OpenAICompatibleClient) CreateChatCompletion(ctx context.Context, req domain.ChatCompletionRequest) (*domain.ChatCompletionResponse, error) {
	model := req.Model
	if model == "" {
		model = c.llmModel
	}

	type chatReq struct {
		Model    string               `json:"model"`
		Messages []domain.ChatMessage `json:"messages"`
	}

	payload := chatReq{
		Model:    model,
		Messages: req.Messages,
	}

	var resp domain.ChatCompletionResponse
	if err := c.postJSON(ctx, "/chat/completions", payload, &resp); err != nil {
		return nil, fmt.Errorf("chat completion: %w", err)
	}
	if len(resp.Choices) == 0 {
		return nil, fmt.Errorf("chat completion: empty choices in response")
	}
	return &resp, nil
}

// GenerateCompletion returns the assistant message text from a chat completion request.
func (c *OpenAICompatibleClient) GenerateCompletion(ctx context.Context, req domain.ChatCompletionRequest) (string, error) {
	resp, err := c.CreateChatCompletion(ctx, req)
	if err != nil {
		return "", err
	}
	content := strings.TrimSpace(resp.Choices[0].Message.Content)
	if content == "" {
		return "", fmt.Errorf("chat completion: empty assistant content")
	}
	return content, nil
}

func (c *OpenAICompatibleClient) postJSON(ctx context.Context, path string, payload any, out any) error {
	body, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("marshal request: %w", err)
	}

	url := c.baseURL + path
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if c.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.apiKey)
	}

	res, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("http do: %w", err)
	}
	defer res.Body.Close()

	raw, err := io.ReadAll(io.LimitReader(res.Body, 1<<20))
	if err != nil {
		return fmt.Errorf("read response: %w", err)
	}

	if res.StatusCode != http.StatusOK {
		apiErr := parseAPIError(res.StatusCode, raw)
		if res.StatusCode == http.StatusTooManyRequests {
			return fmt.Errorf("%w: %s", ErrRateLimited, apiErr.Error())
		}
		return apiErr
	}

	if err := json.Unmarshal(raw, out); err != nil {
		return fmt.Errorf("decode response: %w", err)
	}
	return nil
}

func parseAPIError(statusCode int, raw []byte) error {
	type apiErrorBody struct {
		Error struct {
			Message string `json:"message"`
			Type    string `json:"type"`
			Code    any    `json:"code"`
		} `json:"error"`
	}

	var parsed apiErrorBody
	_ = json.Unmarshal(raw, &parsed)

	msg := strings.TrimSpace(parsed.Error.Message)
	if msg == "" {
		msg = strings.TrimSpace(string(raw))
	}

	return &APIError{
		StatusCode: statusCode,
		Message:    msg,
		Body:       string(raw),
	}
}

func float64SliceTo32(in []float64) []float32 {
	out := make([]float32, len(in))
	for i, v := range in {
		out[i] = float32(v)
	}
	return out
}