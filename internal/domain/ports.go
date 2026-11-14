package domain

import (
	"context"
	"time"
)

// CacheRepository stores and retrieves cached LLM completions (exact and semantic).
type CacheRepository interface {
	LookupExact(ctx context.Context, namespace, normalizedPrompt string) (*CacheLookupResult, error)
	LookupSemantic(ctx context.Context, namespace string, embedding []float32, threshold float64) (*CacheLookupResult, error)
	Store(ctx context.Context, task CacheWriteTask, ttl time.Duration) error
}

// EmbeddingProvider produces vector embeddings for prompt text.
type EmbeddingProvider interface {
	Embed(ctx context.Context, texts []string) ([][]float32, error)
	Model() string
	Dimensions() int
}

// LLMProvider forwards chat completion requests to the upstream LLM API.
type LLMProvider interface {
	CreateChatCompletion(ctx context.Context, req ChatCompletionRequest) (*ChatCompletionResponse, error)
}
