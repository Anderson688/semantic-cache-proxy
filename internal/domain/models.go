package domain

import "time"

// ChatMessage is a single turn in an OpenAI-compatible chat completion request.
type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ChatCompletionRequest mirrors the OpenAI chat completions API with an optional
// namespace for multi-tenant cache isolation.
type ChatCompletionRequest struct {
	Model     string        `json:"model"`
	Messages  []ChatMessage `json:"messages"`
	Namespace string        `json:"namespace,omitempty"`
}

// ChatCompletionChoice is one completion choice in the response payload.
type ChatCompletionChoice struct {
	Index        int         `json:"index"`
	Message      ChatMessage `json:"message"`
	FinishReason string      `json:"finish_reason,omitempty"`
}

// UsageTokenCounts reports token usage for a completion.
type UsageTokenCounts struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// ChatCompletionResponse is an OpenAI-compatible chat completion response.
type ChatCompletionResponse struct {
	ID      string                 `json:"id"`
	Object  string                 `json:"object"`
	Created int64                  `json:"created"`
	Model   string                 `json:"model"`
	Choices []ChatCompletionChoice `json:"choices"`
	Usage   UsageTokenCounts       `json:"usage,omitempty"`
}

// CacheStatus describes how a cache lookup was satisfied.
type CacheStatus string

const (
	CacheStatusHitExact    CacheStatus = "HIT_EXACT"
	CacheStatusHitSemantic CacheStatus = "HIT_SEMANTIC"
	CacheStatusMiss        CacheStatus = "MISS"
)

// CacheLookupResult is the outcome of an exact or semantic cache lookup.
type CacheLookupResult struct {
	Response        *ChatCompletionResponse `json:"response,omitempty"`
	Status          CacheStatus             `json:"status"`
	SimilarityScore float64                 `json:"similarity_score,omitempty"`
	MatchedPrompt   string                  `json:"matched_prompt,omitempty"`
}

// CacheWriteTask is enqueued for asynchronous persistence after a cache miss.
type CacheWriteTask struct {
	Namespace        string
	NormalizedPrompt string
	RawPrompt        string
	Embedding        []float32
	Response         ChatCompletionResponse
	EnqueuedAt       time.Time
}