package api

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"github.com/anderson688/semantic-cache-proxy/internal/cache"
	"github.com/anderson688/semantic-cache-proxy/internal/domain"
	"github.com/anderson688/semantic-cache-proxy/internal/metrics"
	"github.com/redis/go-redis/v9"
)

const (
	headerCacheNamespace  = "X-Cache-Namespace"
	headerCacheStatus     = "X-Cache-Status"
	headerCacheSimilarity = "X-Cache-Similarity"
	headerMatchedPrompt   = "X-Matched-Prompt"
	headerLatencyMs       = "X-Latency-Ms"

	defaultNamespace = "default"

	// Baseline used to estimate latency savings on cache hits (ms).
	assumedMissLatencyMs = 1500

	// Rough output token cost for savings estimates (USD per 1K tokens).
	estimatedOutputCostPer1KTokens = 0.0006
)

// CacheWriter enqueues asynchronous cache persistence tasks.
type CacheWriter interface {
	Enqueue(task domain.CacheWriteTask) bool
}

// LLMCompleter generates assistant text for cache misses.
type LLMCompleter interface {
	GenerateCompletion(ctx context.Context, req domain.ChatCompletionRequest) (string, error)
}

// ChatHandler serves OpenAI-compatible chat routes backed by the semantic cache.
type ChatHandler struct {
	cache     domain.CacheRepository
	embedder  domain.EmbeddingProvider
	llm       LLMCompleter
	writer    CacheWriter
	stats     *metrics.StatsTracker
	redis     *redis.Client
	threshold float64
	logger    *slog.Logger
}

// NewChatHandler wires HTTP handlers for chat, health, and stats.
func NewChatHandler(
	cache domain.CacheRepository,
	embedder domain.EmbeddingProvider,
	llm LLMCompleter,
	writer CacheWriter,
	stats *metrics.StatsTracker,
	rdb *redis.Client,
	similarityThreshold float64,
	logger *slog.Logger,
) *ChatHandler {
	if logger == nil {
		logger = slog.Default()
	}
	return &ChatHandler{
		cache:     cache,
		embedder:  embedder,
		llm:       llm,
		writer:    writer,
		stats:     stats,
		redis:     rdb,
		threshold: similarityThreshold,
		logger:    logger,
	}
}

// Routes returns the HTTP handler tree for the API.
func (h *ChatHandler) Routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/chat/completions", h.handleChatCompletions)
	mux.HandleFunc("GET /health", h.handleHealth)
	mux.HandleFunc("GET /stats", h.handleStats)
	return LoggingMiddleware(h.logger, mux)
}

func (h *ChatHandler) handleHealth(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	if err := h.redis.Ping(ctx).Err(); err != nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{
			"status": "unhealthy",
			"redis":  err.Error(),
		})
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func (h *ChatHandler) handleStats(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, h.stats.Snapshot())
}

func (h *ChatHandler) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	ctx := r.Context()

	var req domain.ChatCompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid json body"})
		return
	}
	if len(req.Messages) == 0 {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "messages required"})
		return
	}

	namespace := strings.TrimSpace(r.Header.Get(headerCacheNamespace))
	if namespace == "" {
		namespace = defaultNamespace
	}
	if req.Namespace == "" {
		req.Namespace = namespace
	}

	userText, err := lastUserMessage(req.Messages)
	if err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
		return
	}

	normalized := cache.NormalizePrompt(userText)

	exact, err := h.cache.LookupExact(ctx, namespace, normalized)
	if err != nil {
		h.logger.Error("exact cache lookup failed", slog.String("error", err.Error()))
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "cache lookup failed"})
		return
	}
	if exact != nil && exact.Response != nil {
		latency := time.Since(start)
		h.writeCacheHeaders(w, "HIT-EXACT", 1.0, "", latency)
		h.recordHit(latency, true, false, exact.Response)
		writeJSON(w, http.StatusOK, exact.Response)
		return
	}

	vectors, err := h.embedder.Embed(ctx, []string{userText})
	if err != nil {
		h.logger.Error("embedding failed", slog.String("error", err.Error()))
		writeJSON(w, http.StatusBadGateway, map[string]string{"error": "embedding failed"})
		return
	}
	queryVec := vectors[0]

	semantic, err := h.cache.LookupSemantic(ctx, namespace, queryVec, h.threshold)
	if err != nil {
		h.logger.Error("semantic cache lookup failed", slog.String("error", err.Error()))
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "cache lookup failed"})
		return
	}
	if semantic != nil && semantic.Response != nil {
		latency := time.Since(start)
		h.writeCacheHeaders(w, "HIT-SEMANTIC", semantic.SimilarityScore, semantic.MatchedPrompt, latency)
		h.recordHit(latency, false, true, semantic.Response)
		writeJSON(w, http.StatusOK, semantic.Response)
		return
	}

	text, err := h.llm.GenerateCompletion(ctx, req)
	if err != nil {
		h.logger.Error("llm completion failed", slog.String("error", err.Error()))
		writeJSON(w, http.StatusBadGateway, map[string]string{"error": "upstream llm failed"})
		return
	}

	model := req.Model
	if model == "" {
		model = "gpt-4o-mini"
	}
	resp := buildCompletionResponse(model, text)
	latency := time.Since(start)

	h.writeCacheHeaders(w, "MISS", 0, "", latency)
	h.stats.RecordRequest(uint64(latency.Milliseconds()), false, false, true, 0, 0)

	_ = h.writer.Enqueue(domain.CacheWriteTask{
		Namespace:        namespace,
		NormalizedPrompt: normalized,
		RawPrompt:        userText,
		Embedding:        queryVec,
		Response:         *resp,
		EnqueuedAt:       time.Now(),
	})

	writeJSON(w, http.StatusOK, resp)
}

func (h *ChatHandler) writeCacheHeaders(w http.ResponseWriter, status string, similarity float64, matched string, latency time.Duration) {
	w.Header().Set(headerCacheStatus, status)
	if status != "MISS" {
		w.Header().Set(headerCacheSimilarity, fmt.Sprintf("%.4f", similarity))
	}
	if matched != "" {
		w.Header().Set(headerMatchedPrompt, matched)
	}
	w.Header().Set(headerLatencyMs, fmt.Sprintf("%d", latency.Milliseconds()))
}

func (h *ChatHandler) recordHit(latency time.Duration, exact, semantic bool, resp *domain.ChatCompletionResponse) {
	latMs := uint64(latency.Milliseconds())
	var saved uint64
	if assumedMissLatencyMs > latMs {
		saved = assumedMissLatencyMs - latMs
	}
	cost := estimateCostSavedUSD(resp)
	h.stats.RecordRequest(latMs, exact, semantic, false, saved, cost)
}

func estimateCostSavedUSD(resp *domain.ChatCompletionResponse) float64 {
	if resp == nil {
		return 0
	}
	tokens := resp.Usage.CompletionTokens
	if tokens == 0 && len(resp.Choices) > 0 {
		tokens = len(resp.Choices[0].Message.Content) / 4
		if tokens < 1 {
			tokens = 1
		}
	}
	return float64(tokens) / 1000 * estimatedOutputCostPer1KTokens
}

func lastUserMessage(messages []domain.ChatMessage) (string, error) {
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == "user" {
			content := strings.TrimSpace(messages[i].Content)
			if content == "" {
				return "", fmt.Errorf("last user message is empty")
			}
			return content, nil
		}
	}
	return "", fmt.Errorf("no user message found")
}

func buildCompletionResponse(model, assistantText string) *domain.ChatCompletionResponse {
	return &domain.ChatCompletionResponse{
		ID:      fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano()),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   model,
		Choices: []domain.ChatCompletionChoice{{
			Index:        0,
			Message:      domain.ChatMessage{Role: "assistant", Content: assistantText},
			FinishReason: "stop",
		}},
		Usage: domain.UsageTokenCounts{
			CompletionTokens: len(assistantText) / 4,
			TotalTokens:      len(assistantText) / 4,
		},
	}
}

func writeJSON(w http.ResponseWriter, status int, payload any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(payload)
}
