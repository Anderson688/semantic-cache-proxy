package config

import (
	"context"
	"fmt"
	"os"
	"strconv"

	"github.com/joho/godotenv"
)

// Config holds runtime settings loaded from the environment.
type Config struct {
	ServerPort          string
	RedisAddr           string
	RedisPassword       string
	LLMBaseURL          string
	LLMAPIKey           string
	LLMModel            string
	EmbeddingModel      string
	EmbeddingDim        int
	SimilarityThreshold float64
	CacheTTLSeconds     int
	WorkerPoolSize      int
	WorkerQueueSize     int
}

// Load reads configuration from the environment. An optional .env file is loaded
// when present; missing .env files are ignored. ctx is reserved for future
// remote config sources and is currently unused.
func Load(ctx context.Context) (*Config, error) {
	_ = ctx

	if err := godotenv.Load(); err != nil && !os.IsNotExist(err) {
		return nil, fmt.Errorf("load .env: %w", err)
	}

	embeddingDim, err := envInt("EMBEDDING_DIM", 1536)
	if err != nil {
		return nil, err
	}

	similarityThreshold, err := envFloat("SIMILARITY_THRESHOLD", 0.90)
	if err != nil {
		return nil, err
	}

	cacheTTLSeconds, err := envInt("CACHE_TTL_SECONDS", 86400)
	if err != nil {
		return nil, err
	}

	workerPoolSize, err := envInt("WORKER_POOL_SIZE", 4)
	if err != nil {
		return nil, err
	}

	workerQueueSize, err := envInt("WORKER_QUEUE_SIZE", 256)
	if err != nil {
		return nil, err
	}

	cfg := &Config{
		ServerPort:          envString("SERVER_PORT", "8080"),
		RedisAddr:           envString("REDIS_ADDR", "localhost:6379"),
		RedisPassword:       envString("REDIS_PASSWORD", ""),
		LLMBaseURL:          envString("LLM_BASE_URL", "https://api.openai.com/v1"),
		LLMAPIKey:           envString("LLM_API_KEY", ""),
		LLMModel:            envString("LLM_MODEL", "gpt-4o-mini"),
		EmbeddingModel:      envString("EMBEDDING_MODEL", "text-embedding-3-small"),
		EmbeddingDim:        embeddingDim,
		SimilarityThreshold: similarityThreshold,
		CacheTTLSeconds:     cacheTTLSeconds,
		WorkerPoolSize:      workerPoolSize,
		WorkerQueueSize:     workerQueueSize,
	}

	return cfg, nil
}

func envString(key, defaultVal string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return defaultVal
}

func envInt(key string, defaultVal int) (int, error) {
	raw := os.Getenv(key)
	if raw == "" {
		return defaultVal, nil
	}
	n, err := strconv.Atoi(raw)
	if err != nil {
		return 0, fmt.Errorf("invalid %s %q: %w", key, raw, err)
	}
	return n, nil
}

func envFloat(key string, defaultVal float64) (float64, error) {
	raw := os.Getenv(key)
	if raw == "" {
		return defaultVal, nil
	}
	f, err := strconv.ParseFloat(raw, 64)
	if err != nil {
		return 0, fmt.Errorf("invalid %s %q: %w", key, raw, err)
	}
	return f, nil
}
