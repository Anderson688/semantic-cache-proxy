package main

import (
	"context"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/anderson688/semantic-cache-proxy/internal/api"
	"github.com/anderson688/semantic-cache-proxy/internal/cache"
	"github.com/anderson688/semantic-cache-proxy/internal/config"
	"github.com/anderson688/semantic-cache-proxy/internal/metrics"
	"github.com/anderson688/semantic-cache-proxy/internal/provider"
	"github.com/anderson688/semantic-cache-proxy/internal/worker"
	"github.com/redis/go-redis/v9"
)

func main() {
	logger := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelInfo}))
	slog.SetDefault(logger)

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	cfg, err := config.Load(ctx)
	if err != nil {
		logger.Error("failed to load configuration", slog.String("error", err.Error()))
		os.Exit(1)
	}

	rdb := redis.NewClient(&redis.Options{
		Addr:     cfg.RedisAddr,
		Password: cfg.RedisPassword,
	})

	pingCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	if err := rdb.Ping(pingCtx).Err(); err != nil {
		cancel()
		logger.Error("redis ping failed", slog.String("error", err.Error()))
		os.Exit(1)
	}
	cancel()

	cacheRepo := cache.NewRedisCache(rdb, cfg.EmbeddingDim)
	if err := cacheRepo.EnsureIndex(ctx); err != nil {
		logger.Error("ensure search index failed", slog.String("error", err.Error()))
		os.Exit(1)
	}

	llmClient := provider.NewOpenAICompatibleClient(
		cfg.LLMBaseURL,
		cfg.LLMAPIKey,
		cfg.LLMModel,
		cfg.EmbeddingModel,
		cfg.EmbeddingDim,
	)

	stats := &metrics.StatsTracker{}
	cacheTTL := time.Duration(cfg.CacheTTLSeconds) * time.Second

	writeWorker := worker.NewCacheWriteWorker(
		cfg.WorkerPoolSize,
		cfg.WorkerQueueSize,
		cacheRepo,
		llmClient,
		cacheTTL,
		logger,
	)

	handler := api.NewChatHandler(
		cacheRepo,
		llmClient,
		llmClient,
		writeWorker,
		stats,
		rdb,
		cfg.SimilarityThreshold,
		logger,
	)

	addr := ":" + cfg.ServerPort
	server := &http.Server{
		Addr:              addr,
		Handler:           handler.Routes(),
		ReadHeaderTimeout: 10 * time.Second,
	}

	go func() {
		logger.Info("server listening", slog.String("addr", addr))
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			logger.Error("server failed", slog.String("error", err.Error()))
			os.Exit(1)
		}
	}()

	<-ctx.Done()
	logger.Info("shutting down")

	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer shutdownCancel()

	if err := server.Shutdown(shutdownCtx); err != nil {
		logger.Error("http shutdown failed", slog.String("error", err.Error()))
	}

	workerCtx, workerCancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer workerCancel()
	if err := writeWorker.Shutdown(workerCtx); err != nil {
		logger.Warn("cache worker shutdown", slog.String("error", err.Error()))
	}

	if err := rdb.Close(); err != nil {
		logger.Warn("redis close", slog.String("error", err.Error()))
	}
}
