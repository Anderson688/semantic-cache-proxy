package worker

import (
	"context"
	"fmt"
	"log/slog"
	"sync"
	"time"

	"github.com/anderson688/semantic-cache-proxy/internal/domain"
)

const storeTimeout = 5 * time.Second

// CacheWriteWorker asynchronously persists cache entries with bounded concurrency.
type CacheWriteWorker struct {
	tasks    chan domain.CacheWriteTask
	repo     domain.CacheRepository
	embedder domain.EmbeddingProvider
	ttl      time.Duration
	logger   *slog.Logger

	wg       sync.WaitGroup
	stopOnce sync.Once
	mu       sync.RWMutex
	stopped  bool
}

// NewCacheWriteWorker starts WORKER_POOL_SIZE goroutines draining a buffered task queue.
func NewCacheWriteWorker(
	poolSize int,
	queueSize int,
	repo domain.CacheRepository,
	embedder domain.EmbeddingProvider,
	ttl time.Duration,
	logger *slog.Logger,
) *CacheWriteWorker {
	if poolSize < 1 {
		poolSize = 1
	}
	if queueSize < 1 {
		queueSize = 1
	}
	if logger == nil {
		logger = slog.Default()
	}

	w := &CacheWriteWorker{
		tasks:    make(chan domain.CacheWriteTask, queueSize),
		repo:     repo,
		embedder: embedder,
		ttl:      ttl,
		logger:   logger,
	}

	for i := 0; i < poolSize; i++ {
		w.wg.Add(1)
		go w.runWorker()
	}
	return w
}

// Enqueue attempts a non-blocking send. Returns false when shut down or the queue is full.
func (w *CacheWriteWorker) Enqueue(task domain.CacheWriteTask) bool {
	w.mu.RLock()
	stopped := w.stopped
	w.mu.RUnlock()
	if stopped {
		return false
	}

	select {
	case w.tasks <- task:
		return true
	default:
		w.logger.Warn("cache write queue full, dropping task",
			slog.String("namespace", task.Namespace),
			slog.String("normalized_prompt", task.NormalizedPrompt),
		)
		return false
	}
}

// Shutdown closes the task channel and waits for in-flight work or ctx cancellation.
func (w *CacheWriteWorker) Shutdown(ctx context.Context) error {
	w.stopOnce.Do(func() {
		w.mu.Lock()
		w.stopped = true
		w.mu.Unlock()
		close(w.tasks)
	})

	done := make(chan struct{})
	go func() {
		w.wg.Wait()
		close(done)
	}()

	select {
	case <-done:
		return nil
	case <-ctx.Done():
		return fmt.Errorf("cache write worker shutdown: %w", ctx.Err())
	}
}

func (w *CacheWriteWorker) runWorker() {
	defer w.wg.Done()

	for task := range w.tasks {
		w.handleTask(task)
	}
}

func (w *CacheWriteWorker) handleTask(task domain.CacheWriteTask) {
	if len(task.Embedding) == 0 {
		text := task.RawPrompt
		if text == "" {
			text = task.NormalizedPrompt
		}

		embedCtx, cancel := context.WithTimeout(context.Background(), storeTimeout)
		vectors, err := w.embedder.Embed(embedCtx, []string{text})
		cancel()
		if err != nil {
			w.logger.Error("cache write embedding failed",
				slog.String("namespace", task.Namespace),
				slog.String("error", err.Error()),
			)
			return
		}
		task.Embedding = vectors[0]
	}

	storeCtx, cancel := context.WithTimeout(context.Background(), storeTimeout)
	defer cancel()

	if err := w.repo.Store(storeCtx, task, w.ttl); err != nil {
		w.logger.Error("cache write store failed",
			slog.String("namespace", task.Namespace),
			slog.String("error", err.Error()),
		)
	}
}