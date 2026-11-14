package cache

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/anderson688/semantic-cache-proxy/internal/domain"
	"github.com/google/uuid"
	"github.com/redis/go-redis/v9"
)

const (
	semanticIndexName = "idx:semantic_cache"
	docKeyPrefix      = "doc:"
)

// RedisCache implements domain.CacheRepository using exact keys and RediSearch HNSW.
type RedisCache struct {
	client       *redis.Client
	embeddingDim int
}

// NewRedisCache wires a CacheRepository backed by the given Redis client.
func NewRedisCache(client *redis.Client, embeddingDim int) *RedisCache {
	return &RedisCache{
		client:       client,
		embeddingDim: embeddingDim,
	}
}

var _ domain.CacheRepository = (*RedisCache)(nil)

// EnsureIndex creates the RediSearch HNSW index when it does not already exist.
func (c *RedisCache) EnsureIndex(ctx context.Context) error {
	_, err := c.client.Do(ctx, "FT.INFO", semanticIndexName).Result()
	if err == nil {
		return nil
	}
	if !isIndexMissing(err) {
		return fmt.Errorf("ft.info %s: %w", semanticIndexName, err)
	}

	dim := strconv.Itoa(c.embeddingDim)
	args := []interface{}{
		"FT.CREATE", semanticIndexName,
		"ON", "HASH",
		"PREFIX", "1", docKeyPrefix,
		"SCHEMA",
		"namespace", "TAG",
		"prompt", "TEXT",
		"response", "TEXT",
		"embedding", "VECTOR", "HNSW", "6",
		"TYPE", "FLOAT32",
		"DIM", dim,
		"DISTANCE_METRIC", "COSINE",
	}
	if err := c.client.Do(ctx, args...).Err(); err != nil {
		return fmt.Errorf("ft.create %s: %w", semanticIndexName, err)
	}
	return nil
}

// LookupExact loads a response by normalized prompt hash.
func (c *RedisCache) LookupExact(ctx context.Context, namespace, normalizedPrompt string) (*domain.CacheLookupResult, error) {
	key := HashPrompt(namespace, normalizedPrompt)
	raw, err := c.client.Get(ctx, key).Bytes()
	if err != nil {
		if errors.Is(err, redis.Nil) {
			return nil, nil
		}
		return nil, fmt.Errorf("get exact cache key %q: %w", key, err)
	}

	resp, err := decodeResponse(raw)
	if err != nil {
		return nil, fmt.Errorf("decode exact cache payload: %w", err)
	}

	return &domain.CacheLookupResult{
		Response:        resp,
		Status:          domain.CacheStatusHitExact,
		SimilarityScore: 1.0,
	}, nil
}

// LookupSemantic runs a KNN vector search scoped to namespace.
func (c *RedisCache) LookupSemantic(ctx context.Context, namespace string, queryVec []float32, threshold float64) (*domain.CacheLookupResult, error) {
	if len(queryVec) == 0 {
		return nil, nil
	}

	vecBytes := Float32SliceToBytes(queryVec)
	escapedNS := escapeRediSearchTag(namespace)
	query := fmt.Sprintf(`(@namespace:{%s})=>[KNN 1 @embedding $vec AS vector_distance]`, escapedNS)

	raw, err := c.client.Do(ctx,
		"FT.SEARCH", semanticIndexName,
		query,
		"PARAMS", 2, "vec", vecBytes,
		"RETURN", 3, "prompt", "response", "vector_distance",
		"SORTBY", "vector_distance", "ASC",
		"DIALECT", 2,
	).Result()
	if err != nil {
		return nil, fmt.Errorf("ft.search semantic lookup: %w", err)
	}

	return parseSemanticSearchResult(raw, threshold)
}

// Store persists exact and semantic cache entries atomically.
func (c *RedisCache) Store(ctx context.Context, task domain.CacheWriteTask, ttl time.Duration) error {
	payload, err := json.Marshal(task.Response)
	if err != nil {
		return fmt.Errorf("marshal cache response: %w", err)
	}

	exactKey := HashPrompt(task.Namespace, task.NormalizedPrompt)
	docID := docKeyPrefix + uuid.New().String()
	vecBytes := Float32SliceToBytes(task.Embedding)
	ttlSec := int64(ttl.Seconds())
	if ttlSec < 1 {
		ttlSec = 1
	}

	pipe := c.client.TxPipeline()
	pipe.Set(ctx, exactKey, payload, ttl)
	pipe.HSet(ctx, docID,
		"namespace", task.Namespace,
		"prompt", task.RawPrompt,
		"response", string(payload),
		"embedding", vecBytes,
	)
	pipe.Expire(ctx, docID, ttl)

	if _, err := pipe.Exec(ctx); err != nil {
		return fmt.Errorf("store cache entries: %w", err)
	}
	return nil
}

func decodeResponse(raw []byte) (*domain.ChatCompletionResponse, error) {
	var resp domain.ChatCompletionResponse
	if err := json.Unmarshal(raw, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

func isIndexMissing(err error) bool {
	if err == nil {
		return false
	}
	msg := strings.ToLower(err.Error())
	return strings.Contains(msg, "unknown index name") ||
		strings.Contains(msg, "no such index")
}

func escapeRediSearchTag(value string) string {
	var b strings.Builder
	for _, r := range value {
		switch r {
		case '\\', ',', '.', '<', '>', '{', '}', '"', '\'', ':', ';', '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '-', '+', '=', '~', '|', '/':
			b.WriteByte('\\')
		}
		b.WriteRune(r)
	}
	return b.String()
}

func parseSemanticSearchResult(raw interface{}, threshold float64) (*domain.CacheLookupResult, error) {
	top, err := firstSearchHit(raw)
	if err != nil {
		return nil, err
	}
	if top == nil {
		return nil, nil
	}

	distance, err := strconv.ParseFloat(top.fields["vector_distance"], 64)
	if err != nil {
		return nil, fmt.Errorf("parse vector_distance %q: %w", top.fields["vector_distance"], err)
	}

	similarity := 1.0 - distance
	if similarity < threshold {
		return nil, nil
	}

	resp, err := decodeResponse([]byte(top.fields["response"]))
	if err != nil {
		return nil, fmt.Errorf("decode semantic cache response: %w", err)
	}

	return &domain.CacheLookupResult{
		Response:        resp,
		Status:          domain.CacheStatusHitSemantic,
		SimilarityScore: similarity,
		MatchedPrompt:   top.fields["prompt"],
	}, nil
}

type searchHit struct {
	fields map[string]string
}

func firstSearchHit(raw interface{}) (*searchHit, error) {
	arr, ok := raw.([]interface{})
	if !ok || len(arr) == 0 {
		return nil, nil
	}

	total := int64FromRedis(arr[0])
	if total == 0 || len(arr) < 3 {
		return nil, nil
	}

	fieldsArr, ok := arr[2].([]interface{})
	if !ok {
		return nil, fmt.Errorf("unexpected ft.search field payload type %T", arr[2])
	}

	fields := make(map[string]string, len(fieldsArr)/2)
	for i := 0; i+1 < len(fieldsArr); i += 2 {
		key, err := redisString(fieldsArr[i])
		if err != nil {
			return nil, fmt.Errorf("parse ft.search field name: %w", err)
		}
		val, err := redisString(fieldsArr[i+1])
		if err != nil {
			return nil, fmt.Errorf("parse ft.search field value for %q: %w", key, err)
		}
		fields[key] = val
	}

	return &searchHit{fields: fields}, nil
}

func int64FromRedis(v interface{}) int64 {
	switch n := v.(type) {
	case int64:
		return n
	case int:
		return int64(n)
	case string:
		i, _ := strconv.ParseInt(n, 10, 64)
		return i
	default:
		return 0
	}
}

func redisString(v interface{}) (string, error) {
	switch s := v.(type) {
	case string:
		return s, nil
	case []byte:
		return string(s), nil
	default:
		return "", fmt.Errorf("unsupported redis string type %T", v)
	}
}
