# semantic-cache-proxy

High-throughput OpenAI-compatible proxy with **two-tier caching**:

1. **Exact tier** — SHA-256 keys on normalized prompts (`exact:<namespace>:<hash>`)
2. **Semantic tier** — Redis Stack RediSearch **HNSW** cosine similarity on embeddings

Cache misses call the upstream LLM asynchronously back-fill Redis via a bounded worker pool.

## Architecture

```
 Client                semantic-cache-proxy                         Upstream
   |                           |                                       |
   | POST /v1/chat/completions |                                       |
   |------------------------->| 1) Normalize + exact GET              |
   |                           | 2) Embed + FT.SEARCH KNN              |
   |                           | 3) On miss: LLM + async Store         |---> LLM API
   |                           |       (worker pool)                   |---> Embeddings API
   |<-------------------------|                                       |
   |  JSON + X-Cache-* headers |                                       |
   v                           v                                       v
                        +------------+
                        | Redis Stack|
                        | exact keys |
                        | HNSW index |
                        +------------+
```

## Quickstart

```bash
cp .env.example .env
# Set LLM_API_KEY (or point LLM_BASE_URL at local Ollama http://localhost:11434/v1)

docker compose up --build
```

Verify health and stats:

```bash
curl -s http://localhost:8080/health | jq
curl -s http://localhost:8080/stats | jq
```

### Chat completion (cache-aware)

```bash
curl -i http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Cache-Namespace: support" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "How do I request a refund?"}]
  }'
```

Response headers:

| Header | Example |
|--------|---------|
| `X-Cache-Status` | `HIT-EXACT`, `HIT-SEMANTIC`, or `MISS` |
| `X-Cache-Similarity` | `0.9234` (omitted on miss) |
| `X-Matched-Prompt` | prior prompt on semantic hit |
| `X-Latency-Ms` | end-to-end proxy latency |

Repeat the same normalized prompt for an exact hit; send a paraphrase for a semantic hit after the first request warms the cache.

## Configuration

See `.env.example` for all variables. Key settings:

| Variable | Default | Purpose |
|----------|---------|---------|
| `SERVER_PORT` | `8080` | HTTP listen port |
| `REDIS_ADDR` | `localhost:6379` | Redis Stack address |
| `LLM_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible API base |
| `SIMILARITY_THRESHOLD` | `0.90` | Minimum cosine similarity for semantic hits |
| `WORKER_POOL_SIZE` | `4` | Async cache writers |
| `WORKER_QUEUE_SIZE` | `256` | Back-pressure queue (drops when full) |

## Load benchmark

Run against a live proxy (after cache warm-up requests complete):

```bash
go run ./cmd/benchmark \
  -url http://localhost:8080 \
  -namespace benchmark \
  -c 8 \
  -n 120
```

The benchmark sends a mix of **identical**, **paraphrased**, and **unique** user messages and reports latency percentiles per `X-Cache-Status`.

### Example benchmark output

```
CACHE STATUS    COUNT   P50 (ms)  P95 (ms)  P99 (ms)
HIT-EXACT       48      3         6         9
HIT-SEMANTIC    36      18        42        58
MISS            36      812       1340      1588
UNKNOWN         0       -         -         -

Total samples: 120  Failures: 0
```

*(Illustrative numbers — yours depend on hardware, Redis, and upstream latency.)*

## Project layout

```
cmd/server/          HTTP entrypoint
cmd/benchmark/       CLI load generator
internal/api/        Handlers + logging middleware
internal/cache/      Normalization, Redis exact + HNSW semantic cache
internal/config/     Environment configuration
internal/domain/     Models and ports
internal/metrics/    Atomic stats tracker
internal/provider/   OpenAI-compatible LLM + embedding client
internal/worker/     Bounded cache write pool
```

## Development

```bash
# Requires Go 1.22+, Redis Stack with RediSearch
export $(grep -v '^#' .env | xargs)
go run ./cmd/server
go test ./...
```

## License

See [LICENSE](LICENSE).