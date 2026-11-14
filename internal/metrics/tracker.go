package metrics

import (
	"sync/atomic"
)

// StatsTracker aggregates request and cache outcomes with lock-free counters.
type StatsTracker struct {
	totalRequests             atomic.Uint64
	exactHits                 atomic.Uint64
	semanticHits              atomic.Uint64
	cacheMisses               atomic.Uint64
	totalLatencyMs            atomic.Uint64
	savedLatencyMs            atomic.Uint64
	estimatedCostSavedMicroUSD atomic.Int64
}

// Snapshot is a point-in-time view of tracker counters and derived rates.
type Snapshot struct {
	TotalRequests            uint64  `json:"total_requests"`
	ExactHits                uint64  `json:"exact_hits"`
	SemanticHits             uint64  `json:"semantic_hits"`
	CacheMisses              uint64  `json:"cache_misses"`
	TotalLatencyMs           uint64  `json:"total_latency_ms"`
	SavedLatencyMs           uint64  `json:"saved_latency_ms"`
	EstimatedCostSavedUSD    float64 `json:"estimated_cost_saved_usd"`
	HitRatePercent           float64 `json:"hit_rate_percent"`
	AvgLatencyMs             float64 `json:"avg_latency_ms"`
	AvgSavedLatencyMsPerHit  float64 `json:"avg_saved_latency_ms_per_hit"`
}

// RecordRequest updates counters for a completed request.
func (s *StatsTracker) RecordRequest(latencyMs uint64, exactHit, semanticHit, miss bool, savedLatencyMs uint64, costSavedUSD float64) {
	s.totalRequests.Add(1)
	s.totalLatencyMs.Add(latencyMs)
	if exactHit {
		s.exactHits.Add(1)
	}
	if semanticHit {
		s.semanticHits.Add(1)
	}
	if miss {
		s.cacheMisses.Add(1)
	}
	if savedLatencyMs > 0 {
		s.savedLatencyMs.Add(savedLatencyMs)
	}
	if costSavedUSD > 0 {
		micro := int64(costSavedUSD * 1_000_000)
		s.estimatedCostSavedMicroUSD.Add(micro)
	}
}

// Snapshot returns derived hit rate and average latency metrics.
func (s *StatsTracker) Snapshot() Snapshot {
	total := s.totalRequests.Load()
	exact := s.exactHits.Load()
	semantic := s.semanticHits.Load()
	misses := s.cacheMisses.Load()
	totalLat := s.totalLatencyMs.Load()
	savedLat := s.savedLatencyMs.Load()
	costMicro := s.estimatedCostSavedMicroUSD.Load()

	hits := exact + semantic
	var hitRate float64
	if total > 0 {
		hitRate = float64(hits) / float64(total) * 100
	}

	var avgLatency float64
	if total > 0 {
		avgLatency = float64(totalLat) / float64(total)
	}

	var avgSavedPerHit float64
	if hits > 0 {
		avgSavedPerHit = float64(savedLat) / float64(hits)
	}

	return Snapshot{
		TotalRequests:           total,
		ExactHits:               exact,
		SemanticHits:            semantic,
		CacheMisses:             misses,
		TotalLatencyMs:          totalLat,
		SavedLatencyMs:          savedLat,
		EstimatedCostSavedUSD:   float64(costMicro) / 1_000_000,
		HitRatePercent:          hitRate,
		AvgLatencyMs:            avgLatency,
		AvgSavedLatencyMsPerHit: avgSavedPerHit,
	}
}