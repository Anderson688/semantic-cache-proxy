package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"text/tabwriter"
	"time"

	"github.com/anderson688/semantic-cache-proxy/internal/domain"
)

type workloadKind string

const (
	kindExact     workloadKind = "HIT-EXACT"
	kindSemantic  workloadKind = "HIT-SEMANTIC"
	kindMiss      workloadKind = "MISS"
	kindUnknown   workloadKind = "UNKNOWN"
)

type requestSpec struct {
	kind     workloadKind
	messages []domain.ChatMessage
}

type sample struct {
	kind    workloadKind
	latency time.Duration
}

func main() {
	baseURL := flag.String("url", "http://localhost:8080", "proxy base URL")
	concurrency := flag.Int("c", 8, "concurrent workers")
	total := flag.Int("n", 120, "total requests")
	namespace := flag.String("namespace", "benchmark", "X-Cache-Namespace header value")
	apiKey := flag.String("api-key", "", "optional Authorization bearer token for upstream (unused by proxy)")
	timeout := flag.Duration("timeout", 90*time.Second, "per-request HTTP timeout")
	flag.Parse()

	client := &http.Client{Timeout: *timeout}
	specs := buildWorkload(*total)

	samples := make([]sample, 0, len(specs))
	var samplesMu sync.Mutex
	var failures atomic.Uint64

	jobs := make(chan requestSpec, len(specs))
	for _, s := range specs {
		jobs <- s
	}
	close(jobs)

	var wg sync.WaitGroup
	for i := 0; i < *concurrency; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for job := range jobs {
				lat, kind, err := sendRequest(client, *baseURL, *namespace, *apiKey, job.messages)
				if err != nil {
					failures.Add(1)
					fmt.Fprintf(os.Stderr, "request failed: %v\n", err)
					continue
				}
				samplesMu.Lock()
				samples = append(samples, sample{kind: kind, latency: lat})
				samplesMu.Unlock()
				_ = job.kind
			}
		}()
	}
	wg.Wait()

	printSummary(samples, failures.Load())
}

func buildWorkload(total int) []requestSpec {
	exactPrompt := "How do I request a refund?"
	paraphrases := []string{
		"My package is damaged, how do I get my money back?",
		"What's the process to receive a refund for my order?",
		"Can I get reimbursed if the product arrived broken?",
	}

	var specs []requestSpec
	for i := 0; i < total; i++ {
		switch i % 10 {
		case 0, 1, 2, 3:
			specs = append(specs, requestSpec{
				kind:     kindExact,
				messages: []domain.ChatMessage{{Role: "user", Content: exactPrompt}},
			})
		case 4, 5, 6:
			p := paraphrases[i%len(paraphrases)]
			specs = append(specs, requestSpec{
				kind:     kindSemantic,
				messages: []domain.ChatMessage{{Role: "user", Content: p}},
			})
		default:
			specs = append(specs, requestSpec{
				kind:     kindMiss,
				messages: []domain.ChatMessage{{Role: "user", Content: fmt.Sprintf("Unique troubleshooting question %d about firmware rollback", i)}},
			})
		}
	}
	return specs
}

func sendRequest(client *http.Client, baseURL, namespace, apiKey string, messages []domain.ChatMessage) (time.Duration, workloadKind, error) {
	payload := domain.ChatCompletionRequest{
		Model:    "gpt-4o-mini",
		Messages: messages,
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return 0, kindUnknown, err
	}

	url := strings.TrimRight(baseURL, "/") + "/v1/chat/completions"
	req, err := http.NewRequest(http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return 0, kindUnknown, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Cache-Namespace", namespace)
	if apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+apiKey)
	}

	start := time.Now()
	res, err := client.Do(req)
	latency := time.Since(start)
	if err != nil {
		return latency, kindUnknown, err
	}
	defer res.Body.Close()
	_, _ = io.Copy(io.Discard, res.Body)

	if res.StatusCode != http.StatusOK {
		return latency, kindUnknown, fmt.Errorf("status %d", res.StatusCode)
	}

	status := workloadKind(res.Header.Get("X-Cache-Status"))
	if status == "" {
		status = kindUnknown
	}
	return latency, status, nil
}

func printSummary(samples []sample, failures uint64) {
	groups := map[workloadKind][]time.Duration{
		kindExact:    {},
		kindSemantic: {},
		kindMiss:     {},
		kindUnknown:  {},
	}
	for _, s := range samples {
		groups[s.kind] = append(groups[s.kind], s.latency)
	}

	w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(w, "CACHE STATUS\tCOUNT\tP50 (ms)\tP95 (ms)\tP99 (ms)")
	for _, kind := range []workloadKind{kindExact, kindSemantic, kindMiss, kindUnknown} {
		vals := groups[kind]
		if len(vals) == 0 {
			fmt.Fprintf(w, "%s\t0\t-\t-\t-\n", kind)
			continue
		}
		fmt.Fprintf(w, "%s\t%d\t%s\t%s\t%s\n",
			kind,
			len(vals),
			formatMs(percentile(vals, 0.50)),
			formatMs(percentile(vals, 0.95)),
			formatMs(percentile(vals, 0.99)),
		)
	}
	_ = w.Flush()

	fmt.Printf("\nTotal samples: %d  Failures: %d\n", len(samples), failures)
}

func percentile(values []time.Duration, p float64) time.Duration {
	if len(values) == 0 {
		return 0
	}
	cp := append([]time.Duration(nil), values...)
	sort.Slice(cp, func(i, j int) bool { return cp[i] < cp[j] })
	idx := int(float64(len(cp)-1) * p)
	return cp[idx]
}

func formatMs(d time.Duration) string {
	return fmt.Sprintf("%d", d.Milliseconds())
}