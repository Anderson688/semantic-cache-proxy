package cache

import (
	"crypto/sha256"
	"encoding/hex"
	"strings"
)

const exactKeyPrefix = "exact:"

// NormalizePrompt canonicalizes user text for exact-match cache keys.
func NormalizePrompt(raw string) string {
	s := strings.TrimSpace(raw)
	s = strings.ToLower(s)
	s = strings.Join(strings.Fields(s), " ")
	s = strings.TrimRight(s, "?!.")
	return s
}

// HashPrompt returns the Redis exact-cache key for a namespace and normalized prompt.
func HashPrompt(namespace, normalizedPrompt string) string {
	sum := sha256.Sum256([]byte(normalizedPrompt))
	return exactKeyPrefix + namespace + ":" + hex.EncodeToString(sum[:])
}
