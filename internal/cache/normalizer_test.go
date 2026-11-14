package cache

import (
	"encoding/binary"
	"math"
	"testing"
)

func TestNormalizePrompt(t *testing.T) {
	tests := []struct {
		name string
		raw  string
		want string
	}{
		{
			name: "trim and lowercase",
			raw:  "  Hello World  ",
			want: "hello world",
		},
		{
			name: "collapse newlines and spaces",
			raw:  "What is\n\nGo   language?",
			want: "what is go language",
		},
		{
			name: "strip trailing question mark",
			raw:  "How are you?",
			want: "how are you",
		},
		{
			name: "strip trailing exclamation",
			raw:  "Hello there!",
			want: "hello there",
		},
		{
			name: "strip trailing period",
			raw:  "The answer is 42.",
			want: "the answer is 42",
		},
		{
			name: "strip mixed trailing punctuation",
			raw:  "Really?!.",
			want: "really",
		},
		{
			name: "preserve internal punctuation",
			raw:  "What's up, friend?",
			want: "what's up, friend",
		},
		{
			name: "empty after trim",
			raw:  "   ",
			want: "",
		},
		{
			name: "tabs collapsed",
			raw:  "a\t\tb\t c",
			want: "a b c",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := NormalizePrompt(tt.raw)
			if got != tt.want {
				t.Fatalf("NormalizePrompt(%q) = %q, want %q", tt.raw, got, tt.want)
			}
		})
	}
}

func TestHashPromptFormatAndStability(t *testing.T) {
	normalized := NormalizePrompt("Hello World?")
	key1 := HashPrompt("tenant-a", normalized)
	key2 := HashPrompt("tenant-a", normalized)

	if key1 != key2 {
		t.Fatalf("hash not stable: %q vs %q", key1, key2)
	}
	if key1 == HashPrompt("tenant-b", normalized) {
		t.Fatal("expected different namespaces to produce different keys")
	}

	const prefix = "exact:tenant-a:"
	if len(key1) <= len(prefix) || key1[:len(prefix)] != prefix {
		t.Fatalf("unexpected key format: %q", key1)
	}
}

func TestFloat32SliceToBytesRoundTrip(t *testing.T) {
	original := []float32{0, -1.5, 3.14159, 1e-6, -0.0}

	encoded := Float32SliceToBytes(original)
	if len(encoded) != len(original)*4 {
		t.Fatalf("encoded length = %d, want %d", len(encoded), len(original)*4)
	}

	decoded := make([]float32, len(original))
	for i := range original {
		bits := binary.LittleEndian.Uint32(encoded[i*4 : (i+1)*4])
		decoded[i] = math.Float32frombits(bits)
	}

	for i := range original {
		if decoded[i] != original[i] {
			t.Fatalf("index %d: decoded %v, want %v", i, decoded[i], original[i])
		}
	}
}

func TestFloat32SliceToBytesLittleEndianLayout(t *testing.T) {
	vec := []float32{1.0}
	got := Float32SliceToBytes(vec)
	want := make([]byte, 4)
	binary.LittleEndian.PutUint32(want, math.Float32bits(1.0))

	if string(got) != string(want) {
		t.Fatalf("little-endian encoding mismatch: got %v, want %v", got, want)
	}
}

func TestFloat32SliceToBytesEmpty(t *testing.T) {
	got := Float32SliceToBytes(nil)
	if len(got) != 0 {
		t.Fatalf("expected empty slice, got len %d", len(got))
	}
}
