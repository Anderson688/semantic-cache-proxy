package cache

import (
	"encoding/binary"
	"math"
)

// Float32SliceToBytes encodes a vector as little-endian FLOAT32 bytes for RediSearch HNSW.
func Float32SliceToBytes(vec []float32) []byte {
	out := make([]byte, len(vec)*4)
	for i, v := range vec {
		binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(v))
	}
	return out
}