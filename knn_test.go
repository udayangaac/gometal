package gometal_test

import (
	"math"
	"testing"

	"github.com/udayangaac/gometal"
)

func TestMetalDistanceCalculator_Run(t *testing.T) {
	calculator := &gometal.MetalDistanceCalculator{}

	train := []float32{
		1.0, 2.0,
		2.0, 3.0,
		3.0, 3.0,
	}
	test := []float32{2.0, 2.0}
	dims := 2
	trainLen := 3

	distances, err := calculator.Run(train, test, trainLen, dims)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	expected := []float32{
		square(1.0-2.0) + square(2.0-2.0), // 1.0
		square(2.0-2.0) + square(3.0-2.0), // 1.0
		square(3.0-2.0) + square(3.0-2.0), // 2.0
	}

	if len(distances) != len(expected) {
		t.Fatalf("Expected %d distances, got %d", len(expected), len(distances))
	}

	for i := range expected {
		if math.Abs(float64(distances[i]-expected[i])) > 1e-5 {
			t.Errorf("Distance mismatch at %d: got %f, want %f", i, distances[i], expected[i])
		}
	}
}

func square(x float32) float32 {
	return x * x
}
