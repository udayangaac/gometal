package gometal

/*
#cgo CXXFLAGS: -std=c++17 -x objective-c++
#cgo LDFLAGS: -lstdc++ -framework Metal -framework Foundation
#include "metal_bridge.h"
*/
import "C"
import (
	"unsafe"
)

type DistanceCalculator interface {
	Run(trainData []float32, testPoint []float32, trainLen, dims int) ([]float32, error)
}

type MetalDistanceCalculator struct{}

func (m *MetalDistanceCalculator) Run(trainData []float32, testPoint []float32, trainLen, dims int) ([]float32, error) {
	output := make([]float32, trainLen)

	C.run_knn_distance(
		(*C.float)(unsafe.Pointer(&trainData[0])),
		(*C.float)(unsafe.Pointer(&testPoint[0])),
		(*C.float)(unsafe.Pointer(&output[0])),
		C.int(trainLen),
		C.int(dims),
	)

	return output, nil
}
