#include <metal_stdlib>
using namespace metal;

kernel void knn_distance(device const float* train_data [[ buffer(0) ]],
                         device const float* test_point [[ buffer(1) ]],
                         device float* output_distances [[ buffer(2) ]],
                         uint id [[ thread_position_in_grid ]]) {
    const uint dims = 2; // Change if needed
    float dist = 0.0;
    for (uint d = 0; d < dims; ++d) {
        float diff = train_data[id * dims + d] - test_point[d];
        dist += diff * diff;
    }
    output_distances[id] = dist;
}