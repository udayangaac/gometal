#ifndef METAL_BRIDGE_H
#define METAL_BRIDGE_H

#ifdef __cplusplus
extern "C"
{
#endif

   int run_knn_distance(const float *train_data, const float *test_point, float *output_distances, int train_len, int dims);

#ifdef __cplusplus
}
#endif

#endif // METAL_BRIDGE_H