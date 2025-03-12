#ifndef KERNELS_H
#define KERNELS_H

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

// 核函数声明，注意 extern "C"
#ifdef __cplusplus
extern "C" {
#endif


void generate_hypothesis_cuda(
    const float* direct,
    const float* coords,
    const int* idxs,
    float* hypo_pts,
    int h, int w, int tn, int hn,
    cudaStream_t stream=0
);

void launch_voting_cuda(
    float* d_coords,
    float* d_direct,
    float* d_hypo_pts,
    int* d_inlier_counts,
    bool* d_inliers,
    int h, int w, int tn, int hn,
    float inlier_thresh,
    cudaStream_t stream=0
);

#ifdef __cplusplus
}
#endif

#endif