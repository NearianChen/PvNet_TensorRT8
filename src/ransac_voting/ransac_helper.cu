#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include "ransac_helper.h"
#include <math.h>

#define cudaCheckError() { \
    cudaError_t e = cudaGetLastError(); \
    if (e != cudaSuccess) { \
        printf("CUDA error: %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(EXIT_FAILURE); \
    } \
}
__global__ void generate_hypothesis_kernel(
    const float* direct,     // [h][w][2]
    const float* coords,     // [tn][2]
    const int* idxs,         // [hn][2]
    float* hypo_pts,         // [hn][2]
    int h, int w, int tn, int hn
) {
    int hi = threadIdx.x + blockIdx.x * blockDim.x;
    if (hi >= hn) return;

    int t0 = idxs[hi * 2];
    int t1 = idxs[hi * 2 + 1];

    float cx0 = coords[t0 * 2];
    float cy0 = coords[t0 * 2 + 1];
    float cx1 = coords[t1 * 2];
    float cy1 = coords[t1 * 2 + 1];

    // 将坐标转换为图像像素位置
    int ix0 = static_cast<int>(cx0);
    int iy0 = static_cast<int>(cy0);
    int ix1 = static_cast<int>(cx1);
    int iy1 = static_cast<int>(cy1);

    // 边界检查
    if (ix0 < 0 || ix0 >= w || iy0 < 0 || iy0 >= h ||
        ix1 < 0 || ix1 >= w || iy1 < 0 || iy1 >= h) {
        hypo_pts[hi * 2] = -1;
        hypo_pts[hi * 2 + 1] = -1;
        return;
    }

    // 读取偏移量
    float dx0 = direct[(iy0 * w + ix0) * 2];
    float dy0 = direct[(iy0 * w + ix0) * 2 + 1];
    float dx1 = direct[(iy1 * w + ix1) * 2];
    float dy1 = direct[(iy1 * w + ix1) * 2 + 1];

    // 计算法向量 (dy, -dx)
    float nx0 = dy0;
    float ny0 = -dx0;
    float nx1 = dy1;
    float ny1 = -dx1;

    // 修正分母：原式符号错误
    float denominator = nx0 * ny1 - nx1 * ny0; // 正确行列式

    if (fabs(denominator) < 1e-6) {
        hypo_pts[hi * 2] = -1;
        hypo_pts[hi * 2 + 1] = -1;
        return;
    }

    float C0 = nx0 * cx0 + ny0 * cy0;
    float C1 = nx1 * cx1 + ny1 * cy1;

    float numerator_x = C0 * ny1 - C1 * ny0;
    float numerator_y = nx0 * C1 - nx1 * C0; // 注意符号调整

    hypo_pts[hi * 2] = numerator_x / denominator;
    hypo_pts[hi * 2 + 1] = numerator_y / denominator;
}

extern "C" void generate_hypothesis_cuda(
    const float* direct,
    const float* coords,
    const int* idxs,
    float* hypo_pts,
    int h, int w, int tn, int hn,
    cudaStream_t stream
) {
    // 检查输入参数有效性
    if (hn <= 0 || h <= 0 || w <= 0) {
        std::cerr << "Invalid parameters to CUDA kernel: "
                  << "h=" << h << " w=" << w << " hn=" << hn << std::endl;
        exit(EXIT_FAILURE);
    }

    dim3 block(256);
    dim3 grid((hn + block.x - 1) / block.x);

    // 检查网格有效性
    if (grid.x == 0) {
        std::cerr << "Invalid grid size calculation: hn=" << hn 
                  << " block.x=" << block.x << std::endl;
        exit(EXIT_FAILURE);
    }

    generate_hypothesis_kernel<<<grid, block, 0, stream>>>(
        direct, coords, idxs, hypo_pts, h, w, tn, hn
    );
    cudaCheckError();
}
__global__ void voting_kernel(
    float* coords,      // [tn][2]
    float* direct,      // [h][w][2]
    float* hypo_pts,    // [hn][2]
    int* inlier_counts, // [hn]
    bool* inliers,      // [hn][tn]
    int h, int w, int tn, int hn,
    float inlier_thresh
) {
    // 修改索引计算方式
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = gridDim.x * blockDim.x;
    
    // 每个线程处理多个元素
    for (int idx = tid; idx < hn*tn; idx += total_threads) {
        const int hi = idx / tn; // 假设点索引
        const int ti = idx % tn; // 数据点索引
        if (hi >= hn || ti >= tn) return;

        // 获取数据点坐标（浮点，无需量化）
        float cx = coords[ti * 2];
        float cy = coords[ti * 2 + 1];

        // 直接根据浮点坐标索引方向向量（需确保direct结构正确）
        int ix = static_cast<int>(cx);
        int iy = static_cast<int>(cy);
        if (ix < 0 || ix >= w || iy < 0 || iy >= h) return;

        // 读取方向向量（此处假设direct为[h][w][2]）
        float nx = direct[(iy * w + ix) * 2];      // 方向向量x分量
        float ny = direct[(iy * w + ix) * 2 + 1];  // 方向向量y分量

        // 获取假设点坐标
        float hypo_x = hypo_pts[hi * 2];
        float hypo_y = hypo_pts[hi * 2 + 1];

        // 计算当前点到假设点的向量
        float dx = hypo_x - cx;
        float dy = hypo_y - cy;

        // 计算余弦相似度
        float norm_n = sqrt(nx * nx + ny * ny);
        float norm_d = sqrt(dx * dx + dy * dy);
        if (norm_n < 1e-6 || norm_d < 1e-6) return;

        float angle_dist = (dx * nx + dy * ny) / (norm_n * norm_d);
        bool is_inlier = (angle_dist > inlier_thresh);

        // 原子操作更新计数
        if (is_inlier) {
            atomicAdd(&inlier_counts[hi], 1);
        }

        // 记录内点标记
        // inliers[hi * tn + ti] = is_inlier;
        // 修改inlier存储方式
        inliers[hi*tn + ti] = is_inlier;
        // if (is_inlier) atomicAdd(&inlier_counts[hi], 1); 
    }       
}

void launch_voting_cuda(
    float* d_coords,
    float* d_direct,
    float* d_hypo_pts,
    int* d_inlier_counts,
    bool* d_inliers,
    int h, int w, int tn, int hn,
    float inlier_thresh, // 直接传递PyTorch的阈值（如0.999）
    cudaStream_t stream
) {
    // 更高效的1D网格划分
    int total = hn * tn;
    int block = 256;
    int grid = (total + block - 1) / block;
    
    voting_kernel<<<grid, block, 0, stream>>>(
        d_coords, d_direct, d_hypo_pts,
        d_inlier_counts, d_inliers,
        h, w, tn, hn, inlier_thresh
    );
    cudaCheckError();
}    