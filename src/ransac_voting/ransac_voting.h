#include <opencv2/opencv.hpp>
#include <vector>
#include <numeric>
#include <random>
#include <cmath>

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <algorithm>
#include <iostream>

namespace ransacVoting{
    void ransacVotingV4(
        const cv::Mat& mask,
        const std::vector<cv::Mat>& vertex,
        std::vector<cv::Point2f>& output_points,
        int round_hyp_num,
        float inlier_thresh,
        float confidence,
        int max_iter,
        int min_num = 5,
        int max_num = 30000
    ); 
    void ransacVotingCUDAV2(
        cv::Mat& mask, std::vector<cv::Mat>& vertex, 
        std::vector<cv::Point2f>& output_points, 
        int round_hyp_num, float inlier_thresh, float confidence, 
        int max_iter, int min_num, int max_num
    );
}