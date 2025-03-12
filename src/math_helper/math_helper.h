#include <vector>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <sys/stat.h>
#include <stdexcept>
#include <numeric>
#include <opencv2/opencv.hpp>

void convertTensorRTOutputToMat(
    const std::vector<float>& seg_pred, // TensorRT output for seg_pred
    const std::vector<float>& ver_pred, // TensorRT output for ver_pred
    int seg_channels, int seg_height, int seg_width,  // seg_pred dimensions
    int ver_channels, int ver_height, int ver_width,  // ver_pred dimensions
    cv::Mat& mask,              // Output mask as cv::Mat
    std::vector<cv::Mat>& vertex // Output vertex as a vector of cv::Mat
);


template <typename T>
std::vector<std::vector<std::vector<std::vector<T>>>> reshape(
    const std::vector<T>& input, 
    const std::vector<size_t>& shape);

void preprocessImage(const std::string& imagePath, 
    float* hostInput, const cv::Size& inputSize, const float mean[3], const float std[3]);

cv::Mat solvePnPWithPoseMatrix(
    const std::vector<cv::Point3f>& points_3d, 
    const std::vector<cv::Point2f>& points_2d, 
    const cv::Mat& camera_matrix, 
    int method = cv::SOLVEPNP_ITERATIVE
);

void transformPointsToOriginal(
    const std::vector<cv::Point2f>& input_points,
    const std::vector<float>& bbox, // bbox: [x_min, y_min, x_max, y_max]
    std::vector<cv::Point2f>& output_points);
void transformPointsToOriginalTest(
    std::vector<cv::Point2f>& points, 
    const std::vector<float>& bbox,   
    float w,                          
    float h                           
);