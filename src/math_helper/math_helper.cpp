#include "math_helper.h"

void convertTensorRTOutputToMat(
    const std::vector<float>& seg_pred, // TensorRT output for seg_pred
    const std::vector<float>& ver_pred, // TensorRT output for ver_pred
    int seg_channels, int seg_height, int seg_width,  // seg_pred dimensions
    int ver_channels, int ver_height, int ver_width,  // ver_pred dimensions
    cv::Mat& mask,              // Output mask as cv::Mat
    std::vector<cv::Mat>& vertex // Output vertex as a vector of cv::Mat
) {
    // Ensure dimensions match
    if (seg_height != ver_height || seg_width != ver_width) {
        throw std::runtime_error("seg_pred and ver_pred dimensions do not match!");
    }

    // Step 1: Generate mask
    mask = cv::Mat(seg_height, seg_width, CV_8U);
    for (int i = 0; i < seg_height; ++i) {
        for (int j = 0; j < seg_width; ++j) {
            int idx_0 = 0 * seg_height * seg_width + i * seg_width + j; // Channel 0 index
            int idx_1 = 1 * seg_height * seg_width + i * seg_width + j; // Channel 1 index
            mask.at<uchar>(i, j) = (seg_pred[idx_1] > seg_pred[idx_0]) ? 1 : 0;
        }
    }

    // Convert ver_pred to vertex Mats
    int vn = ver_channels / 2; // Assuming ver_channels = VN * 2
    vertex.resize(vn);
    for (int k = 0; k < vn; ++k) {
        vertex[k] = cv::Mat(ver_height, ver_width, CV_32FC2);
        for (int i = 0; i < ver_height; ++i) {
            for (int j = 0; j < ver_width; ++j) {
                int idx_x = (k * 2) * ver_height * ver_width + i * ver_width + j; // x offset
                int idx_y = (k * 2 + 1) * ver_height * ver_width + i * ver_width + j; // y offset
                vertex[k].at<cv::Vec2f>(i, j) = cv::Vec2f(ver_pred[idx_x], ver_pred[idx_y]);
            }
        }
    }

}

/*
    std::vector<size_t> shape = {1, 2, 480, 640};
    auto reshaped_array = reshapeTRT2Matrix(hostOutputSeg, shape);   
    int c = 0, h = 205, w = 325;
    std::cout << "Reshaped value at [0][0][205][325]: " 
            << reshaped_array[0][0][h][w] << std::endl;            
    std::cout << "Reshaped value at [0][1][205][325]: " 
            << reshaped_array[0][1][h][w] << std::endl;
*/
template <typename T>
std::vector<std::vector<std::vector<std::vector<T>>>> reshapeTRT2Matrix(
    const std::vector<T>& input, 
    const std::vector<size_t>& shape) 
{
    // 计算总元素数量
    size_t total_size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
    
    if (input.size() != total_size) {
        throw std::invalid_argument("Input size does not match the target shape.");
    }

    // 初始化多维数组
    std::vector<std::vector<std::vector<std::vector<T>>>> result(
        shape[0], 
        std::vector<std::vector<std::vector<T>>>(shape[1], 
        std::vector<std::vector<T>>(shape[2], 
        std::vector<T>(shape[3]))));

    // 填充多维数组
    size_t index = 0;
    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            for (size_t k = 0; k < shape[2]; ++k) {
                for (size_t l = 0; l < shape[3]; ++l) {
                    result[i][j][k][l] = input[index++];
                }
            }
        }
    }

    return result;
}

void preprocessImage(const std::string& imagePath, float* hostInput, const cv::Size& inputSize, const float mean[3], const float std[3]) {
    cv::Mat img = cv::imread(imagePath, cv::IMREAD_COLOR);
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    if (img.empty()) {
        throw std::runtime_error("Failed to read image: " + imagePath);
    }
    cv::resize(img, img, inputSize, 0, 0, cv::INTER_LINEAR);
    img.convertTo(img, CV_32FC3, 1.0 / 255.0);

    std::vector<cv::Mat> channels(3);
    cv::split(img, channels);

    for (int c = 0; c < 3; ++c) {
        channels[c] = (channels[c] - mean[c]) / std[c];
    }
    size_t idx = 0;
    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < img.rows; ++h) {
            for (int w = 0; w < img.cols; ++w) {
                hostInput[idx++] = channels[c].at<float>(h, w);
            }
        }
    }
}

cv::Mat solvePnPWithPoseMatrix(const std::vector<cv::Point3f>& points_3d,
                               const std::vector<cv::Point2f>& points_2d,
                               const cv::Mat& camera_matrix,
                               int method) {
    if (points_3d.size() != points_2d.size()) {
        throw std::runtime_error("3D points and 2D points must have the same size");
    }

    // 无畸变，初始化畸变系数
    cv::Mat dist_coeffs = cv::Mat::zeros(8, 1, CV_64F);

    cv::Mat rvec, tvec;
    bool success = cv::solvePnP(points_3d, points_2d, camera_matrix, dist_coeffs, rvec, tvec, false, cv::SOLVEPNP_ITERATIVE);

    if (!success) {
        throw std::runtime_error("solvePnP failed to find a valid pose.");
    }
    // std::cout<< rvec << "\ntvec\n" << tvec << std::endl;
    cv::Mat R;
    cv::Rodrigues(rvec, R);

    // std::cout << "Rotation Vector (rvec):\n" << rvec << std::endl;
    // std::cout << "Translation Vector (tvec):\n" << tvec << std::endl;
    // std::cout << "R Matrix:\n" << R << std::endl;

    cv::Mat pose = cv::Mat::eye(3, 4, CV_64F);
    R.copyTo(pose(cv::Rect(0, 0, 3, 3)));  
    tvec.copyTo(pose(cv::Rect(3, 0, 1, 3))); 
    return pose;
}

void transformPointsToOriginal(
    const std::vector<cv::Point2f>& input_points,
    const std::vector<float>& bbox, // bbox: [x_min, y_min, x_max, y_max]
    std::vector<cv::Point2f>& output_points) {

    // 检查 bbox 的尺寸是否正确
    if (bbox.size() != 4) {
        throw std::invalid_argument("Bounding box should have exactly 4 elements: [x_min, y_min, x_max, y_max]");
    }

    float x_min = bbox[0];
    float y_min = bbox[1];
    float x_max = bbox[2];
    float y_max = bbox[3];

    // 计算裁剪框的宽度和高度
    float bbox_width = x_max - x_min;
    float bbox_height = y_max - y_min;

    if (bbox_width <= 0 || bbox_height <= 0) {
        throw std::invalid_argument("Invalid bounding box dimensions: width and height must be positive.");
    }

    // 将 400×400 的坐标映射回原图的坐标
    output_points.clear();
    for (const auto& pt : input_points) {
        float original_x = x_min + (pt.x / 400.0f) * bbox_width;
        float original_y = y_min + (pt.y / 400.0f) * bbox_height;
        output_points.emplace_back(original_x, original_y);
    }
}

void transformPointsToOriginalTest(
    std::vector<cv::Point2f>& points, 
    const std::vector<float>& bbox,   
    float w,                          
    float h                           
) {
    // 检查 bbox 的尺寸是否正确
    if (bbox.size() != 4) {
        throw std::invalid_argument("Bounding box should have exactly 4 elements: [x_min, y_min, x_max, y_max]");
    }

    float x_min = bbox[0];
    float y_min = bbox[1];
    float x_max = bbox[2];
    float y_max = bbox[3];

    // 检查裁剪框的大小是否有效
    if (w <= 0 || h <= 0) {
        throw std::invalid_argument("Width and height of the cropped region must be positive.");
    }

    // 计算裁剪框的原图尺寸
    float bbox_width = x_max - x_min;
    float bbox_height = y_max - y_min;

    if (bbox_width <= 0 || bbox_height <= 0) {
        throw std::invalid_argument("Invalid bounding box dimensions: width and height must be positive.");
    }

    // 将裁剪框坐标映射到原图坐标
    for (auto& pt : points) {
        pt.x = x_min + (pt.x / w) * bbox_width;
        pt.y = y_min + (pt.y / h) * bbox_height;
    }
}
