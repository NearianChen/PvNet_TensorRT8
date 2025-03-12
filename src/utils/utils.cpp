#include <vector>
#include <string>
#include "utils.h"
#include <filesystem>  // need C++ 17

/*
    std::string imageFolder = "/home/nvidia/work/pvnet/data/demo_cat";
    std::vector<std::string> imagePaths = getAllImagePaths(imageFolder);
    for(auto imagePath : imagePaths){
        coding...
    }
*/
std::vector<std::string> getAllImagePaths(const std::string& directory) {
    std::vector<std::string> imagePaths;
    DIR* dir = opendir(directory.c_str());
    if (dir == nullptr) {
        std::cerr << "Error: Unable to open directory " << directory << std::endl;
        return imagePaths;
    }

    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string fileName = entry->d_name;
        // Check if it's a file and ends with ".jpg"
        if (fileName.size() > 4 && fileName.substr(fileName.size() - 4) == ".jpg") {
            imagePaths.push_back(directory + "/" + fileName);
        }
    }
    closedir(dir);
    return imagePaths;
}


void savePoseMatrixToNpy(const cv::Mat& poseMatrix, const std::string& outputPath) {
    // 确保输入是 CV_32F 或 CV_64F 类型
    CV_Assert(poseMatrix.type() == CV_32F || poseMatrix.type() == CV_64F);

    // 将 cv::Mat 转换为 std::vector<float> 或 std::vector<double>
    if (poseMatrix.type() == CV_32F) {
        std::vector<float> data(poseMatrix.begin<float>(), poseMatrix.end<float>());
        cnpy::npy_save(outputPath, &data[0], {poseMatrix.rows, poseMatrix.cols}, "w");
    } else if (poseMatrix.type() == CV_64F) {
        std::vector<double> data(poseMatrix.begin<double>(), poseMatrix.end<double>());
        cnpy::npy_save(outputPath, &data[0], {poseMatrix.rows, poseMatrix.cols}, "w");
    }
}

/*
        save TensorRT input or output to .bin file, 
    then use pytools transform .bin-->.hex  to check.
        how to use ?
    saveToFile(hostInput.data(), bufferSizes[0] / sizeof(float), "cpp_input.bin");
*/

void saveToFile(const float* data, size_t size, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    file.write(reinterpret_cast<const char*>(data), size * sizeof(float));
    file.close();
}

/*
    std::vector<size_t> segShape = {1, 2, 480, 640};
    std::vector<size_t> verShape = {1, 18, 480, 640};
    std::filesystem::path path(imagePath);  // 使用 std::filesystem
    std::string baseName = path.stem().string(); // 提取基础文件名
    std::string segOutputFile = baseName + ".jpg_seg.npy";  // 拼接 _seg.npy 后缀
    std::string verOutputFile = baseName + ".jpg_vertex.npy";  // 拼接 _ver.npy 后缀
    saveToNpy(segOutputFile , hostOutputSeg, segShape);
    saveToNpy(verOutputFile, hostOutputVer, verShape);
*/
void saveToNpy(const std::string& filename, const std::vector<float>& data, const std::vector<size_t>& shape) {
    cnpy::npy_save(filename, data.data(), shape, "w");
}

// 定义函数来保存 cv::Mat 为 TXT 文件
void saveMatToTxt(const cv::Mat& mat, const std::string& filename) {
    std::ofstream file(filename);
    if (file.is_open()) {
        file << std::fixed << std::setprecision(10); // 设置浮点数精度
        for (int i = 0; i < mat.rows; ++i) {
            for (int j = 0; j < mat.cols; ++j) {
                file << mat.at<double>(i, j);
                if (j < mat.cols - 1) {
                    file << ", ";
                }
            }
            file << std::endl;
        }
        file.close();
    } else {
        std::cerr << "Unable to open file " << filename << std::endl;
    }
}

// 读取 3D 点文件
std::vector<cv::Point3f> read3DPoints(const std::string& filename) {
    std::vector<cv::Point3f> points;
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        float x, y, z;
        if (iss >> x >> y >> z) {
            points.emplace_back(x, y, z);
        }
    }
    file.close();
    return points;
}

cv::Mat readCameraMatrix(const std::string& filename) {
    cv::Mat camera_matrix(3, 3, CV_64F);
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            if (!(file >> camera_matrix.at<double>(i, j))) {
                throw std::runtime_error("Error reading camera matrix from file: " + filename);
            }
        }
    }
    file.close();
    return camera_matrix;
}
