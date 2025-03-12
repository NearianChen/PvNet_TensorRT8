#include <vector>
#include <string>
#include <dirent.h>
#include <fstream>
#include "cnpy.h"
#include <opencv2/opencv.hpp>

#include <filesystem> // C++ 17

std::vector<std::string> getAllImagePaths(const std::string& directory);
void savePoseMatrixToNpy(const cv::Mat& poseMatrix, const std::string& outputPath);
void saveToFile(const float* data, size_t size, const std::string& filename);
void saveToNpy(const std::string& filename, const std::vector<float>& data, const std::vector<size_t>& shape);
void saveMatToTxt(const cv::Mat& mat, const std::string& filename);

std::vector<cv::Point3f> read3DPoints(const std::string& filename);
cv::Mat readCameraMatrix(const std::string& filename);
