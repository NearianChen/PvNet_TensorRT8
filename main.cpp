#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include "math_helper/math_helper.h"
#include "ransac_voting/ransac_voting.h"
#include "utils/utils.h"
#include <Eigen/Dense>
using namespace nvinfer1;

class Logger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity != Severity::kINFO) {
            std::cerr << "[TensorRT] " << msg << std::endl;
        }
    }
};

class PoseEstimator {
private:
    Logger gLogger;
    IRuntime* runtime = nullptr;
    ICudaEngine* engine = nullptr;
    IExecutionContext* context = nullptr;
    std::vector<void*> buffers;
    std::vector<size_t> bufferSizes;
    cv::Size inputSize;
    float mean[3];
    float std[3];
    int height_;
    int width_;
    bool is_crop_;
    // Load TensorRT engine from file
    std::vector<char> loadEngineFile(const std::string& engineFilePath) {
        std::ifstream file(engineFilePath, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Failed to open engine file: " + engineFilePath);
        }
        file.seekg(0, std::ios::end);
        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);
        std::vector<char> buffer(size);
        file.read(buffer.data(), size);
        return buffer;
    }

    // Allocate buffers for TensorRT inputs and outputs
    void allocateBuffers() {
        int numBindings = engine->getNbBindings();
        buffers.resize(numBindings);
        bufferSizes.resize(numBindings);

        for (int i = 0; i < numBindings; ++i) {
            Dims dims = engine->getBindingDimensions(i);
            size_t bindingSize = 1;
            for (int j = 0; j < dims.nbDims; ++j) {
                bindingSize *= dims.d[j];
            }
            bindingSize *= sizeof(float);
            bufferSizes[i] = bindingSize;
            cudaMalloc(&buffers[i], bindingSize);
        }
    }

public:
    PoseEstimator(const std::string& engineFilePath, const cv::Size& inputSize_, const float mean_[3], const float std_[3], const bool is_crop)
        : inputSize(inputSize_) {
        std::copy(mean_, mean_ + 3, mean);
        std::copy(std_, std_ + 3, std);
        height_ = inputSize.height;
        width_ = inputSize.width;
        is_crop_ = is_crop;
        height_ = inputSize.height;
        width_ = inputSize.width;
        is_crop_ = is_crop;
        // Load engine and create execution context
        std::vector<char> engineData = loadEngineFile(engineFilePath);
        runtime = createInferRuntime(gLogger);
        engine = runtime->deserializeCudaEngine(engineData.data(), engineData.size());
        context = engine->createExecutionContext();

        // Allocate buffers
        allocateBuffers();
    }

    ~PoseEstimator() {
        for (void* buffer : buffers) {
            cudaFree(buffer);
        }
        if (context) context->destroy();
        if (engine) engine->destroy();
        if (runtime) runtime->destroy();
    }

    cv::Mat processImageAndEstimatePose(const std::string& imagePath, const std::vector<float>& bbox,
                                        const std::string& kpt3dPath, const std::string& cameraPath) {
        // Preprocess input image
        std::vector<float> hostInput(bufferSizes[0] / sizeof(float));

        preprocessImage(imagePath, hostInput.data(), inputSize, mean, std);
        cudaMemcpy(buffers[0], hostInput.data(), bufferSizes[0], cudaMemcpyHostToDevice);

        // Run inference
        context->executeV2(buffers.data());

        // Copy output data to host
        std::vector<float> hostOutputSeg(bufferSizes[1] / sizeof(float));
        std::vector<float> hostOutputVer(bufferSizes[2] / sizeof(float));
        cudaMemcpy(hostOutputSeg.data(), buffers[1], bufferSizes[1], cudaMemcpyDeviceToHost);
        cudaMemcpy(hostOutputVer.data(), buffers[2], bufferSizes[2], cudaMemcpyDeviceToHost);

        // Convert output to cv::Mat
        cv::Mat mask;
        std::vector<cv::Mat> vertex;
        auto start_convert = std::chrono::high_resolution_clock::now();
        convertTensorRTOutputToMat(hostOutputSeg, hostOutputVer, 2, height_, width_, 18, height_, width_, mask, vertex);
        auto end_convert = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> duration_convert = end_convert - start_convert;
        std::cout<<"Convert time:" <<duration_convert.count() << " ms" <<std::endl;

        auto start_ransac = std::chrono::high_resolution_clock::now();
        // ransacVotingV5(outputPoints, 128, 0.999, 0.99, 20, 5, 30000, state);

        // Perform RANSAC voting
        std::vector<cv::Point2f> outputPoints;
        ransacVoting::ransacVotingCUDAV2(mask, vertex, outputPoints, 128, 0.999, 0.99, 20, 5, 100);
        auto end_ransac = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> duration_ransac = end_ransac - start_ransac;
        std::cout<<"RANSAC time:" <<duration_ransac.count() << " ms" <<std::endl;
        // std::cout<< "kpt2d nocrop: \n" << outputPoints<<std::endl;
        if(is_crop_){
            transformPointsToOriginalTest(outputPoints, bbox, width_, height_);
        }
        // std::cout<< "kpt2d: \n" << outputPoints<<std::endl;
        // Load 3D points and camera matrix
        std::vector<cv::Point3f> points3D = read3DPoints(kpt3dPath);
        cv::Mat cameraMatrix = readCameraMatrix(cameraPath);

        // Solve PnP and return pose matrix
        return solvePnPWithPoseMatrix(points3D, outputPoints, cameraMatrix);
    }
};

int main() {
    /*
        Params:
        imageFolder: Path to the folder containing input images
        engineFilePath: Path to the TensorRT engine file
        cameraPath: Path to the camera file
        kpt3dPath: Path to the 3D keypoint file
    */
    const std::string imageFolder = "../data/crop/rgb";
    const std::string engineFilePath = "../model/crop/old_crop_399.engine";
    const std::string cameraPath = "../data/crop/camera.txt";
    const std::string kpt3dPath = "../data/crop/kpt_3d.txt";
    int width = 1280;
    int height = 720;
    bool is_crop = false;

    const cv::Size inputSize(width, height); //// w h
    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float std[3] = {0.229f, 0.224f, 0.225f};
    std::vector<std::string> imagePaths = getAllImagePaths(imageFolder);
    PoseEstimator estimator(engineFilePath, inputSize, mean, std, is_crop);
    /*
        example:
        imagePath = imageFolder + "/" + "123.jpg"
        std::filesystem::path(imagePath).stem().string() = 123
    */
    for(auto imagePath : imagePaths){
        try {
            std::cout << "Image:" << std::filesystem::path(imagePath).stem().string() << std::endl;
            auto start = std::chrono::high_resolution_clock::now();
            cv::Mat poseMatrix = estimator.processImageAndEstimatePose(imagePath, bbox, kpt3dPath, cameraPath);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<float, std::milli> duration = end - start;
            std::cout<<"All process time:" <<duration.count() << " ms" <<std::endl;
            std::cout << "Pose Matrix (3x4):\n" << poseMatrix << std::endl;
            /*
                save result RT Matrix to .npy file, need thirdparty library cnpy, github: https://github.com/rogersce/cnpy.git
                use like:
            std::string outputFilePath = "RT_result.npy";
            savePoseMatrixToNpy(poseMatrix, outputFilePath); // function path : src/utils/utils.cpp
            */
            // break;
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            return -1;
        }
    }
    return 0;
}
