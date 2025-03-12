#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <sys/stat.h>
#include "math_helper/math_helper.h"
#include "ransac_voting/ransac_voting.h"
#include "utils/utils.h"
using namespace nvinfer1;
// Logger for TensorRT
class Logger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity != Severity::kINFO) { // Suppress INFO messages
            std::cerr << "[TensorRT] " << msg << std::endl;
        }
    }
};
Logger gLogger;

// Helper function to load engine file
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
void allocateBuffers(const ICudaEngine* engine, std::vector<void*>& buffers, std::vector<size_t>& bufferSizes) {
    int numBindings = engine->getNbBindings();
    buffers.resize(numBindings);
    bufferSizes.resize(numBindings);

    for (int i = 0; i < numBindings; ++i) {
        Dims dims = engine->getBindingDimensions(i);
        size_t bindingSize = 1;
        for (int j = 0; j < dims.nbDims; ++j) {
            bindingSize *= dims.d[j];
        }
        bindingSize *= sizeof(float); // Assuming float32
        bufferSizes[i] = bindingSize;
        cudaMalloc(&buffers[i], bindingSize);
    }
}

/*
    int idx = c * seg_height * seg_width + h * seg_width + w;
    std::cout << "Value at (c=0, h=205, w=325): " << hostOutputSeg[idx] << std::endl;
    int idx_1 = 1 * seg_height * seg_width + h * seg_width + w;
    std::cout << "Value at (c=1, h=205, w=325): " << hostOutputSeg[idx_1] << std::endl;
*/
int main() {
    const std::string engineFilePath = "../model/crpo.engine";
    const std::string camera_path = "../data/camera.txt";
    const std::string kpt3d_path = "../data/kpt_3d.txt";
    const cv::Size inputSize(400, 400);
    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float std[3] = {0.229f, 0.224f, 0.225f};
    std::vector<float> bbox = {451.4214341990951, 12.33254212371736, 738.3436476749814, 355.95188691657444};
    // Load TensorRT engine
    std::vector<char> engineData = loadEngineFile(engineFilePath);
    IRuntime* runtime = createInferRuntime(gLogger);
    ICudaEngine* engine = runtime->deserializeCudaEngine(engineData.data(), engineData.size());
    IExecutionContext* context = engine->createExecutionContext();

    // Allocate buffers
    std::vector<void*> buffers;
    std::vector<size_t> bufferSizes;
    allocateBuffers(engine, buffers, bufferSizes);
    // TensorRT 模型的输入尺寸
    Dims inputDims = engine->getBindingDimensions(0);
        std::cout << "Input shape: (";
    for (int i = 0; i < inputDims.nbDims; ++i) {
        std::cout << inputDims.d[i] << (i < inputDims.nbDims - 1 ? ", " : "");
    }
    std::cout << ")" << std::endl;
    
    std::string imageFolder = "../data/crop/rgb";
    std::string imagePathTest = "../data/crop/rgb/0.jpg";
    int seg_channels = 2, seg_height = 400, seg_width = 400;
    int ver_channels = 18, ver_height = 400, ver_width = 400;
    std::vector<std::string> imagePaths = getAllImagePaths(imageFolder);
    // for(auto imagePath : imagePaths){
    //     std::cout << "Image: " << imagePath << std::endl;     
    //     std::filesystem::path path(imagePath);  // 使用 std::filesystem
    while(1){
        std::vector<float> hostInput(bufferSizes[0] / sizeof(float));
        // std::cout<< imagePath<< std::endl;
        preprocessImage(imagePathTest, hostInput.data(), inputSize, mean, std);
        cudaMemcpy(buffers[0], hostInput.data(), bufferSizes[0], cudaMemcpyHostToDevice);   
        auto start0 = std::chrono::high_resolution_clock::now();
        context->executeV2(buffers.data());
        auto end0 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> duration0 = end0 - start0;
        std::cout << "Inference time: " << duration0.count() << " ms" << std::endl;

        std::vector<float> hostOutputSeg(bufferSizes[1] / sizeof(float));
        std::vector<float> hostOutputVer(bufferSizes[2] / sizeof(float));
        cudaMemcpy(hostOutputSeg.data(), buffers[1], bufferSizes[1], cudaMemcpyDeviceToHost);
        cudaMemcpy(hostOutputVer.data(), buffers[2], bufferSizes[2], cudaMemcpyDeviceToHost);

        Dims segDims = engine->getBindingDimensions(1);
        Dims verDims = engine->getBindingDimensions(2);
        cv::Mat mask;
        std::vector<cv::Mat> vertex;
        convertTensorRTOutputToMat(hostOutputSeg, hostOutputVer, seg_channels, seg_height, seg_width,
                                ver_channels, ver_height, ver_width, mask, vertex); 
        // cv::imwrite("../mask_test.png", mask);
        std::vector<cv::Point> coords;
        cv::findNonZero(mask, coords);
        std::cout<< coords.size()<<std::endl;

        std::vector<cv::Point2f> output_points;
        auto start = std::chrono::high_resolution_clock::now();                              
        // ransacVotingV1(mask, vertex, output_points, 128, 0.999);
        // ransacVoting::ransacVotingV2(mask, vertex, output_points, 128, 0.999, 0.99, 20);    
        ransacVoting::ransacVotingV4(mask, vertex, output_points, 128, 0.999, 0.99, 20, 5, 30000);
        auto end = std::chrono::high_resolution_clock::now();
        std::cout<<" kpt_2d: \n"<< output_points<< std::endl;
        std::chrono::duration<float, std::milli> duration = end - start; 
        std::cout << "RANSAC V2 time: " << duration.count() << " ms" << std::endl;
        std::vector<cv::Point2f> uncrop_points;
        // transformPointsToOriginal(output_points, bbox, uncrop_points);   
        transformPointsToOriginalTest(output_points, bbox, 400.0, 400.0);
        std::cout<<" kpt_2d uncrop:\n"<< output_points<< std::endl;
        std::vector<cv::Point3f> points_3d = read3DPoints(kpt3d_path);
        std::cout << "3D points.\n " << points_3d <<  std::endl;
        cv::Mat camera_matrix = readCameraMatrix(camera_path);
        std::cout << "Camera Matrix:\n" << camera_matrix << std::endl;
        cv::Mat pose_pred = solvePnPWithPoseMatrix(points_3d, output_points, camera_matrix);
        std::cout << "Pose Matrix (3x4):\n" << pose_pred << std::endl;
        // std::string trt_base_name = path.stem().string(); // 提取基础文件名
        // std::string trt_pose_path = trt_base_name + ".jpg_trt.txt";    
        // saveMatToTxt(pose_pred, trt_pose_path);                                                                   
        // break;    
    }
 
    for (void* buffer : buffers) {
        cudaFree(buffer);
    }
    context->destroy();
    engine->destroy();
    runtime->destroy();

    return 0;
}
