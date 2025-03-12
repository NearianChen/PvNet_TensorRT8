#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>
#include <memory>
#include <chrono>
#include <NvInfer.h>
#include <NvUtils.h>
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include "cnpy/cnpy.h"
#include <variant>
#include "nlohmann/json.hpp"  // For JSON parsing, use nlohmann/json
#include "toml/toml.hpp"

using namespace std;
using namespace nvinfer1;
using json = nlohmann::json;

class Logger : public nvinfer1::ILogger
{
public:
    Logger(Severity severity = Severity::kINFO)
        : reportableSeverity(severity)
    {
    }

    void log(Severity severity, const char* msg) noexcept override
    {
        // suppress messages with severity enum value greater than the reportable
        if (severity > reportableSeverity)
            return;

        switch (severity)
        {
        case Severity::kINTERNAL_ERROR:  break;
        case Severity::kERROR: break;
        case Severity::kWARNING:  break;
        case Severity::kINFO:  break;
        default:break;
        }
        std::cerr << msg << std::endl;
    }

    Severity reportableSeverity;
};

class TimeProfiler : public nvinfer1::IProfiler {
public:
    void reportLayerTime(const char* layerName, float ms) noexcept override {
        std::cout << layerName << " took " << ms << " ms." << std::endl;
    }
};

class TRTWrapper {
public:
    TRTWrapper(const std::string& engine_path) {
        cudaError_t status = cudaStreamCreate(&stream_);
        if (status != cudaSuccess) {
            // Handle error
        }

        // Load the engine
        std::ifstream engine_file(engine_path, std::ios::binary);
        if (!engine_file) {
            throw std::runtime_error("Failed to open engine file.");
        }

        engine_file.seekg(0, engine_file.end);
        size_t engine_size = engine_file.tellg();
        engine_file.seekg(0, engine_file.beg);

        std::vector<char> engine_data(engine_size);
        engine_file.read(engine_data.data(), engine_size);

        static Logger gLogger;
        void* handle = dlopen("../../lib/libdeploy_tensorrt_ops.so", RTLD_LAZY);
        if (!handle) {
            fprintf(stderr, "%s\n", dlerror());
            exit(EXIT_FAILURE);
        }
        nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
        engine_ = runtime->deserializeCudaEngine(engine_data.data(), engine_size);
        context_ = engine_->createExecutionContext();
        inputs_names_.push_back("input");
        outputs_names_.push_back("seg");
        outputs_names_.push_back("vertex");

        // malloc input memory
        // Function to allocate GPU memory and copy data
        bindings_ = std::vector<void*>(engine_->getNbBindings(), nullptr);
        auto allocate = [](const std::vector<size_t> &shape, int byte) {
            void* gpuData;
            size_t size = std::accumulate(shape.begin(), shape.end(), static_cast<size_t>(1), std::multiplies<size_t>()) * byte;
            auto cudaStatus = cudaMalloc(&gpuData, size);
            if (cudaStatus != cudaSuccess) {
                std::cerr << "Error: cudaMalloc failed for output buffer" << std::endl;
                throw std::invalid_argument( "cudaMalloc" );
            }
            // cudaMemcpy(gpuData, arrayData.data.data(), size, cudaMemcpyHostToDevice);
            return gpuData;
        };

        // Allocate GPU memory and set bindings for each input and output
        // TO CHANGE
        bindings_[engine_->getBindingIndex("input")] = allocate(std::vector<size_t>{1, 3, 480, 640}, 4);
        bindings_[engine_->getBindingIndex("seg")] = allocate(std::vector<size_t>{1, 2, 480, 640}, 4);
        bindings_[engine_->getBindingIndex("vertex")] = allocate(std::vector<size_t>{1, 18, 480, 640}, 4);
        for (size_t i = 0; i < bindings_.size(); ++i) {
            if (!bindings_[i]) {
                std::cerr << "Error: Binding index " << i << " is not set correctly." << std::endl;
                // Handle error
            }
        }

        // Allocate CPU memory


    }

    std::vector<int32_t> forward(const FloatArray& img, bool debug=false) {
        // clone data
        cudaMemcpy(bindings_[engine_->getBindingIndex("input")], img.data.data(), img.data.size() * sizeof(float), cudaMemcpyHostToDevice);

        // Execute the context
        TimeProfiler profiler;
        context_->setProfiler(&profiler);
        context_->enqueueV2(bindings_.data(), stream_, nullptr); // Assuming 'stream' is your CUDA stream

        // // Free GPU memory after use
        // for (void* buf : bindings) {
        //     cudaFree(buf);
        // }
        
        // TO CHANGE
        int totalSizeSeg = 2 * 480 *640;
        int totalSizeVer = 18 * 480 * 640;
        using _dtype = int32_t;
        _dtype* seg_cpu = static_cast<_dtype*>(malloc(totalSizeSeg * sizeof(_dtype)));
        std::vector<_dtype> seg_output(seg_cpu, seg_cpu + totalSizeSeg);
        auto address_seg = bindings_[engine_->getBindingIndex("seg")];
        cudaMemcpy(seg_output.data(), address_seg, totalSizeSeg * sizeof(_dtype), cudaMemcpyDeviceToHost);

        _dtype* ver_cpu = static_cast<_dtype*>(malloc(totalSizeVer * sizeof(_dtype)));
        std::vector<_dtype> ver_output(ver_cpu, ver_cpu + totalSizeVer);
        auto address_ver = bindings_[engine_->getBindingIndex("ver")];
        cudaMemcpy(ver_output.data(), address_ver, totalSizeVer * sizeof(_dtype), cudaMemcpyDeviceToHost);        
        return occ_segment;
    }

private:
    nvinfer1::ICudaEngine* engine_;
    nvinfer1::IExecutionContext* context_;
    std::vector<int> inputs_ids_;
    std::vector<int> output_ids_;
    std::vector<std::string> inputs_names_;
    std::vector<std::string> outputs_names_;
    // std::vector<Tensor> inputs_tensors_;
    // std::vector<Tensor> output_tensors_;
    // Device device_;
    // Stream stream_;
    // Event event_;
    cudaStream_t stream_;
    std::vector<void*> bindings_ ;
    cudaError_t cudaStatus_;

};

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " --model-name <model_name>\n";
        return 1;
    }
    std::string model_name;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--model-name" && i + 1 < argc) {
            model_name = argv[++i];
        }
    }
    if (model_name.empty()) {
        std::cerr << "Model name is required.\n";
        std::cerr << "Usage: " << argv[0] << " --model-name <model_name>\n";
        return 1;
    }

    std::string config_path = "../config.toml";
    auto tbl = toml::parse_file(config_path);
    std::string_view data_path_view = tbl["model"]["data_path"].value_or(""sv);
    std::string data_path(data_path_view);
    std::cout << data_path << std::endl;
    
    // load input
    // Parse the JSON metadata file
    std::ifstream ifs(data_path + '/' + model_name + '/' + "data_metadata.json");
    json metadata = json::parse(ifs);

    std::unordered_map<std::string, ArrayBase> inputs;

    // const char* inputs_name_[6] = {"img", "ranks_bev", "ranks_depth", "ranks_feat", "interval_starts", "interval_lengths"};

    // Load the engine
    TRTWrapper model(data_path + '/' + model_name + '/' + "workdirbevdet_int8_fuse.engine");
    int num=10;
    for(auto i=0;i<=num;i++){
        std::cout << "iter " << i << "#######################" << std::endl;
        // Load inputs and run the model
        auto occ_segment = model.forward(img, ranks_bev, ranks_depth, ranks_feat, interval_starts, interval_lengths, true);
        if(i==num){
            saveArray(data_path + '/' + model_name + '/' + "output_trt.npy", occ_segment, std::vector<size_t>{400, 400, 16}); // TO CHANGE
        }
    }

    std::cout << "Success" << std::endl;
    return 0;
}
