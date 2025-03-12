cmake_minimum_required(VERSION 3.10)
project(TensorRTInference LANGUAGES CXX)


# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(TENSORRT_ROOT "/usr/src/tensorrt")
find_package(yaml-cpp REQUIRED)
find_package( OpenCV REQUIRED )
find_package(CUDA REQUIRED)
find_package(Eigen3 REQUIRED)

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
        HINTS ${TENSORRT_ROOT} PATH_SUFFIXES include/)
message(STATUS "Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")
find_library(TENSORRT_LIBRARY_INFER nvinfer
        HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)

set(TENSORRT_LIBRARY ${TENSORRT_LIBRARY_INFER})
message(STATUS "Find TensorRT libs: ${TENSORRT_LIBRARY}")

include_directories(
    include
    ${PROJECT_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
    ${Eigen3_INCLUDE_DIR}
    /usr/include/eigen3
    ${CUDA_INCLUDE_DIRS}
    ${YAML_CPP_INCLUDE_DIRS}
)

message(STATUS "Found Eigen headers at ${Eigen3_INCLUDE_DIR}")
message(STATUS "Find OpenCV include at ${OpenCV_INCLUDE_DIRS}")
message(STATUS "Find CUDA include at ${CUDA_INCLUDE_DIRS}")

add_subdirectory(third_party/cnpy)
add_subdirectory(src/ransac_voting)
add_subdirectory(src/math_helper)
add_subdirectory(src/utils)

message(STATUS "Find OpenCV lib: ${OpenCV_LIBS}")
message(STATUS "Find Eigen lib: ${Eigen3_LIBRARIES}")
message(STATUS "Find CUDA libraries: ${CUDA_LIBRARIES}")
message(STATUS "Find OpenCV lib: ${OpenCV_LIBS}")
message(STATUS "Find Eigen lib: ${Eigen3_LIBRARIES}")
message(STATUS "Find CUDA libraries: ${CUDA_LIBRARIES}")
add_executable(tensorrt_inference main.cpp)
add_executable(useClass useClass.cpp)
target_include_directories(tensorrt_inference PRIVATE 
    third_party/cnpy
)
target_include_directories(useClass PRIVATE 
    third_party/cnpy
)
# 链接 TensorRT 和 CUDA 库
target_link_libraries(tensorrt_inference
    ${CUDA_LIBRARIES} 
    ${TENSORRT_LIBRARY}
    ${OpenCV_LIBS}
    cnpy
    ransac_voting
    utils
    math_helper
    ${Eigen3_LIBRARIES}
    ${YAML_CPP_LIBRARIES}
)

target_link_libraries(useClass
    ${CUDA_LIBRARIES} 
    ${TENSORRT_LIBRARY}
    ${CUDA_LIBRARIES} 
    ${TENSORRT_LIBRARY}
    ${OpenCV_LIBS}
    cnpy
    ransac_voting
    utils
    math_helper
    ${Eigen3_LIBRARIES}
    ${YAML_CPP_LIBRARIES}
)
