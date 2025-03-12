cmake_minimum_required(VERSION 3.10) # 建议使用较新版本

# 设置子模块名称 (一定要定义)
set(SUBMODULE_NAME ransac_voting)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

# 设置 CUDA 编译选项 (非常重要!!!)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++17") # 设置C++标准，很重要
set(CMAKE_CUDA_ARCHITECTURES 87)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CUDA_GEN_CODE "-gencode=arch=compute_87,code=sm_87")
# 默认不进行优化， 也可以启用编译优化，建议等级在 -O1 或者-O2 
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --default-stream=per-thread --threads 0  -O0 -Xcompiler -fPIC -g -w ${CUDA_GEN_CODE}")
enable_language(CUDA)


# 添加 .cu 文件并创建库
cuda_add_library(kernels STATIC ransac_helper.cu)

# 编译 ransac_voting.cpp 为静态库
add_library(${SUBMODULE_NAME} STATIC ransac_voting.cpp)

# 设置目标文件的输出目录
set_target_properties(${SUBMODULE_NAME} PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# 链接库 (正确的顺序)
target_link_libraries(${SUBMODULE_NAME} PRIVATE kernels ${OpenCV_LIBS} ${CUDA_LIBRARIES})


message(STATUS "Find CUDA include at ${CUDA_INCLUDE_DIRS}")
message(STATUS "Find CUDA libraries: ${CUDA_LIBRARIES}")
message(STATUS "Find CUDA toolkit : ${CUDA_TOOLKIT_ROOT_DIR}")