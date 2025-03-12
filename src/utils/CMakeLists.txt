# 编译 ransac_voting.cpp 为静态库
add_library(utils STATIC utils.cpp)

target_include_directories(utils PRIVATE 
    ${CMAKE_SOURCE_DIR}/third_party/cnpy
)
# 链接 OpenCV
target_link_libraries(utils ${OpenCV_LIBS} cnpy)
