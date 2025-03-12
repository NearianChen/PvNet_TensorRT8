add_library(math_helper STATIC math_helper.cpp)

target_link_libraries(math_helper ${OpenCV_LIBS})
