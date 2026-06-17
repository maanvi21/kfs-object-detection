cmake_minimum_required(VERSION 3.10)
project(JetsonOBBEngine)

set(CMAKE_CXX_STANDARD 14)
find_package(OpenCV REQUIRED)

include_directories(
    /usr/include/aarch64-linux-gnu 
    /usr/local/cuda/include 
    ${OpenCV_INCLUDE_DIRS}
)

link_directories(
    /usr/lib/aarch64-linux-gnu 
    /usr/local/cuda/lib64
)

add_executable(tensorrt_obb_core tensorrt_obb_core.cpp)

target_link_libraries(tensorrt_obb_core 
    nvinfer 
    cudart 
    ${OpenCV_LIBS}
)
