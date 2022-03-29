cmake_minimum_required(VERSION 3.20)
project(opencl_gpu)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCL REQUIRED)
include_directories(OpenCL/include)
link_directories(OpenCL/lib)

add_executable(opencl_gpu main.cpp)
target_link_libraries(opencl_gpu OpenCL::OpenCL)
