cmake_minimum_required(VERSION 3.16)
project(opencl_q8_gemv LANGUAGES CXX)

find_package(OpenCL REQUIRED)
find_package(OpenMP)

add_executable(gemv_opencl src/main.cpp)
target_compile_features(gemv_opencl PRIVATE cxx_std_17)
target_link_libraries(gemv_opencl PRIVATE OpenCL::OpenCL)
if(OpenMP_CXX_FOUND)
    target_link_libraries(gemv_opencl PRIVATE OpenMP::OpenMP_CXX)
endif()

if(MSVC)
    target_compile_options(gemv_opencl PRIVATE /W4 /O2 /utf-8)
else()
    target_compile_options(gemv_opencl PRIVATE -Wall -Wextra -Wpedantic -O3)
endif()

file(COPY kernels DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
