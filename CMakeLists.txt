cmake_minimum_required (VERSION 2.8)
project (pssolver)

set (CMAKE_CXX_STANDARD 11)

find_package (CUDA REQUIRED)

set (CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 -arch=sm_20" )

set (EXECUTABLE_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/build/bin/")
set (LIBRARY_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/build/lib/")

enable_testing()

add_subdirectory(src)
add_subdirectory(tests)

