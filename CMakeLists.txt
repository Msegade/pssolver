cmake_minimum_required (VERSION 2.8)
project (pssolver)

set (CMAKE_CXX_STANDARD 11)

if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug" OR "${CMAKE_BUILD_TYPE}" STREQUAL "debug")
    set(CMAKE_CXX_FLAGS "-pg -O0 -Wall -fPIC")
else()
    set(CMAKE_CXX_FLAGS "-O3 -Wall -fPIC")
endif()

set (EXECUTABLE_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/build/bin/")
set (LIBRARY_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/build/lib/")

enable_testing()

add_subdirectory(src)
add_subdirectory(tests)

