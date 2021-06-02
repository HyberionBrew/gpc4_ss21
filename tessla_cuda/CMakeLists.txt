cmake_minimum_required(VERSION 3.10)
project(tessla_cuda LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 14)

add_executable(tessla_cuda
        src/main.cu src/main.cuh src/StreamFunctions.cuh src/Stream.cuh src/Stream.cu src/StreamFunctions.cu src/helper.cuh src/device_information.cuh)

find_package(CUDA  REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")
add_executable(tessla_cuda_test
        test/test_main.cpp test/test_cuda.cpp
        src/StreamFunctions.cuh src/Stream.cuh src/Stream.cu src/StreamFunctions.cu src/helper.cuh src/device_information.cuh)
