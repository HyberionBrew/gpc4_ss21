cmake_minimum_required(VERSION 3.10)
project(tessla_cuda LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 14)
find_package(CUDA  REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

# Test executable, add option to not build by default
add_executable(tessla_cuda
        src/main.cu src/main.cuh src/StreamFunctions.cuh src/Reader.cuh src/Reader.cu src/Stream.cuh src/Stream.cu src/StreamFunctions.cu src/StreamFunctionHelper.cuh src/StreamFunctionHelper.cu src/helper.cuh src/device_information.cuh)

add_executable(tessla_cuda_test
        test/test_main.cpp test/test_cuda.cpp
        src/Reader.cuh src/Reader.cu
        src/StreamFunctions.cuh src/StreamFunctions.cu 
        src/StreamFunctionHelper.cuh src/StreamFunctionHelper.cu
        src/Stream.cuh src/Stream.cu 
        src/helper.cuh src/device_information.cuh)

add_library(tessla_cuda_lib
        src/StreamFunctions.cuh
        src/Reader.cuh src/Reader.cu
        src/Stream.cuh src/Stream.cu
        src/StreamFunctions.cu
        src/StreamFunctionHelper.cuh src/StreamFunctionHelper.cu
        src/helper.cuh
        src/device_information.cuh
)

set_target_properties(tessla_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(tessla_cuda_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
