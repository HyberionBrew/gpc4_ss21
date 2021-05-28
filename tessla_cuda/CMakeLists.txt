cmake_minimum_required(VERSION 3.16)
project(tessla_cuda LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 14)

add_executable(tessla_cuda
        src/main.cu src/main.cuh src/StreamFunctions.cuh src/Stream.cuh src/Stream.cu src/StreamFunctions.cu src/helper.cuh src/device_information.cuh)
