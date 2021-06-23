cmake_minimum_required(VERSION 3.10)
project(arc LANGUAGES CUDA CXX)
set(CMAKE_CXX_STANDARD 14)

# Set gcc compiler flags
SET(GCC_COMPILE_FLAGS "-g --std=c++14 -pthread")
SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COMPILE_FLAGS}")

# Download all necessary Git submodules
find_package(Git QUIET)
if (GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
    # Update submodules as needed
    option(GIT_SUBMODULE "Check submodules during build" ON)
    if (GIT_SUBMODULE)
        message(STATUS "Submodule update")
        execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                RESULT_VARIABLE GIT_SUBMOD_RESULT)
        if (NOT GIT_SUBMOD_RESULT EQUAL "0")
            message(FATAL_ERROR "git submodule update --init failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
        endif ()
    endif ()
endif ()

# Check all submodules
# Check lfqueue
if (NOT EXISTS "${PROJECT_SOURCE_DIR}/external/lfqueue/CMakeLists.txt")
    message(FATAL_ERROR "The lfqueue submodule was not downloaded! GIT SUBMODULE was turned off or failed. Please update submodules and try again.")
endif ()

# Set output directory
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

# Require CUDA module
find_package(CUDA  REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

# Require the OpenMP package
FIND_PACKAGE(OpenMP REQUIRED)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Add VM and lib subdirectories
add_subdirectory(tessla_seq)
add_subdirectory(tessla_cuda)
add_subdirectory(external/lfqueue)
add_subdirectory(vm)