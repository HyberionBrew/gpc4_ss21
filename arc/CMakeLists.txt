cmake_minimum_required(VERSION 3.10)
project(arc LANGUAGES CUDA CXX)

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

# Add VM and lib subdirectories
add_subdirectory(tessla_seq)
add_subdirectory(external/lfqueue)
add_subdirectory(vm)

# Add the sequential execution library
#add_subdirectory(../tessla_seq ./tessla_seq)
#target_include_directories(${PROJECT_NAME} PUBLIC ../tessla_seq)
#target_link_directories(${PROJECT_NAME} PRIVATE ../tessla_seq)
#target_link_libraries(${PROJECT_NAME} tessla_seq)