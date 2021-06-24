# Include the worker libraries
include_directories(
        ${PROJECT_SOURCE_DIR}/tessla_cuda/src
)
link_directories(
        ${PROJECT_SOURCE_DIR}/tessla_cuda/src
)

add_executable(tessla_cuda_test
        test_main.cpp
        test_cuda.cpp
        test_time.cpp
        test_last.cpp
        test_benchmark.cpp
        )

set_target_properties(tessla_cuda_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Set CUDA target architectures
set_property(TARGET tessla_cuda_test PROPERTY CUDA_ARCHITECTURES 61 75)

target_link_libraries(tessla_cuda_test tessla_cuda)