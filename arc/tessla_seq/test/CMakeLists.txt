# Include the worker libraries
include_directories(
        ${PROJECT_SOURCE_DIR}/tessla_seq/src
)
link_directories(
        ${PROJECT_SOURCE_DIR}/tessla_seq/src
)

add_executable(tessla_seq_test
        test_main.cpp
        test_test.cpp
        test_benchmark_seq.cpp
)

target_link_libraries(tessla_seq_test tessla_seq)
