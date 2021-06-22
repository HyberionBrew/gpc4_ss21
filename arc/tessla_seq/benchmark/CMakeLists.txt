# Include the worker libraries
include_directories(
        ${PROJECT_SOURCE_DIR}/tessla_seq/src
)
link_directories(
        ${PROJECT_SOURCE_DIR}/tessla_seq/src
)

add_executable(reader_benchmark
        reader_benchmark.cpp
)

target_link_libraries(reader_benchmark tessla_seq)