include_directories(${arc_SOURCE_DIR}/vm/inc)

# Include the worker libraries
include_directories(
        ${arc_SOURCE_DIR}/tessla_seq/src
        ${arc_SOURCE_DIR}/external/lfqueue
)
link_directories(
        ${arc_SOURCE_DIR}/tessla_seq/src
        ${arc_SOURCE_DIR}/external/lfqueue
)
add_executable(arc
        main.cpp
        DebugScheduler.cpp
        Decode.cpp
        GPUScheduler.cpp
        InstrInterface.cpp
        Scheduler.cpp
)
target_link_libraries(arc tessla_seq)