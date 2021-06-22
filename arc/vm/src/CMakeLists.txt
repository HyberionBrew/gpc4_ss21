include_directories(${arc_SOURCE_DIR}/vm/inc)

# Include the worker libraries
include_directories(
        ${PROJECT_SOURCE_DIR}/external/lfqueue
        ${PROJECT_SOURCE_DIR}/tessla_seq/src
)
link_directories(
        ${PROJECT_SOURCE_DIR}/external/lfqueue
        ${PROJECT_SOURCE_DIR}/tessla_seq/src
)
add_executable(arc
        main.cpp
        DebugScheduler.cpp
        SequentialScheduler.cpp
        Decode.cpp
        GPUScheduler.cpp
        InstrInterface.cpp InstrInterface.h
        Scheduler.cpp
)
target_link_libraries(arc tessla_seq)
target_link_libraries(arc lfqueue)