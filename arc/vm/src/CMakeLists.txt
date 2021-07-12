include_directories(${arc_SOURCE_DIR}/vm/src)

# Include the worker libraries
include_directories(
        ${PROJECT_SOURCE_DIR}/external/lfqueue
        ${PROJECT_SOURCE_DIR}/tessla_seq/src
        ${PROJECT_SOURCE_DIR}/tessla_cuda/src
)
link_directories(
        ${PROJECT_SOURCE_DIR}/external/lfqueue
        ${PROJECT_SOURCE_DIR}/tessla_seq/src
        ${PROJECT_SOURCE_DIR}/tessla_cuda/src
)
add_executable(arc
        main.cpp
        DebugScheduler.cpp
        SequentialScheduler.cpp
        Decode.cpp
        GPUScheduler.cpp
        InstrInterface.cpp InstrInterface.h
        ThrustScheduler.cpp ThrustScheduler.h
        Scheduler.cpp
        runner.cpp runner.h)
target_link_libraries(arc tessla_seq)
target_link_libraries(arc tessla_cuda)
target_link_libraries(arc lfqueue)