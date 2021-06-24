include_directories(${arc_SOURCE_DIR}/vm/src)

# Include the worker libraries
include_directories(
        ${PROJECT_SOURCE_DIR}/external/lfqueue
        ${PROJECT_SOURCE_DIR}/tessla_seq/src
#        ${PROJECT_SOURCE_DIR}/tessla_cuda/src
)
link_directories(
        ${PROJECT_SOURCE_DIR}/external/lfqueue
        ${PROJECT_SOURCE_DIR}/tessla_seq/src
#        ${PROJECT_SOURCE_DIR}/tessla_cuda/src
)
add_executable(arc_test
        ../src/runner.cpp
        ../src/DebugScheduler.cpp
        ../src/SequentialScheduler.cpp
        ../src/Decode.cpp
        ../src/GPUScheduler.cpp
        ../src/InstrInterface.cpp ../src/InstrInterface.h
        ../src/Scheduler.cpp
        test_main.cpp
        test_vm.cpp
)

target_link_libraries(arc_test tessla_seq)
#target_link_libraries(arc_test tessla_cuda)
target_link_libraries(arc_test lfqueue)