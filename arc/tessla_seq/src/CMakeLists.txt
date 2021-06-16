include_directories(${arc_SOURCE_DIR}/tessla_seq/src)

FIND_PACKAGE(OpenMP REQUIRED)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

add_library(tessla_seq
        Debug.cpp
        Event.cpp
        Reader.cpp
        Stream.cpp
        StreamFunctions.cpp
        Writer.cpp
)

add_executable(tessla_seq_test
        Debug.cpp
        Event.cpp
        Reader.cpp
        Stream.cpp
        StreamFunctions.cpp
        Writer.cpp
        ../test/test_main.cpp
        ../test/test_test.cpp
        )

add_executable(reader_benchmark
        Debug.cpp
        Event.cpp
        Reader.cpp
        Stream.cpp
        StreamFunctions.cpp
        Writer.cpp
        ../benchmark/reader_benchmark.cpp
        )

