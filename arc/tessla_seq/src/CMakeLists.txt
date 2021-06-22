include_directories(${PROJECT_SOURCE_DIR}/tessla_seq/src)

add_library(tessla_seq
        Debug.cpp
        Event.cpp
        Reader.cpp
        Stream.cpp
        StreamFunctions.cpp
        Writer.cpp
)

