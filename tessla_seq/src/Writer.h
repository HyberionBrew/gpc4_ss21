//
// Created by daniel on 20.05.21.
//

#ifndef GPU_TESSLA_WRITER_H
#define GPU_TESSLA_WRITER_H

#include <string>

class Writer {
    std::vector<Stream> streams;
public:
    Writer(string outputFile);
    UnitStream addUnitStream(string name);
    IntStream addIntStream(string name);
    void writeOutputFile();
}

#endif //GPU_TESSLA_WRITER_H
