//
// Created by daniel on 20.05.21.
//

#ifndef GPU_TESSLA_WRITER_H
#define GPU_TESSLA_WRITER_H

#include <string>
#include "Stream.h"

class Writer {
    std::vector<Stream> streams;
public:
    Writer(std::string outputFile);
    UnitStream addUnitStream(std::string name, UnitStream stream);
    IntStream addIntStream(std::string name, IntStream stream);
    void writeOutputFile();
};

#endif //GPU_TESSLA_WRITER_H
