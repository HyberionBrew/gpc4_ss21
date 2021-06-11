//
// Created by daniel on 20.05.21.
//

#ifndef GPU_TESSLA_READER_H
#define GPU_TESSLA_READER_H

#include "Stream.h"
#include <string>

void readStreams();

class Reader {
    std::string FILENAME;
public:
    Reader(std::string inputFile);
    UnitStream getUnitStream(std::string name);
    IntStream getIntStream(std::string name);
};

#endif //GPU_TESSLA_READER_H
