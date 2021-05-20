//
// Created by daniel on 20.05.21.
//

#ifndef GPU_TESSLA_READER_H
#define GPU_TESSLA_READER_H

#include "Stream.h"
#include <string>

void readStreams();

class Reader {
public:
    Reader(string inputFile);
    UnitStream getUnitStream(string name);
    IntStream getIntStream(string name);
}

#endif //GPU_TESSLA_READER_H
