//
// Created by daniel on 20.05.21.
//

#ifndef GPU_TESSLA_READER_H
#define GPU_TESSLA_READER_H

#include "Stream.h"
#include <string>
#include <memory>

void readStreams();

class Reader {
    std::string FILENAME;
public:
    Reader(std::string inputFile);
    Reader() = delete;
    std::shared_ptr<UnitStream> getUnitStream(std::string name);
    std::shared_ptr<IntStream> getIntStream(std::string name);
};

#endif //GPU_TESSLA_READER_H
