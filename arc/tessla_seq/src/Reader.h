//
// Created by daniel on 20.05.21.
//

#ifndef GPU_TESSLA_READER_H
#define GPU_TESSLA_READER_H

#include "Stream.h"
#include <string>
#include <memory>
#include <map>

void readStreams();

class Reader {
    const char *FILENAME;
    std::map<std::string, std::shared_ptr<UnitStream>> unitStreams;
    std::map<std::string, std::shared_ptr<IntStream>> intStreams;
    void readStreams();
public:
    explicit Reader(const char *inputFile);
    std::shared_ptr<UnitStream> getUnitStream(const char *name);
    std::shared_ptr<IntStream> getIntStream(const char *name);
};

#endif //GPU_TESSLA_READER_H
