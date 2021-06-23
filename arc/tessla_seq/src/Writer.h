//
// Created by daniel on 20.05.21.
//

#ifndef GPU_TESSLA_WRITER_H
#define GPU_TESSLA_WRITER_H

#include <string>
#include <memory>
#include "Stream.h"

class Writer {
    std::vector<std::shared_ptr<Stream>> streams;
    std::vector<std::string> stream_names;
    std::string FILENAME;
    long unitPos; // position of last unit stream
public:
    Writer(std::string outputFile);
    void addUnitStream(std::string name, std::shared_ptr<Stream> unitStream);
    void addIntStream(std::string name, std::shared_ptr<Stream> intStream);
    void writeOutputFile();
};

#endif //GPU_TESSLA_WRITER_H
