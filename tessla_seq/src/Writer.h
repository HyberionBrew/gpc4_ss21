//
// Created by daniel on 20.05.21.
//

#ifndef GPU_TESSLA_WRITER_H
#define GPU_TESSLA_WRITER_H

#include <string>
#include "Stream.h"

class Writer {
    std::vector<OutputStream> streams;
    std::string FILENAME;
public:
    Writer(std::string outputFile);
    void addStream(std::string name, Stream& stream);
    void writeOutputFile();
};

#endif //GPU_TESSLA_WRITER_H
