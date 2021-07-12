//
// Created by klaus on 12.07.21.
//

#ifndef ARC_GPUWRITER_CUH
#define ARC_GPUWRITER_CUH

#include <vector>
#include "GPUStream.cuh"
#include <memory>
#include <string>
#include <iostream>
#include <fstream>
#include <string>
#include <ios>
#include <queue>

class GPUWriter {
    std::vector<std::shared_ptr<GPUIntStream>> int_streams;
    std::vector<std::shared_ptr<GPUUnitStream>> unit_streams;
    std::vector<std::string> int_names;
    std::vector<std::string> unit_names;
    std::string FILENAME;
public:
    GPUWriter(std::string outputFile);
    void addIntStream(std::string name, std::shared_ptr<GPUIntStream> intStream);
    void addUnitStream(std::string name, std::shared_ptr<GPUUnitStream> unitStream);
    void writeOutputFile();
};

#endif //ARC_GPUWRITER_CUH
