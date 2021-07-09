#ifndef TESSLA_CUDA_READER_CUH
#define TESSLA_CUDA_READER_CUH

#include <string>
#include "GPUStream.cuh"
#include <map>
#include <memory>
#include <vector>

class IntInStream {
public:
    std::vector<int> timestamps;
    std::vector<int> values;
    IntInStream();
};

class UnitInStream {
public:
    std::vector<int> timestamps;
    UnitInStream();
};

class GPUReader {
    std::string FILENAME;
    std::map<std::string, std::shared_ptr<UnitInStream>> GPUUnitStreams;
    std::map<std::string, std::shared_ptr<IntInStream>> GPUIntStreams;
    void readStreams();
public:
    GPUReader(std::string inputfile);
    std::shared_ptr<GPUUnitStream> getUnitStream(std::string name);
    std::shared_ptr<GPUIntStream> getIntStream(std::string name);
    std::shared_ptr<GPUUnitStream> getUnitStreamDebug(std::string name);
    std::shared_ptr<GPUIntStream> getIntStreamDebug(std::string name);
};

void printArray(int* array, size_t len, std::string name);

#endif //TESSLA_CUDA_READER_CUH