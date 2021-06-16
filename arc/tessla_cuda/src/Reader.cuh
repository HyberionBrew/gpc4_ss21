#ifndef TESSLA_CUDA_READER_CUH
#define TESSLA_CUDA_READER_CUH

#include <string>
#include "Stream.cuh"
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

class Reader {
    std::string FILENAME;
    std::map<std::string, std::shared_ptr<UnitInStream>> unitStreams;
    std::map<std::string, std::shared_ptr<IntInStream>> intStreams;
    void readStreams();
public:
    Reader(std::string inputfile);
    UnitStream getUnitStream(std::string name);
    IntStream getIntStream(std::string name);
};

#endif //TESSLA_CUDA_READER_CUH