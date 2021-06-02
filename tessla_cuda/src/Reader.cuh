#ifndef TESSLA_CUDA_READER_CUH
#define TESSLA_CUDA_READER_CUH

#include <string>
#include "Stream.cuh"

class Reader {
    std::string FILENAME;
public:
    Reader(std::string inputfile);
    UnitStream getUnitStream(std::string name);
    IntStream getIntStream(std::string name);
};

#endif //TESSLA_CUDA_READER_CUH