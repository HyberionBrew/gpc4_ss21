//
// Created by daniel on 27.05.21.
//

#ifndef GPU_TESSLA_DECODE_H
#define GPU_TESSLA_DECODE_H

#include "InstrInterface.h"
#include <string>
#include <fstream>
#include <vector>
#include <cstddef>

typedef struct IOStream {
    std::string name;
    size_t regname;
} IOStream;

class Decode {
    std::ifstream coil;
    int currMV = 1;
    InstrInterface* instrInterface;
    int registerWidth;
private:
    void parse_header();
    void print_header();
    size_t read_register(unsigned char opcode);
    int32_t read_imm(unsigned char opcode);
public:
    int majorV;
    int minorV;
    bool wideAddresses;
    Decode(std::string coil_file);
    bool decode_next();
    std::vector<IOStream> in_streams;
    std::vector<IOStream> out_streams;
};



#endif //GPU_TESSLA_DECODE_H
