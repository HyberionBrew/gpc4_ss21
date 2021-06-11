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

enum ioStreamType{unit, integer};

struct IOStream {
    std::string name;
    size_t regname;
    ioStreamType type;
};

class Decode {
public:
    int majorV;
    int minorV;
    bool wideAddresses;
    Decode(std::string coil_file, InstrInterface & interface);
    Decode(std::string coil_file, InstrInterface &&) = delete;
    bool decode_next();
    void print_header();
    std::vector<IOStream> in_streams;
    std::vector<IOStream> out_streams;
private:
    std::ifstream coil;
    int currMV = 1;
    InstrInterface & instrInterface;
    int registerWidth;
    void parse_header();
    size_t read_register(unsigned char opcode);
    int32_t read_imm(unsigned char opcode);
    void throw_insttype_error(unsigned char opcode);
    void throw_read_error(unsigned char opcode);
};



#endif //GPU_TESSLA_DECODE_H
