//
// Created by fabian on 11/06/2021.
//

#ifndef GPC4_SS21_IOSTREM_H
#define GPC4_SS21_IOSTREM_H

#include <stddef.h>
#include <string>

enum ioStreamType{io_unit, io_integer};
enum ioDirection{io_in, io_out};

struct IOStream {
    std::string name;
    size_t regname;
    ioStreamType type;
    ioDirection direction;
};

#endif //GPC4_SS21_IOSTREM_H
