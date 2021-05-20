//
// Created by daniel on 20.05.21.
//

#include "Stream.h"
Stream::Stream(std::string name, IntStream in){
    Stream::stream s;
    s.intStream = in;
    this->name = name;
    data_type = INT_STREAM;
}
Stream::Stream(std::string name,UnitStream in){
    Stream::stream s;
    s.unitStream = in;
    this->name = name;
    data_type = UNIT_STREAM;
}

data_type Stream::get_type() {
    return type;
}

IntStream get_IntStream() {
    return s.intStream;
}

UnitStream get_UnitStream() {
    return s.unitStream;
}