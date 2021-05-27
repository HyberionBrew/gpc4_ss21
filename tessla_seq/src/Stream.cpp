//
// Created by daniel on 20.05.21.
//

#include "Stream.h"
Stream::Stream(std::string name, IntStream in){
    this->intStream = in;
    this->name = name;
    this->type = INT_STREAM;
}
Stream::Stream(std::string name,UnitStream in){
    this->unitStream = in;
    this->name = name;
    this->type = UNIT_STREAM;
}

data_type Stream::get_type() {
    return this->type;
}

IntStream Stream::get_IntStream() {
    return this->intStream;
}

UnitStream Stream::get_UnitStream() {
    return this->unitStream;
}
/*
bool Stream::is_equal_to(Stream other) {
    if (this->get_type != other->get_type()) {
        return false;
    }
    if (this->get_type() == INT_STREAM) {
        return this->get_IntStream() == other.get_IntStream();
    } else if (this->get_type() == UNIT_STREAM) {
        return this->get_UnitStream() == other.get_UnitStream();
    }
}
*/