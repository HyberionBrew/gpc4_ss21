//
// Created by daniel on 20.05.21.
//

#include "Stream.h"

IntStream::IntStream(std::vector<IntEvent> stream) {
    this->stream = stream;
}

UnitStream::UnitStream(std::vector<UnitEvent> stream) {
    this->stream = stream;
}

OutputStream::OutputStream(std::string name, Stream& stream) : name(name), stream(stream) {

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
}*/