//
// Created by daniel on 20.05.21.
//

#include "Stream.h"
Stream::Stream(IntStream in){
    Stream::stream s;
    s.intStream = in;
    data_type = INT_STREAM;
}
Stream::Stream(UnitStream in){
    Stream::stream s;
    s.unitStream = in;
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