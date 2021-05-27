//
// Created by daniel on 20.05.21.
//

#ifndef GPU_TESSLA_STREAM_H
#define GPU_TESSLA_STREAM_H

#include "Event.h"
#include <vector>
#include <string>

/**
 * A Vector of UnitEvents which only contain a timestamp
 */
typedef std::vector<UnitEvent> UnitStream;

/**
 * A Vector of IntEvents which contain a timestamp and a value
 */
typedef std::vector<IntEvent> IntStream;

/**
 * Possible stream types
 */
enum data_type {INT_STREAM, UNIT_STREAM};

/**
 * Stream class, can contain either an IntStream or a UnitStream
 */
class Stream {
private:
    IntStream intStream;
    UnitStream unitStream;

    data_type type;

public:
    std::string name;
    Stream(std::string name, IntStream in);
    Stream(std::string name, UnitStream in);
    /**
     * Returns the stream's type (int or unit)
     * @return the type
     */
    data_type get_type();
    /**
     * Returns the IntStream Vector
     * @return the Vector
     */
    IntStream get_IntStream();
    /**
     * Returns the UnitStream Vector
     * @return the Vector
     */
    UnitStream get_UnitStream();
};


// For further implementation:
// typedef BoolStream std::vector<BoolEvent>
// typedef FloatStream std::vector<FloatEvent>

#endif //GPU_TESSLA_STREAM_H
