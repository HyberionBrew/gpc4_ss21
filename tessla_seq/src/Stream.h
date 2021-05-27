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
//typedef std::vector<UnitEvent> UnitStream;

/**
 * A Vector of IntEvents which contain a timestamp and a value
 */
//typedef std::vector<IntEvent> IntStream;


/**
 * Stream class, can contain either an IntStream or a UnitStream
 */
class Stream {
public:
    virtual std::vector<Event> get_event_stream() = 0;

    /**
     * Returns the stream's type (int or unit)
     * @return the type
     */
    bool is_equal_to(const Stream& other);
};

class IntStream : public Stream {
public:
    std::vector<Event> get_event_stream() override;
    std::vector<IntEvent> stream;

    IntStream();
    IntStream(size_t size);
    IntStream(std::vector<IntEvent> stream);

    friend bool operator==(IntStream lhs, IntStream rhs);
};

class UnitStream : public Stream {
public:
    std::vector<Event> get_event_stream() override;
    std::vector<UnitEvent> stream;

    UnitStream();
    UnitStream(size_t size);
    UnitStream(std::vector<UnitEvent> stream);

    friend bool operator==(UnitStream lhs, UnitStream rhs);
};

class OutputStream {
public:
    Stream& stream;
    std::string name;

    OutputStream(std::string name, Stream& stream);
};


// For further implementation:
// typedef BoolStream std::vector<BoolEvent>
// typedef FloatStream std::vector<FloatEvent>

#endif //GPU_TESSLA_STREAM_H
