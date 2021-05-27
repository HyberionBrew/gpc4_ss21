//
// Created by daniel on 20.05.21.
//

#ifndef GPU_TESSLA_EVENT_H
#define GPU_TESSLA_EVENT_H

#include <cstdint>
#include <cstddef>
#include <string>

/*
typedef struct UnitEvent{
    size_t timestamp;
} UnitEvent;

typedef struct IntEvent{
    size_t timestamp;
    int32_t value;
} IntEvent;
*/

class Event{
public:
    size_t timestamp;
    virtual std::string string_rep(std::string name) = 0;
};

class UnitEvent : public Event{

public:
    size_t timestamp;

    UnitEvent(size_t ts);
    std::string string_rep(std::string name) override;

};

class IntEvent : public Event{

public:
    size_t timestamp;
    int32_t value;

    IntEvent(size_t ts, int32_t v);
    std::string string_rep(std::string name) override;

};

#endif //GPU_TESSLA_EVENT_H
