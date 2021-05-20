//
// Created by daniel on 20.05.21.
//

#ifndef GPU_TESSLA_EVENT_H
#define GPU_TESSLA_EVENT_H

#include <cstdint>
#include <cstddef>

typedef struct UnitEvent{
    size_t timestamp;
} UnitEvent;

typedef struct IntEvent{
    size_t timestamp;
    int32_t value;
} IntEvent;

#endif //GPU_TESSLA_EVENT_H
