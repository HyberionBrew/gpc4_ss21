//
// Created by fabian on 30/05/2021.
//

#ifndef GPC4_SS21_SCHEDULER_H
#define GPC4_SS21_SCHEDULER_H

#include "InstrInterface.h"
#include <vector>
#include <cstddef>
#include <atomic>

struct Register {
    void* location;
    std::atomic<bool> available;
    int update;
};

class Scheduler {
protected:
    InstrInterface* instrInterface;
public:
    Scheduler(InstrInterface* interface);
    virtual bool next() = 0;
};

#endif //GPC4_SS21_SCHEDULER_H
