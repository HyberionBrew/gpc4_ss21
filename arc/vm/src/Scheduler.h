//
// Created by fabian on 30/05/2021.
//

#ifndef GPC4_SS21_SCHEDULER_H
#define GPC4_SS21_SCHEDULER_H

#include "InstrInterface.h"
#include <vector>
#include <cstddef>
#include <atomic>
#include <string>

struct Register {
    void* location;
    std::atomic_flag available = ATOMIC_FLAG_INIT;
    int update;
};

class Scheduler {
protected:
    InstrInterface & instrInterface;
    std::string outfile;
public:
    Scheduler() = delete;
    explicit Scheduler(InstrInterface & interface);
    virtual bool next() = 0;
    virtual void warmup (std::string in_file) = 0;
    virtual void cooldown (std::string outfile) = 0;
};

#endif //GPC4_SS21_SCHEDULER_H
