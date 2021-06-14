//
// Created by fabian on 31/05/2021.
//

#ifndef GPC4_SS21_DEBUGSCHEDULER_H
#define GPC4_SS21_DEBUGSCHEDULER_H

#include "Scheduler.h"

class DebugScheduler : public Scheduler {
private:
    size_t line;
public:
    DebugScheduler(InstrInterface & interface);
    bool next() override;
    void warmup(std::string in_file) override;
};

#endif //GPC4_SS21_DEBUGSCHEDULER_H
