//
// Created by fabian on 11/06/2021.
//

#ifndef GPC4_SS21_SEQUENTIALSCHEDULER_H
#define GPC4_SS21_SEQUENTIALSCHEDULER_H

#include "Scheduler.h"

class SequentialScheduler : public Scheduler {
private:
    size_t line;
public:
    SequentialScheduler(InstrInterface & interface);
    bool next() override;
    bool parse_input() override;
};

#endif //GPC4_SS21_SEQUENTIALSCHEDULER_H
