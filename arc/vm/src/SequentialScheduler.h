//
// Created by fabian on 11/06/2021.
//

#ifndef GPC4_SS21_SEQUENTIALSCHEDULER_H
#define GPC4_SS21_SEQUENTIALSCHEDULER_H

#include "Scheduler.h"
#include <Stream.h>
#include <unordered_map>

class SequentialScheduler : public Scheduler {
private:
    size_t line;
    std::unordered_map<size_t, Stream*> registers;
    std::vector<IOStream> out_strems;
    void set_reg (size_t pos, Stream* stream);
    IntStream* get_intst (size_t reg);
    UnitStream* get_ust (size_t reg);
    Stream* get_st (size_t reg);
public:
    SequentialScheduler(InstrInterface & interface);
    bool next() override;
    void warmup(std::string in_file) override;
};

#endif //GPC4_SS21_SEQUENTIALSCHEDULER_H
