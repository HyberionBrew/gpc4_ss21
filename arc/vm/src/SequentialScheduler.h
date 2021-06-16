//
// Created by fabian on 11/06/2021.
//

#ifndef GPC4_SS21_SEQUENTIALSCHEDULER_H
#define GPC4_SS21_SEQUENTIALSCHEDULER_H

#include "Scheduler.h"
#include <Stream.h>
#include <unordered_map>
#include <memory>
using namespace std;

class SequentialScheduler : public Scheduler {
private:
    size_t line;
    unordered_map<size_t, shared_ptr<IntStream>> intRegisters;
    unordered_map<size_t, shared_ptr<UnitStream>> unitRegisters;
    vector<IOStream> out_strems;
    void set_reg (size_t pos, shared_ptr<IntStream> stream);
    void set_reg (size_t pos, shared_ptr<UnitStream> stream);
    shared_ptr<IntStream> get_intst (size_t reg);
    shared_ptr<UnitStream> get_ust (size_t reg);
    shared_ptr<Stream> get_st (size_t reg);
public:
    SequentialScheduler(InstrInterface & interface);
    bool next() override;
    void warmup(const char *in_file) override;
};

#endif //GPC4_SS21_SEQUENTIALSCHEDULER_H
