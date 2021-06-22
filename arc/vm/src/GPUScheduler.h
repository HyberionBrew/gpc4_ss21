//
// Created by fabian on 31/05/2021.
//

#ifndef GPC4_SS21_GPUSCHEDULER_H
#define GPC4_SS21_GPUSCHEDULER_H

#include "Scheduler.h"

class GPUScheduler : public Scheduler {
private:
    std::vector<Instruction> lookahead;
    std::vector<Register> registers;
public:
    explicit GPUScheduler(InstrInterface & interface);
    bool next() override;
    void warmup(std::string in_file) override;
    void cooldown (std::string outfile) override;
};


#endif //GPC4_SS21_GPUSCHEDULER_H
