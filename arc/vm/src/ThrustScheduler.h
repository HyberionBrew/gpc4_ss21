//
// Created by fabian on 31/05/2021.
//

#ifndef GPC4_SS21_THRUSTSCHEDULER_H
#define GPC4_SS21_THRUSTSCHEDULER_H

#include "Scheduler.h"
#include <GPUStream.cuh>
#include <memory>
#include <vector>
#include <unordered_map>

using namespace std;

class ThrustScheduler : public Scheduler {
private:
    std::vector<IOStream> out_streams;
    unordered_map<size_t, shared_ptr<GPUIntStream>> intRegisters;
    unordered_map<size_t, shared_ptr<GPUUnitStream>> unitRegisters;
    void set_reg (size_t pos, shared_ptr<GPUIntStream> stream);
    void set_reg (size_t pos, shared_ptr<GPUUnitStream> stream);
    shared_ptr<GPUIntStream> get_intst (size_t reg);
    shared_ptr<GPUUnitStream> get_ust (size_t reg);
public:
    explicit ThrustScheduler(InstrInterface & interface);
    bool next() override;
    void warmup(std::string in_file) override;
    void cooldown (std::string outfile) override;
};


#endif //GPC4_SS21_THRUSTSCHEDULER_H
