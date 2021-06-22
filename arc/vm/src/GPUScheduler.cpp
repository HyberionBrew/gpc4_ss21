//
// Created by fabian on 31/05/2021.
//

#include "GPUScheduler.h"

GPUScheduler::GPUScheduler(InstrInterface & interface) : Scheduler(interface) {
}

bool GPUScheduler::next() {
    return false;
}

void GPUScheduler::warmup(std::string in_file) {
}

void GPUScheduler::cooldown(std::string outfile) {
}