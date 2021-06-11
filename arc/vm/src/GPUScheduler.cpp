//
// Created by fabian on 31/05/2021.
//

#include "GPUScheduler.h"

// TODO remember to resolve "free" update counters

GPUScheduler::GPUScheduler(InstrInterface & interface) : Scheduler(interface) {
}

bool GPUScheduler::next() {
    return false;
}

bool GPUScheduler::parse_input() {
    return true;
}