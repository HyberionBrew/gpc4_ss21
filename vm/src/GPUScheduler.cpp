//
// Created by fabian on 31/05/2021.
//

#include "GPUScheduler.h"

// TODO remember to resolve "free" update counters

GPUScheduler::GPUScheduler(InstrInterface *interface) {
    instrInterface = interface;
}

bool GPUScheduler::next() {
    return false;
}