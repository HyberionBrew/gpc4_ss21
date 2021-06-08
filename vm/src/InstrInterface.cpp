//
// Created by fabian on 30/05/2021.
//

#include "InstrInterface.h"
#include <stdexcept>

InstrInterface::InstrInterface() {
    if (lfqueue_init(&queue) == -1) {
        throw std::runtime_error("Could not initialize instruction interface. Memory full?");
    }
}

void InstrInterface::push(Instruction inst) {
    Instruction* inst_data = (Instruction*) malloc(sizeof(Instruction));
    if (inst_data == NULL) {
        throw std::runtime_error("Could not enqueue instruction. Out of memory?");
    }
    *inst_data = inst;
    // Wait for the instruction to be enqueued
    while (lfqueue_enq(&queue, inst_data) == -1) {
    }
}

Instruction InstrInterface::pop() {
    void* data = nullptr;
    // Wait for element to be dequeued
    while ( (data = lfqueue_single_deq_must(&queue)) == NULL) {
    }
    Instruction* inst_data = (Instruction*)data;
    Instruction inst = *inst_data;
    free(inst_data);
    return inst;
}

bool InstrInterface::is_nonempty() {
    size_t size = lfqueue_size(&queue);
    if (size > 0) {
        return true;
    }
    return false;
}

InstrInterface::~InstrInterface() {
    lfqueue_destroy(&queue);
}