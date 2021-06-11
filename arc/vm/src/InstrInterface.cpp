//
// Created by fabian on 30/05/2021.
//

#include "InstrInterface.h"
#include <stdexcept>

InstrInterface::InstrInterface() : ioReady() {
    ioReady = false;
    if (lfqueue_init(&queue) == -1) {
        throw std::runtime_error("Could not initialize instruction interface. Memory full?");
    }
    if (lfqueue_init(&ioStreams) == -1) {
        throw std::runtime_error("Could not initialize IO stream interface. Memory full?");
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

void InstrInterface::push_iostream(IOStream stream) {
    IOStream* stream_data = (IOStream*) malloc(sizeof(IOStream));
    if (stream_data == NULL) {
        throw std::runtime_error("Could not enqueue IO stream. Out of memory?");
    }
    *stream_data = stream;
    // Wait for the instruction to be enqueued
    while (lfqueue_enq(&queue, stream_data) == -1) {
    }
}

IOStream InstrInterface::pop_iostream() {
    void* data = nullptr;
    // Wait for element to be dequeued
    while ( (data = lfqueue_single_deq_must(&ioStreams)) == NULL) {
    }
    IOStream* stream_data = (IOStream*)data;
    IOStream stream = *stream_data;
    free(stream_data);
    return stream;
}

bool InstrInterface::iostream_pending() {
    size_t size = lfqueue_size(&ioStreams);
    if (size > 0) {
        return true;
    }
    return false;
}

InstrInterface::~InstrInterface() {
    lfqueue_destroy(&queue);
    lfqueue_destroy(&ioStreams);
}