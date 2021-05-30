//
// Created by fabian on 30/05/2021.
//

#include "InstrInterface.h"

InstrInterface::InstructionInterface() {
    size = 0;
}

void InstrInterface::push(Instruction inst) {
    queue.push(isnt);
    size++;
}

Instruction InstrInterface::pop() {
    Instruction inst;
    if (size > 0) {
        inst = queue.pop();
        size--;
    }
    return inst;
}

bool InstrInterface::is_empty() {
    if (size > 0) {
        return true;
    }
    return false;
}