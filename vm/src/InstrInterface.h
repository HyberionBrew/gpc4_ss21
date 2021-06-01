//
// Created by fabian on 30/05/2021.
//

#ifndef GPC4_SS21_INSTRINTERFACE_H
#define GPC4_SS21_INSTRINTERFACE_H

#include "lfqueue.h"
#include <cstddef>
#include <cstdint>

// Careful! This is a single reader multiple writer construction!!!

// Instruction type enum
enum instructionType{inst_add, inst_mul, inst_sub, inst_div, inst_mod, inst_delay, inst_last, inst_time,
        inst_merge, inst_count, inst_addi, inst_muli, inst_subi, inst_subii, inst_divi, inst_divii, inst_modi,
        inst_modii, inst_default, inst_load, inst_load4, inst_load6, inst_load8, inst_store, inst_free,
        inst_unit, inst_exit};

struct Instruction {
    instructionType type;
    size_t r1;
    size_t r2;
    size_t rd;
    int32_t imm;
};

class InstrInterface {
private:
    lfqueue_t queue;
public:
    InstrInterface();
    void push(Instruction inst);
    Instruction pop();
    bool is_nonempty();
    ~InstrInterface();
};


#endif //GPC4_SS21_INSTRINTERFACE_H
