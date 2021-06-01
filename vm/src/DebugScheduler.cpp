//
// Created by fabian on 31/05/2021.
//

#include "DebugScheduler.h"
#include <iostream>
#include <assert.h>

DebugScheduler::DebugScheduler(InstrInterface & interface) : Scheduler(interface) {
    line = 0;
}

bool DebugScheduler::next() {
    line++;
    Instruction inst = instrInterface.pop();
    switch (inst.type) {
        case inst_add:
            std::cout << line << ": Add, R1: " << inst.r1 << " R2: " << inst.r2 << " RD: " << inst.rd << std::endl;
            break;
        case inst_mul:
            std::cout << line << ": Mul, R1: " << inst.r1 << " R2: " << inst.r2 << " RD: " << inst.rd << std::endl;
            break;;
        case inst_sub:
            std::cout << line << ": Sub, R1: " << inst.r1 << " R2: " << inst.r2 << " RD: " << inst.rd << std::endl;
            break;
        case inst_div:
            std::cout << line << ": Div, R1: " << inst.r1 << " R2: " << inst.r2 << " RD: " << inst.rd << std::endl;
            break;
        case inst_mod:
            std::cout << line << ": Mod, R1: " << inst.r1 << " R2: " << inst.r2 << " RD: " << inst.rd << std::endl;
            break;
        case inst_delay:
            std::cout << line << ": Delay, R1: " << inst.r1 << " R2: " << inst.r2 << " RD: " << inst.rd << std::endl;
            break;
        case inst_last:
            std::cout << line << ": Last, R1: " << inst.r1 << " R2: " << inst.r2 << " RD: " << inst.rd << std::endl;
            break;
        case inst_time:
            std::cout << line << ": Time, R1: " << inst.r1 << " RD: " << inst.rd << std::endl;
            break;
        case inst_merge:
            std::cout << line << ": Merge, R1: " << inst.r1 << " R2: " << inst.r2 << " RD: " << inst.rd << std::endl;
            break;
        case inst_count:
            std::cout << line << ": Count, R1: " << inst.r1 << " RD: " << inst.rd << std::endl;
            break;
        case inst_addi:
            std::cout << line << ": AddI, R1: " << inst.r1 << " IMM: " << inst.imm << " RD: " << inst.rd << std::endl;
            break;
        case inst_muli:
            std::cout << line << ": MulI, R1: " << inst.r1 << " IMM: " << inst.imm << " RD: " << inst.rd << std::endl;
            break;
        case inst_subi:
            std::cout << line << ": SubI, R1: " << inst.r1 << " IMM: " << inst.imm << " RD: " << inst.rd << std::endl;
            break;
        case inst_subii:
            std::cout << line << ": SubII, R1: " << inst.r1 << " IMM: " << inst.imm << " RD: " << inst.rd << std::endl;
            break;
        case inst_divi:
            std::cout << line << ": DivI, R1: " << inst.r1 << " IMM: " << inst.imm << " RD: " << inst.rd << std::endl;
            break;
        case inst_divii:
            std::cout << line << ": DivII, R1: " << inst.r1 << " IMM: " << inst.imm << " RD: " << inst.rd << std::endl;
            break;
        case inst_modi:
            std::cout << line << ": ModI, R1: " << inst.r1 << " IMM: " << inst.imm << " RD: " << inst.rd << std::endl;
            break;
        case inst_modii:
            std::cout << line << ": ModII, R1: " << inst.r1 << " IMM: " << inst.imm << " RD: " << inst.rd << std::endl;
            break;
        case inst_default:
            std::cout << line << ": Default, R1: " << inst.r1 << " IMM: " << inst.imm << " RD: " << inst.rd << std::endl;
            break;
        case inst_load:
            std::cout << line << ": Load, R1: " << inst.r1 << std::endl;
            break;
        case inst_load4:
            std::cout << line << ": Load4, R1: " << inst.r1 << std::endl;
            break;
        case inst_load6:
            std::cout << line << ": Load6, R1: " << inst.r1 << std::endl;
            break;
        case inst_load8:
            std::cout << line << ": Load8, R1: " << inst.r1 << std::endl;
            break;
        case inst_store:
            std::cout << line << ": Store, R1: " << inst.r1 << std::endl;
            break;
        case inst_free:
            std::cout << line << ": Free, R1: " << inst.r1 << std::endl;
            break;
        case inst_unit:
            std::cout << line << ": Unit, R1: " << inst.r1 << std::endl;
            break;
        case inst_exit:
            std::cout << line << ": Exit" << std::endl;
            return false;
        default:
            // Unreachable code
            assert(false);
    }
    return true;
}