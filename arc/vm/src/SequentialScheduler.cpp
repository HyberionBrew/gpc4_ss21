//
// Created by fabian on 11/06/2021.
//

#include "SequentialScheduler.h"
#include <iostream>
#include <assert.h>
#include <Reader.h>
#include <StreamFunctions.h>

SequentialScheduler::SequentialScheduler(InstrInterface & interface) : Scheduler(interface) {
    line = 0;
}

bool SequentialScheduler::next() {
    line++;
    Instruction inst = instrInterface.pop();
    Stream* res;
    switch (inst.type) {
        case inst_add: {
            // Add two int streams together
            res = new IntStream;
            *res = add(*(get_intst(inst.r1)),
                       *(get_intst(inst.r2)));
            set_reg(inst.rd, res);
            break;
        }
        case inst_mul: {
            // Add two int streams together
            res = new IntStream;
            *res = mul(*(get_intst(inst.r1)),
                       *(get_intst(inst.r2)));
            set_reg(inst.rd, res);
            break;
        }
        case inst_sub: {
            // Add two int streams together
            res = new IntStream;
            *res = sub(*(get_intst(inst.r1)),
                       *(get_intst(inst.r2)));
            set_reg(inst.rd, res);
            break;
        }
        case inst_div: {
            // Add two int streams together
            res = new IntStream;
            *res = div(*(get_intst(inst.r1)),
                       *(get_intst(inst.r2)));
            set_reg(inst.rd, res);
            break;
        }
        case inst_mod: {
            // Add two int streams together
            res = new IntStream;
            *res = mod(*(get_intst(inst.r1)),
                       *(get_intst(inst.r2)));
            set_reg(inst.rd, res);
            break;
        }
        case inst_delay: {
            // Add two int streams together
            res = new UnitStream;
            *res = delay(*(get_intst(inst.r1)),
                       *(get_st(inst.r2)));
            set_reg(inst.rd, res);
            break;
        }
        case inst_last: {
            // Add two int streams together
            res = new IntStream;
            *res = last(*(get_intst(inst.r1)),
                         *(get_st(inst.r2)));
            set_reg(inst.rd, res);
            break;
        }
        case inst_time: {
            // Add two int streams together
            res = new IntStream;
            *res = time(*(get_st(inst.r1)));
            set_reg(inst.rd, res);
            break;
        }
        case inst_merge: {
            // Add two int streams together
            // TODO fix this!!!
            res = new IntStream;
            *res = merge(*(get_intst(inst.r1)),
                       *(get_intst(inst.r2)));
            set_reg(inst.rd, res);
            break;
        }
        case inst_count: {
            // Add two int streams together
            res = new IntStream;
            *res = count(*(get_ust(inst.r1)));
            set_reg(inst.rd, res);
            break;
        }
        case inst_addi: {
            // Add two int streams together
            res = new IntStream;
            *res = add(*(get_intst(inst.r1)),
                        inst.imm);
            set_reg(inst.rd, res);
            break;
        }
        case inst_muli: {
            // Add two int streams together
            res = new IntStream;
            *res = mul(*(get_intst(inst.r1)),
                       inst.imm);
            set_reg(inst.rd, res);
            break;
        }
        case inst_subi: {
            // Add two int streams together
            res = new IntStream;
            *res = sub1(*(get_intst(inst.r1)),
                       inst.imm);
            set_reg(inst.rd, res);
            break;
        }
        case inst_subii: {
            // Add two int streams together
            res = new IntStream;
            *res = sub2(*(get_intst(inst.r1)),
                       inst.imm);
            set_reg(inst.rd, res);
            break;
        }
        case inst_divi: {
            // Add two int streams together
            res = new IntStream;
            *res = div1(*(get_intst(inst.r1)),
                       inst.imm);
            set_reg(inst.rd, res);
            break;
        }
        case inst_divii: {
            // Add two int streams together
            res = new IntStream;
            *res = div2(*(get_intst(inst.r1)),
                       inst.imm);
            set_reg(inst.rd, res);
            break;
        }
        case inst_modi: {
            // Add two int streams together
            res = new IntStream;
            *res = mod1(*(get_intst(inst.r1)),
                       inst.imm);
            set_reg(inst.rd, res);
            break;
        }
        case inst_modii: {
            // Add two int streams together
            res = new IntStream;
            *res = mod2(*(get_intst(inst.r1)),
                       inst.imm);
            set_reg(inst.rd, res);
            break;
        }
        case inst_default: {
            // Add two int streams together
            res = new IntStream;
            *res = def(inst.imm);
            set_reg(inst.rd, res);
            break;
        }
        case inst_load:
            // No device to load to. Ignore.
            break;
        case inst_load4:
            // No device to load to. Ignore.
            break;
        case inst_load6:
            // No device to load to. Ignore.
            break;
        case inst_load8:
            // No device to load to. Ignore.
            break;
        case inst_store:
            // No device to store from. Ignore.
            break;
        case inst_free: {
            // Don't touch free. It's eeeevil
            break;
        }
        case inst_unit: {
            // Add two int streams together
            res = new UnitStream;
            *res = unit();
            set_reg(inst.rd, res);
            break;
        }
        case inst_exit:
            // Terminate the program
            return false;
        default:
            // Unreachable code
            assert(false);
    }
    return true;
}

void SequentialScheduler::set_reg(size_t pos, Stream* stream) {
    registers[pos] = stream;
}

IntStream* SequentialScheduler::get_intst(size_t reg) {
    assert(registers.find(reg) != registers.end());
    IntStream* intst = static_cast<IntStream *>(registers[reg]);
    return intst;
}

UnitStream* SequentialScheduler::get_ust(size_t reg) {
    assert(registers.find(reg) != registers.end());
    UnitStream* ust = static_cast<UnitStream *>(registers[reg]);
    return ust;
}

Stream* SequentialScheduler::get_st(size_t reg) {
    assert(registers.find(reg) != registers.end());
    Stream* st = registers[reg];
    return st;
}

void SequentialScheduler::warmup(std::string in_file) {
    Reader reader(in_file);
    // Implement something smarter than busy waiting
    while (!instrInterface.ioReady);
    IOStream current;
    while (instrInterface.iostream_pending()) {
        current = instrInterface.pop_iostream();
        if (current.direction == io_in) {
            // If current IO stream is an input stream, read the data
            if (current.type == io_integer) {
                IntStream* stream = new IntStream;
                *stream = reader.getIntStream(current.name);
                set_reg(current.regname, stream);
            } else if (current.type == io_unit) {
                UnitStream* stream = new UnitStream;
                *stream = reader.getUnitStream(current.name);
                set_reg(current.regname, stream);
            } else {
                // Unreachable code
                assert(false);
            }
        } else if (current.direction == io_out) {
            // If it is an output stream, save it for later
            out_strems.push_back(current);
        } else {
            // Unreachable code
            assert(false);
        }
    }
}