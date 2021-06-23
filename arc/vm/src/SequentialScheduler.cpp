//
// Created by fabian on 11/06/2021.
//

#include "SequentialScheduler.h"
#include <iostream>
#include <assert.h>
#include <Reader.h>
#include <StreamFunctions.h>
#include <Debug.h>

SequentialScheduler::SequentialScheduler(InstrInterface & interface) : Scheduler(interface) {
    line = 0;
}

bool SequentialScheduler::next() {
    line++;
    Instruction inst = instrInterface.pop();
    switch (inst.type) {
        case inst_add: {
            // Add two int streams together
            set_reg(inst.rd, add(*(get_intst(inst.r1)), *(get_intst(inst.r2))));
            break;
        }
        case inst_mul: {
            // Add two int streams together
            set_reg(inst.rd, mul(*(get_intst(inst.r1)), *(get_intst(inst.r2))));
            break;
        }
        case inst_sub: {
            // Add two int streams together
            set_reg(inst.rd, sub(*(get_intst(inst.r1)), *(get_intst(inst.r2))));
            break;
        }
        case inst_div: {
            // Add two int streams together
            set_reg(inst.rd, div(*(get_intst(inst.r1)), *(get_intst(inst.r2))));
            break;
        }
        case inst_mod: {
            // Add two int streams together
            set_reg(inst.rd, mod(*(get_intst(inst.r1)), *(get_intst(inst.r2))));
            break;
        }
        case inst_delay: {
            // Add two int streams together
            set_reg(inst.rd, delay(*(get_intst(inst.r1)), *(get_st(inst.r2))));
            break;
        }
        case inst_last: {
            // Add two int streams together
            set_reg(inst.rd, last(*(get_intst(inst.r1)), *(get_st(inst.r2))));
            break;
        }
        case inst_time: {
            // Add two int streams together
            set_reg(inst.rd, time(*(get_st(inst.r1))));
            break;
        }
        case inst_merge: {
            // Add two int streams together
            shared_ptr<IntStream> in1 = get_intst(inst.r1);
            shared_ptr<IntStream> in2 = get_intst(inst.r2);
            shared_ptr<UnitStream> un1 = get_ust(inst.r1);
            shared_ptr<UnitStream> un2 = get_ust(inst.r2);

            if (in1 != nullptr && in2 != nullptr) {
                set_reg(inst.rd, merge(*in1, *in2));
            } else if (un1 != nullptr && un2 != nullptr) {
                set_reg(inst.rd, merge(*un1, *un2));
            } else {
                assert(false);
            }
            break;
        }
        case inst_count: {
            // Add two int streams together
            set_reg(inst.rd, count(*(get_ust(inst.r1))));
            break;
        }
        case inst_addi: {
            // Add two int streams together
            set_reg(inst.rd, add(*(get_intst(inst.r1)), inst.imm));
            break;
        }
        case inst_muli: {
            // Add two int streams together
            set_reg(inst.rd, mul(*(get_intst(inst.r1)), inst.imm));
            break;
        }
        case inst_subi: {
            // Add two int streams together
            set_reg(inst.rd, sub1(*(get_intst(inst.r1)), inst.imm));
            break;
        }
        case inst_subii: {
            // Add two int streams together
            set_reg(inst.rd, sub2(*(get_intst(inst.r1)), inst.imm));
            break;
        }
        case inst_divi: {
            // Add two int streams together
            set_reg(inst.rd, div1(*(get_intst(inst.r1)), inst.imm));
            break;
        }
        case inst_divii: {
            // Add two int streams together
            set_reg(inst.rd, div2(*(get_intst(inst.r1)), inst.imm));
            break;
        }
        case inst_modi: {
            // Add two int streams together
            set_reg(inst.rd, mod1(*(get_intst(inst.r1)), inst.imm));
            break;
        }
        case inst_modii: {
            // Add two int streams together
            set_reg(inst.rd, mod2(*(get_intst(inst.r1)), inst.imm));
            break;
        }
        case inst_default: {
            // Add two int streams together
            set_reg(inst.rd, def(inst.imm));
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
            // Remove register from map
            shared_ptr<IntStream> is = get_intst(inst.r1);
            shared_ptr<UnitStream> us = get_ust(inst.r1);
            bool done = false;
            if (is != nullptr) {
                intRegisters.erase(inst.r1);
                done = !done;
            }
            if (us != nullptr) {
                unitRegisters.erase(inst.r1);
                done = !done;
            }
            assert(done);
            break;
        }
        case inst_unit: {
            // Add two int streams together
            set_reg(inst.rd, unit());
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

void SequentialScheduler::set_reg(size_t pos, shared_ptr<IntStream> stream) {
    intRegisters[pos] = stream;
    unitRegisters[pos] = nullptr;
}

void SequentialScheduler::set_reg(size_t pos, shared_ptr<UnitStream> stream) {
    unitRegisters[pos] = stream;
    intRegisters[pos] = nullptr;
}

shared_ptr<IntStream> SequentialScheduler::get_intst(size_t reg) {
    if (intRegisters.find(reg) != intRegisters.end()) {
        return intRegisters[reg];
    } else {
        return nullptr;
    }
}

shared_ptr<UnitStream> SequentialScheduler::get_ust(size_t reg) {
    if (unitRegisters.find(reg) != unitRegisters.end()) {
        return unitRegisters[reg];
    } else {
        return nullptr;
    }
}

shared_ptr<Stream> SequentialScheduler::get_st(size_t reg) {
    shared_ptr<Stream> stream;
    if (intRegisters.find(reg) == intRegisters.end()) {
        if (unitRegisters.find(reg) == unitRegisters.end()) {
            assert(false);
        } else {
            stream = static_pointer_cast<Stream>(unitRegisters[reg]);
        }
    } else {
        stream = static_pointer_cast<Stream>(intRegisters[reg]);
    }
    return stream;
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
                set_reg(current.regname, reader.getIntStream(current.name));
            } else if (current.type == io_unit) {
                set_reg(current.regname, reader.getUnitStream(current.name));
            } else {
                // Unreachable code
                assert(false);
            }
        } else if (current.direction == io_out) {
            // If it is an output stream, save it for later
            out_streams.push_back(current);
        } else {
            // Unreachable code
            assert(false);
        }
    }
}

void SequentialScheduler::cooldown(std::string outfile) {
    Writer writer(outfile);
    Stream* stream_ptr;
    for (auto & stream : out_streams) {
        if (get_ust(stream.regname) != nullptr) {
            // Add unit stream
            writer.addUnitStream(stream.name, get_ust(stream.regname));
        } else if (get_intst(stream.regname) != nullptr) {
            // Add integer stream
            writer.addIntStream((stream.name), get_intst(stream.regname));
        } else {
            assert(false);
        }
    }
    writer.writeOutputFile();
}