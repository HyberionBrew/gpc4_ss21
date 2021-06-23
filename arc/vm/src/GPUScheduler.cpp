//
// Created by fabian on 31/05/2021.
//

#include "GPUScheduler.h"
#include <GPUReader.cuh>
#include <iostream>
#include <cassert>

GPUScheduler::GPUScheduler(InstrInterface & interface) : Scheduler(interface) {
}

bool GPUScheduler::next() {
    Instruction inst = instrInterface.pop();
    switch (inst.type) {
        case inst_add:
            std::cout << ": Add, R1: " << inst.r1 << " R2: " << inst.r2 << " RD: " << inst.rd << std::endl;
            break;
        case inst_mul:
            std::cout << ": Mul, R1: " << inst.r1 << " R2: " << inst.r2 << " RD: " << inst.rd << std::endl;
            break;;
        case inst_sub:
            std::cout << ": Sub, R1: " << inst.r1 << " R2: " << inst.r2 << " RD: " << inst.rd << std::endl;
            break;
        case inst_div:
            std::cout << ": Div, R1: " << inst.r1 << " R2: " << inst.r2 << " RD: " << inst.rd << std::endl;
            break;
        case inst_mod:
            std::cout << ": Mod, R1: " << inst.r1 << " R2: " << inst.r2 << " RD: " << inst.rd << std::endl;
            break;
        case inst_delay:
            std::cout << ": Delay, R1: " << inst.r1 << " R2: " << inst.r2 << " RD: " << inst.rd << std::endl;
            break;
        case inst_last:
            std::cout << ": Last, R1: " << inst.r1 << " R2: " << inst.r2 << " RD: " << inst.rd << std::endl;
            break;
        case inst_time:
            std::cout << ": Time, R1: " << inst.r1 << " RD: " << inst.rd << std::endl;
            break;
        case inst_merge:
            std::cout << ": Merge, R1: " << inst.r1 << " R2: " << inst.r2 << " RD: " << inst.rd << std::endl;
            break;
        case inst_count:
            std::cout << ": Count, R1: " << inst.r1 << " RD: " << inst.rd << std::endl;
            break;
        case inst_addi:
            std::cout << ": AddI, R1: " << inst.r1 << " IMM: " << inst.imm << " RD: " << inst.rd << std::endl;
            break;
        case inst_muli:
            std::cout << ": MulI, R1: " << inst.r1 << " IMM: " << inst.imm << " RD: " << inst.rd << std::endl;
            break;
        case inst_subi:
            std::cout << ": SubI, R1: " << inst.r1 << " IMM: " << inst.imm << " RD: " << inst.rd << std::endl;
            break;
        case inst_subii:
            std::cout << ": SubII, R1: " << inst.r1 << " IMM: " << inst.imm << " RD: " << inst.rd << std::endl;
            break;
        case inst_divi:
            std::cout << ": DivI, R1: " << inst.r1 << " IMM: " << inst.imm << " RD: " << inst.rd << std::endl;
            break;
        case inst_divii:
            std::cout << ": DivII, R1: " << inst.r1 << " IMM: " << inst.imm << " RD: " << inst.rd << std::endl;
            break;
        case inst_modi:
            std::cout << ": ModI, R1: " << inst.r1 << " IMM: " << inst.imm << " RD: " << inst.rd << std::endl;
            break;
        case inst_modii:
            std::cout << ": ModII, R1: " << inst.r1 << " IMM: " << inst.imm << " RD: " << inst.rd << std::endl;
            break;
        case inst_default:
            std::cout << ": Default, R1: " << inst.r1 << " IMM: " << inst.imm << " RD: " << inst.rd << std::endl;
            break;
        case inst_load: {
            // Load stream on device
            shared_ptr<GPUIntStream> GPUIntStream;
            shared_ptr<GPUUnitStream> unitStream;
            if ((GPUIntStream = get_intst(inst.r1)) != nullptr) {
                // Load GPUIntStream
                GPUIntStream->copy_to_device();
            } else if ((unitStream = get_ust(inst.r1)) != nullptr) {
                // Load unitstream
                unitStream->copy_to_device();
            } else {
                assert(false);
            }
            break;
        }
        case inst_load4:
            // Look ahead loading not implemented
            break;
        case inst_load6:
            // Look ahead loading not implemented
            break;
        case inst_load8:
            // Look ahead loading not implemented
            break;
        case inst_store:{
            // Download stream from device
            shared_ptr<GPUIntStream> GPUIntStream;
            shared_ptr<GPUUnitStream> unitStream;
            if ((GPUIntStream = get_intst(inst.r1)) != nullptr) {
                // Load GPUIntStream
                GPUIntStream->copy_to_host();
            } else if ((unitStream = get_ust(inst.r1)) != nullptr) {
                // Load unitstream
                unitStream->copy_to_host();
            } else {
                assert(false);
            }
            break;
        }
        case inst_free: {
            // Download stream from device
            shared_ptr<GPUIntStream> GPUIntStream;
            shared_ptr<GPUUnitStream> unitStream;
            if ((GPUIntStream = get_intst(inst.r1)) != nullptr) {
                // Load GPUIntStream
                GPUIntStream->free_device();
            } else if ((unitStream = get_ust(inst.r1)) != nullptr) {
                // Load unitstream
                unitStream->free_device();
            } else {
                assert(false);
            }
            break;
        }
        case inst_unit:
            std::cout << ": Unit, R1: " << inst.r1 << std::endl;
            break;
        case inst_exit:
            std::cout << ": Exit" << std::endl;
            return false;
        default:
            // Unreachable code
            assert(false);
    }
    return true;}

void GPUScheduler::warmup(std::string in_file) {
    // Initialize the file reader
    GPUReader reader(in_file);

    // Wait for the instruction interface to be ready (it's probably faster than us anyway)
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

void GPUScheduler::cooldown(std::string outfile) {
}

void GPUScheduler::set_reg(size_t pos, shared_ptr<GPUIntStream> stream) {
    intRegisters[pos] = stream;
    unitRegisters[pos] = nullptr;
}

void GPUScheduler::set_reg(size_t pos, shared_ptr<GPUUnitStream> stream) {
    unitRegisters[pos] = stream;
    intRegisters[pos] = nullptr;
}

shared_ptr<GPUIntStream> GPUScheduler::get_intst(size_t reg) {
    if (intRegisters.find(reg) != intRegisters.end()) {
        return intRegisters[reg];
    } else {
        return nullptr;
    }
}

shared_ptr<GPUUnitStream> GPUScheduler::get_ust(size_t reg) {
    if (unitRegisters.find(reg) != unitRegisters.end()) {
        return unitRegisters[reg];
    } else {
        return nullptr;
    }
}