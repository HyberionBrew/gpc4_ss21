//
// Created by fabian on 31/05/2021.
//

#include "ThrustScheduler.h"
#include <GPUReader.cuh>
#include <StreamFunctionsThrust.cuh>
#include <ImmediateFunctionsThrust.cuh>
#include <iostream>
#include <cassert>
#include <GPUWriter.cuh>

ThrustScheduler::ThrustScheduler(InstrInterface & interface) : Scheduler(interface) {
}

bool ThrustScheduler::next() {
    Instruction inst = instrInterface.pop();
    switch (inst.type) {
        case inst_add:
            set_reg(inst.rd, slift_thrust(get_intst(inst.r1), get_intst(inst.r2), TH_OP_add, 0));
            break;
        case inst_mul:
            set_reg(inst.rd, slift_thrust(get_intst(inst.r1), get_intst(inst.r2), TH_OP_multiply, 0));
            break;;
        case inst_sub:
            set_reg(inst.rd, slift_thrust(get_intst(inst.r1), get_intst(inst.r2), TH_OP_subtract, 0));
            break;
        case inst_div:
            set_reg(inst.rd, slift_thrust(get_intst(inst.r1), get_intst(inst.r2), TH_OP_divide, 0));
            break;
        case inst_mod:
            set_reg(inst.rd, slift_thrust(get_intst(inst.r1), get_intst(inst.r2), TH_OP_modulo, 0));
            break;
        case inst_delay:
            set_reg(inst.rd, delay_thrust(get_intst(inst.r1), get_ust(inst.r2), 0));
            break;
        case inst_last:
            set_reg(inst.rd, last_thrust(get_intst(inst.r1), get_ust(inst.r2), 0));
            break;
        case inst_time:
            //set_reg(inst.rd, time(get_intst(inst.r1), 0));
            break;
        case inst_merge:
            set_reg(inst.rd, slift_thrust(get_intst(inst.r1), get_intst(inst.r2), TH_OP_merge, 0));
            break;
        case inst_count:
            set_reg(inst.rd, count_thrust(get_ust(inst.r1)));
            break;
        case inst_addi:
            set_reg(inst.rd, add_imm_thrust(get_intst(inst.r1), inst.imm));
            break;
        case inst_muli:
            set_reg(inst.rd, mul_imm_thrust(get_intst(inst.r1), inst.imm));
            break;
        case inst_subi:
            // stream - imm
            set_reg(inst.rd, sub_imm_thrust(get_intst(inst.r1), inst.imm));
            break;
        case inst_subii:
            // imm - stream
            set_reg(inst.rd, sub_inv_imm_thrust(get_intst(inst.r1), inst.imm));
            break;
        case inst_divi:
            // stream / imm
            set_reg(inst.rd, div_imm_thrust(get_intst(inst.r1), inst.imm));
            break;
        case inst_divii:
            // imm / stream
            set_reg(inst.rd, div_inv_imm_thrust(get_intst(inst.r1), inst.imm));
            break;
        case inst_modi:
            // stream % imm
            set_reg(inst.rd, mod_imm_thrust(get_intst(inst.r1), inst.imm));;
            break;
        case inst_modii:
            // imm % stream
            set_reg(inst.rd, mod_inv_imm_thrust(get_intst(inst.r1), inst.imm));
            break;
        case inst_default:
            //std::cout << ": Default, R1: " << inst.r1 << " IMM: " << inst.imm << " RD: " << inst.rd << std::endl;
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
            // Terminate the program
            return false;
        default:
            // Unreachable code
            assert(false);
    }
    return true;}

void ThrustScheduler::warmup(std::string in_file) {
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

void ThrustScheduler::cooldown(std::string outfile) {
    GPUWriter writer(outfile);
    for (auto &stream : out_streams) {
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

void ThrustScheduler::set_reg(size_t pos, shared_ptr<GPUIntStream> stream) {
    intRegisters[pos] = stream;
    unitRegisters[pos] = nullptr;
}

void ThrustScheduler::set_reg(size_t pos, shared_ptr<GPUUnitStream> stream) {
    unitRegisters[pos] = stream;
    intRegisters[pos] = nullptr;
}

shared_ptr<GPUIntStream> ThrustScheduler::get_intst(size_t reg) {
    if (intRegisters.find(reg) != intRegisters.end()) {
        return intRegisters[reg];
    } else {
        return nullptr;
    }
}

shared_ptr<GPUUnitStream> ThrustScheduler::get_ust(size_t reg) {
    if (unitRegisters.find(reg) != unitRegisters.end()) {
        return unitRegisters[reg];
    } else {
        return nullptr;
    }
}