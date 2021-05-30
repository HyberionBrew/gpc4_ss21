//
// Created by daniel on 27.05.21.
//

#include "Decode.h"
#include <iostream>
#include <fstream>
#include <assert.h>

// Op types
#define RTYPE 0b00000000;
#define ITYPE 0b01000000;
#define MTYPE 0b10000000;
#define EXIT  0b11000000;

// State machines
enum init_read_states{head_read_spec, head_error, head_unknown_field, head_read_field, head_delim_needed, head_read_string, head_ready, head_accept};
enum init_delim_states{del_stby, delim_started};
enum init_term_states{term_stby, term_started};
enum init_spec_field_states{spec_stby, spec_h1, spec_h2, spec_h3, spec_h4, spec_v1, spec_v2, spec_v3};
enum init_reg_bytel_states{reg_stby, reg_h1, reg_h2, reg_h3, reg_h4};
enum init_instream_states{inst_stby, inst_h1, inst_h2, inst_h3, inst_h4, inst_r1, inst_r2, inst_r3, inst_r4};
enum init_outstream_states{outst_stby, outst_h1, outst_h2, outst_h3, outst_h4, outst_r1, outst_r2, outst_r3, outst_r4};


Decode::Decode(std::string coil_file) {
    coil.open(coil_file, std::ios::binary);
    // Make sure the file exists
    if (coil.fail()) {
        // It does not. Exit.
        throw std::runtime_error("Could not open coil file.");
    }
    // Start reading the header configuration
    std::vector<unsigned char> bytes;
    unsigned char byte;
    // Check the file signature
    for (int i = 0; i < 4; i++) {
        coil >> byte;
        if (coil.fail()) {
            // Unexpected file error
            throw std::runtime_error("Error reading coil file.");
        }
        bytes.push_back(byte);
    }
    if (bytes[0] != 'X' || bytes[1] != 'R' || bytes[2] != 'A' || bytes[3] != 'Y') {
        throw std::runtime_error("Not a coil file.");
    }
    bytes.clear();
    parse_header();
    print_header();
};


void Decode::parse_header() {
    unsigned char byte;
    std::string bytes;
    std::string ill = "Illformed coil file header.";

    // Reader field state machines
    init_read_states readState = head_read_spec;
    init_delim_states delimState = del_stby;
    init_term_states termState = term_stby;
    init_spec_field_states specFieldState = spec_stby;
    init_reg_bytel_states regBytelState = reg_stby;
    init_instream_states instreamState = inst_stby;
    init_outstream_states outstreamState = outst_stby;

    // Check for double setting
    bool bwidth_set = false;

    // Make sure each state machine transitions otherwise reset them
    bool delim_trans = false;
    bool term_trans = false;
    bool specf_trans = false;
    bool regbw_trans = false;
    bool inst_trans = false;
    bool outst_trans = false;

    // Current variables for stream IO
    IOStream* current;

    // Read the actual header fields
    while (readState != head_accept && coil.peek() != EOF) {
        coil >> byte;

        if (coil.fail()) {
            // Unexpected file error
            throw std::runtime_error("Error reading coil file.");
        }

        // Check for reading string
        if (readState == head_read_string) {
            bytes.push_back(byte);
            // In case of finished string, save everything
            if (byte == 0x00) {
                // Save the current stream
                assert(current != nullptr);
                current->name = bytes;
                if (instreamState == inst_r4) {
                    // Add the stream to the input streams
                    in_streams.push_back(*current);
                } else if (outstreamState == outst_r4) {
                    out_streams.push_back(*current);
                } else {
                    throw std::runtime_error("Bad state machine configuration while parsing header. String parsed when no string needed.");
                }
                // Free the current read stream representation
                current = nullptr;
                bytes.clear();
                // Require head field delimiter
                readState = head_delim_needed;
            }
            continue;
        }

        // Header field switch case
        switch (byte) {
            case 0x43:
                if (specFieldState == spec_h3) {
                    specFieldState = spec_h4;
                    specf_trans = true;
                    if (readState != head_read_spec) {
                        throw std::runtime_error(ill + " Specification set more than once.");
                    }
                }
                break;
            case 0x45:
                if (specFieldState == spec_h2) {
                    specFieldState = spec_h3;
                    specf_trans = true;
                }
                if (regBytelState == reg_h1) {
                    regBytelState = reg_h2;
                    regbw_trans = true;
                }
                break;
            case 0x47:
                if (regBytelState == reg_h2) {
                    regBytelState = reg_h3;
                    regbw_trans = true;
                }
                break;
            case 0x49:
                if (readState == head_ready) {
                    readState = head_read_field;
                    instreamState = inst_h1;
                    inst_trans = true;
                }
                break;
            case 0x4C:
                if (regBytelState == reg_h3) {
                    if (bwidth_set) {
                        throw std::runtime_error(ill + " Register byte width set twice.");
                    }
                    regBytelState = reg_h4;
                    regbw_trans = true;
                }
                break;
            case 0x4E:
                if (instreamState == inst_h1) {
                    instreamState = inst_h2;
                    inst_trans = true;
                }
                break;
            case 0x4F:
                if (readState == head_ready) {
                    readState = head_read_field;
                    outstreamState = outst_h1;
                    outst_trans = true;
                }
                break;
            case 0x50:
                if (specFieldState == spec_h1) {
                    specFieldState = spec_h2;
                    specf_trans = true;
                }
                break;
            case 0x52:
                if (readState == head_ready) {
                    readState = head_read_field;
                    regBytelState = reg_h1;
                    regbw_trans = true;
                }
                break;
            case 0x53:
                if (readState == head_read_spec || readState == head_ready) {
                    // Make sure spec version is not double set
                    if (readState == head_ready) {
                        readState = head_read_field;
                    }
                    specFieldState = spec_h1;
                    specf_trans = true;
                }
                if (instreamState == inst_h2) {
                    instreamState = inst_h3;
                    inst_trans = true;
                }
                if (outstreamState == outst_h2) {
                    outstreamState = outst_h3;
                    outst_trans = true;
                }
                break;
            case 0x54:
                if (instreamState == inst_h3) {
                    instreamState = inst_h4;
                    inst_trans = true;
                }
                if (outstreamState == outst_h3) {
                    outstreamState = outst_h4;
                    outst_trans = true;
                }
                break;
            case 0x55:
                if (outstreamState == outst_h1) {
                    outstreamState = outst_h2;
                    outst_trans = true;
                }
            case 0xF0:
                if (readState == head_unknown_field || readState == head_delim_needed) {
                    if (delimState == del_stby) {
                        delimState = delim_started;
                        delim_trans = true;
                    } else {
                        readState = head_ready;
                        delimState = del_stby;
                        delim_trans = true;
                    }
                }
                break;
            case 0xFF:
                if (readState == head_ready) {
                    if (termState == term_stby) {
                        termState = term_started;
                        term_trans = true;
                    } else {
                        readState = head_accept;
                        termState = term_stby;
                        term_trans = true;
                    }
                }
        }

        // Read output stream register address
        if (outstreamState == outst_r3) {
            current->regname = current->regname << 8;
            current->regname = current->regname + byte;
            readState = head_read_string;
            outstreamState = outst_r4;
            outst_trans = true;
        }
        if (outstreamState == outst_r2) {
            assert(current != nullptr);
            current->regname = current->regname << 8;
            current->regname = current->regname + byte;
            outstreamState = outst_r3;
            outst_trans = true;
        }
        if (outstreamState == outst_r1) {
            assert(current != nullptr);
            current->regname = current->regname << 8;
            current->regname = current->regname + byte;
            outstreamState = outst_r2;
            outst_trans = true;
        }
        // Make sure we have not transitioned this cycle and start reading
        if (!outst_trans && outstreamState == outst_h4) {
            current = new IOStream;
            current->regname = byte;
            if (wideAddresses) {
                outstreamState = outst_r1;
            } else {
                outstreamState = outst_r3;
            }
            outst_trans = true;
        }

        // Read input stream register address
        if (instreamState == inst_r3) {
            current->regname = current->regname << 8;
            current->regname = current->regname + byte;
            readState = head_read_string;
            instreamState = inst_r4;
            inst_trans = true;
        }
        if (instreamState == inst_r2) {
            assert(current != nullptr);
            current->regname = current->regname << 8;
            current->regname = current->regname + byte;
            instreamState = inst_r3;
            inst_trans = true;
        }
        if (instreamState == inst_r1) {
            assert(current != nullptr);
            current->regname = current->regname << 8;
            current->regname = current->regname + byte;
            instreamState = inst_r2;
            inst_trans = true;
        }
        // Make sure we have not transitioned this cycle and start reading
        if (!inst_trans && instreamState == inst_h4) {
            current = new IOStream;
            current->regname = byte;
            if (wideAddresses) {
                instreamState = inst_r1;
            } else {
                instreamState = inst_r3;
            }
            inst_trans = true;
        }

        // Make sure we have not transitioned this cycle and read the register byte length
        if (!regbw_trans && regBytelState == reg_h4) {
            if (byte == 0x00) {
                wideAddresses = false;
            } else if (byte == 0x01) {
                wideAddresses = true;
            } else {
                throw std::runtime_error(ill + " Unsupported register width.");
            }
            bwidth_set = true;
            regBytelState = reg_stby;
            readState = head_delim_needed;
            regbw_trans = true;
        }

        // Read the specification info
        if (specFieldState == spec_v3) {
            minorV = minorV << 8;
            minorV = minorV || byte;
            specFieldState = spec_stby;
            readState = head_delim_needed;
            specf_trans = true;
        }
        if (specFieldState == spec_v2) {
            minorV = byte;
            specFieldState = spec_v3;
            specf_trans = true;
        }
        if (specFieldState == spec_v1) {
            majorV = majorV << 8;
            majorV = majorV || byte;
            specFieldState = spec_v2;
            specf_trans = true;
        }
        // Make sure we have not transitioned into the version field this cycle and start reading
        if (!specf_trans && specFieldState == spec_h4) {
            majorV = byte;
            specFieldState = spec_v1;
            specf_trans = true;
        }
        // If we should be reading the specification version but are reading something else, there must be an error
        if (readState == head_read_spec && !specf_trans) {
            throw std::runtime_error(ill + " Specification version not specified.");
        }

        // If one of the parsing state machines has not transitioned this round, the input read must be invalid for it
        // so we reset it to its start state.
        if (!delim_trans) {
            delimState = del_stby;
        }
        if (!term_trans) {
            termState = term_stby;
        }
        if (!specf_trans) {
            specFieldState = spec_stby;
        }
        if (!regbw_trans) {
            regBytelState = reg_stby;
        }
        if (!inst_trans) {
            instreamState = inst_stby;
        }
        if (!outst_trans) {
            outstreamState = outst_stby;
        }

        // If no state machine transitioned this round, the field must be unknown. Wait for delimiter.
        if (!delim_trans && !term_trans && !specf_trans && !regbw_trans && !inst_trans && !outst_trans) {
            if (readState != head_delim_needed) {
                readState = head_unknown_field;
            } else {
                throw std::runtime_error(ill + " Missing field delimiter.");
            }
        }

        // Reset the transition checkers for next round
        delim_trans = false;
        term_trans = false;
        specf_trans = false;
        regbw_trans = false;
        inst_trans = false;
        outst_trans = false;

    }
    if (readState != head_accept) {
        throw std::runtime_error(ill + " Header not accepted.");
    }

    // Warn about newer specification version
    if (majorV > currMV) {
        std::cout << "WARNING! The coil file uses a newer specification version than this engine." << std::endl;
        std::cout << "Some features might not be supported." << std::endl;
    }
}

bool Decode::decode_next() {

    // Read in the next byte
    unsigned char byte;
    coil >> byte;

    if (coil.fail()) {
        // Unexpected file error
        throw std::runtime_error("Error reading coil file.");
    }

    unsigned char msb_mask = 0b11000000;
    unsigned char optype = msb_mask && byte;
    unsigned char opcode = byte;

    // Set register width
    int register_width = 2;
    if (wideAddresses) {
        register_width = 4;
    }

    // If this is an exit instruction, return
    if (!(optype ^ EXIT)) {
        // Operation is EXIT
        return true;
    }

    // Read first register address
    coil >> byte;

    if (coil.fail()) {
        // Unexpected file error
        throw std::runtime_error("Error reading coil file at opcode " << (int)opcode);
    }
    size_t r1 = byte;

    for (int i = 1; i < register_width; i++) {
        coil >> byte;

        if (coil.fail()) {
            // Unexpected file error
            throw std::runtime_error("Error reading coil file at opcode " << (int)opcode);
        }
        r1 = r1 << 8;
        r1 += byte;
    }

    if (!(optype ^ RTYPE)) {
        // Operation is R-Type
        // Read second register address
        coil >> byte;

        if (coil.fail()) {
            // Unexpected file error
            throw std::runtime_error("Error reading coil file at opcode " << (int)opcode);
        }
        size_t r2 = byte;

        for (int i = 1; i < register_width; i++) {
            coil >> byte;

            if (coil.fail()) {
                // Unexpected file error
                throw std::runtime_error("Error reading coil file at opcode " << (int)opcode);
            }
            r2 = r2 << 8;
            r2 += byte;
        }
        // Read return register address
        coil >> byte;

        if (coil.fail()) {
            // Unexpected file error
            throw std::runtime_error("Error reading coil file at opcode " << (int)opcode);
        }
        size_t rd = byte;

        for (int i = 1; i < register_width; i++) {
            coil >> byte;

            if (coil.fail()) {
                // Unexpected file error
                throw std::runtime_error("Error reading coil file at opcode " << (int)opcode);
            }
            rd = rd << 8;
            rd += byte;
        }

        switch (opcode) {
            case 0x08:
                // Add
                break;
            case 0x09:
                // Mul
                break;
            case 0x0A:
                // Sub
                break;
            case 0x0B:
                // Div
                break;
            case 0x0C:
                // Mod
                break;
            case 0x0D:
                // Delay
                break;
            case 0x0E:
                // Last
                break;
            case 0x0F:
                // TimeIn
                break;
            case 0x10:
                // TimeUn
                break;
            case 0x11:
                // MergeIn
                break;
            case 0x12:
                // MergeUn
                break;
            case 0x13:
                // Count
                break;
            default:
                // Unknown instruction type
                throw std::runtime_error("Error at opcode " << (int)opcode << ". Unknown instruction type.");
        }

    } else if (!(optype ^ ITYPE)) {
        // Operation is I-Type
    } else if (!(optype ^ MTYPE)) {
        // Operation is M-Type
    } else {
        // Unreachable code. Something weird happened.
        assert(false);
    }
    return 0;
}

void queue_instruction() {
    // TODO write instruction enum and implement this
}

void Decode::print_header() {
    std::cout << "Header for current coil file:" << std::endl;
    std::cout << "Operating on specification version " << majorV << "." << minorV << std::endl;
    if (wideAddresses) {
        std::cout << "4 Byte register addresses." << std::endl;
    } else {
        std::cout << "2 Byte register addresses." << std::endl;
    }
    std::cout << "Input streams:" << std::endl;
    for (std::vector<IOStream>::iterator current = in_streams.begin(); current != in_streams.end(); current++) {
        std::cout << current->name << " in pseudo register " << current->regname << std::endl;
    }
    std::cout << "Output streams:" << std::endl;
    for (std::vector<IOStream>::iterator current = out_streams.begin(); current != out_streams.end(); current++) {
        std::cout << current->name << " in pseudo register " << current->regname << std::endl;
    }
}