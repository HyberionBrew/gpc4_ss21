//
// Created by fabian on 27/05/2021.
//

#include "main.h"
#include "Decode.h"
#include "DebugScheduler.h"
#include "GPUScheduler.h"
#include "SequentialScheduler.h"
#include <iostream>
#include <unistd.h>
#include <thread>
#include <functional>
#include <sstream>

std::string usage_string = "Usage: arc [-D | -t | -s] -v coil_file input_file";
constexpr scheduler DEFAULT_SCHEDULER = debug;

void decode (InstrInterface & interface, std::string coil_file, bool verbose) {

    // Initialize the decoder
    Decode* decodeprt = nullptr;
    try {
        decodeprt = new Decode(coil_file, interface);
    } catch (std::exception &ex) {
        std::cout << "Ouch! Something went wrong: " << ex.what() << std::endl;
        std::cout << "Exiting." << std::endl;
        exit(1);
    }

    if (verbose) {
        decodeprt->print_header();
    }

    bool next = true;
    while (next) {
        next = false;
        try {
            next = decodeprt->decode_next();
        } catch (std::exception &ex) {
            std::cout << "Ouch! Something went wrong: " << ex.what() << std::endl;
            std::cout << "Exiting." << std::endl;
            exit(1);
        }
    }
}

void schedule (InstrInterface & interface, scheduler type, std::string in_file, bool verbose) {
    std::stringstream str;

    if (verbose) {
        str << "Computing using ";
    }
    Scheduler *scheduler;
    switch (type) {
        case debug:
            scheduler = new DebugScheduler(interface);
            if (verbose) str << "Debug";
            break;
        case gpu:
            scheduler = new GPUScheduler(interface);
            if (verbose) str << "GPU";
            break;
        case sequential:
            scheduler = new SequentialScheduler(interface);
            if (verbose) str << "Sequential";
            break;
        default:
            std::cout << "Required scheduler not yet implemented.\n";
            return;
    }
    if (verbose) {
        str << " scheduler." << std::endl;
        std::cout << str.str();
    }
    if (verbose) {
        std::cout << "Warming up on input file " << in_file << std::endl;
    }
    try {
        scheduler->warmup(in_file);
        if (verbose) {
            std::cout << "Warmup complete.\n";
        }
    } catch (std::exception &ex) {
        std::cout << "Ouch! Something went wrong: " << ex.what() << std::endl;
        std::cout << "Could not warm up on input file " << in_file << std::endl;
        std::cout << "Exiting." << std::endl;
        exit(1);
    }
    if (verbose) {
        std::cout << "Starting calculations...\n";
    }
    while (scheduler->next()) {}
}

int main(int argc, char **argv) {
    // Check for right number of arguments
    if (argc < 3) {
        std::cout << "Not enough arguments provided." << std::endl;
        std::cout << usage_string << std::endl;
        std::cout << "Exiting." << std::endl;
        exit(1);
    }

    // Read in command line arguments
    scheduler scheduler = DEFAULT_SCHEDULER;
    const char * optstring = "Dtsvc:i:";
    bool verbose = false;
    char opt;
    while ((opt = getopt(argc, argv, optstring)) != -1) {
        // Set the right scheduler/execution method
        if (opt == 'D' || opt == 't' || opt == 's') {
            if (scheduler != DEFAULT_SCHEDULER) {
                std::cout << "Scheduler set twice." << std::endl;
                std::cout << usage_string << std::endl;
                std::cout << "Exiting." << std::endl;
                exit(1);
            }
        }
        if (opt == 'D') {
            scheduler = debug;
        } else if (opt == 't') {
            scheduler = thrust;
        } else if (opt == 's') {
            scheduler = sequential;
        } else if (opt == 'v') {
            // Set verbose mode
            verbose = true;
        } else {
            std::cout << "Unknown argument." << std::endl;
            std::cout << usage_string << std::endl;
            std::cout << "Exiting." << std::endl;
            exit(1);
        }
    }

    // Else use the arguments provided as coil and input file names
    std::string coil_file = argv[argc-2];
    std::string input_file = argv[argc-1];

    InstrInterface interface;

    // Thread out
    std::thread decoder_th (decode, std::ref(interface), coil_file, verbose);
    std::thread scheduler_th (schedule, std::ref(interface), scheduler, input_file, verbose);

    // Wait till both threads complete
    decoder_th.join();
    scheduler_th.join();

    return 0;
}