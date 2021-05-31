//
// Created by fabian on 27/05/2021.
//

#include "main.h"
#include "Decode.h"
#include "Scheduler.h"
#include "DebugScheduler.h"
#include "GPUScheduler.h"
#include "InstrInterface.h"
#include <iostream>
#include <string>
#include <unistd.h>
#include <thread>

std::string usage_string = "Usage: arc [-D | -t | -s] -v coil_file input_file";
enum scheduler{debug, sequential, gpu, thrust};
constexpr scheduler DEFAULT_SCHEDULER = gpu;

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
    char opt;
    bool verbose = false;
    while (getopt(argc, argv, &opt) != -1) {
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
    std::thread decoder_th (decode, &interface, verbose);
    std::thread scheduler_th (schedule, &interface, verbose);

    // Wait till both threads complete
    decoder_th.join();
    scheduler_th.join();

    return 0;
}

void decode (InstrInterface* interface, bool verbose) {
    // Initialize the decoder
    Decode* decodeprt = nullptr;
    try {
        decodeprt = new Decode(coil_file);
    } catch (std::exception &ex) {
        std::cout << "Ouch! Something went wrong: " << ex.what() << std::endl;
        std::cout << "Exiting." << std::endl;
        exit(1);
    }

    if (verbose) {
        decodeprt->print_header();
    }

    bool next = false;
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

void schedule (InstrInterface* interface, scheduler type, bool verbose) {
    Scheduler* scheduler;
    switch (type) {
        case debug:
            scheduler = new DebugScheduler(interface);
            break;
        case gpu:
            scheduler = new GPUScheduler(interface);
            break;
        default:
            std::cout << "Required scheduler not yet implemented." << std::endl;
            return;
    }
    while (scheduler->next()) {}
}