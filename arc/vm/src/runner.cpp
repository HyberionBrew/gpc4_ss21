//
// Created by klaus on 24.06.21.
//

#include "runner.h"
#include "Decode.h"
#include "DebugScheduler.h"
#include "GPUScheduler.h"
#include "SequentialScheduler.h"
#include <iostream>
#include <unistd.h>
#include <thread>
#include <functional>
#include <sstream>
#include <cassert>
#include <time.h>

std::string usage_string = "Usage: arc [-D | -t | -s | -o FILENAME] -v coil_file input_file";
constexpr scheduler DEFAULT_SCHEDULER = debug;

void decode (InstrInterface & interface, std::string coil_file, bool verbose) {

    // Initialize the decoder
    Decode* decodeprt = nullptr;
    try {
        decodeprt = new Decode(coil_file, interface);
    } catch (std::exception &ex) {
        std::cout << "Ouch! Something went wrong: " << ex.what() << std::endl;
        std::cout << "Exiting." << std::endl;
        std::terminate();
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
            std::terminate();
        }
    }
}

void schedule (InstrInterface & interface, scheduler type, std::string in_file, std::string outfile, bool verbose) {
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
        std::cout << "Warming up on input file " << in_file << std::endl;
    }

    clock_t warmup_time = clock();
    try {
        scheduler->warmup(in_file);
        if (verbose) {
            std::cout << "Warmup complete.\n";
        }
    } catch (std::exception &ex) {
        std::cout << "Ouch! Something went wrong: " << ex.what() << std::endl;
        std::cout << "Could not warm up on input file " << in_file << std::endl;
        std::cout << "Exiting." << std::endl;
        std::terminate();
    }
    warmup_time = clock() - warmup_time;
    if (verbose) {
        std::cout << "Took " << static_cast<float>(warmup_time) / 1000000 << "s to do so.\n";
        std::cout << "Starting calculations...\n";
    }

    clock_t calc_time = clock();
    while (scheduler->next()) {}
    calc_time = clock() - calc_time;

    if (verbose) {
        std::cout << "Finished calculations.\n";
        std::cout << "Took " << static_cast<float>(calc_time) / 1000000 << "s to do so.\n";
        std::cout << "Writing output file.\n";
    }
    clock_t cooldown_time = clock();
    scheduler->cooldown(outfile);
    cooldown_time = clock() - cooldown_time;
    if (verbose) {
        std::cout << "Output written to " << outfile << "\n";
        std::cout << "Took " << static_cast<float>(cooldown_time) / 1000000 << "s to do so.\n";
        std::cout << "\n";
        std::cout << "Done.\n";
        std::cout << "Took " << static_cast<float>(warmup_time + calc_time + cooldown_time) / 1000000
            << "s (Warmup: " << static_cast<float>(warmup_time) / 1000000
            << "s, Calculations: " << static_cast<float>(calc_time) / 1000000
            << "s, Cooldown: " << static_cast<float>(cooldown_time) / 1000000 << "s) to do so.\n";
        std::cout << "\n";
    }
}

int run(int argc, char **argv) {
    // Check for right number of arguments
    if (argc < 3) {
        std::cout << "Not enough arguments provided." << std::endl;
        std::cout << usage_string << std::endl;
        std::cout << "Exiting." << std::endl;
        std::terminate();
    }

    // Read in command line arguments
    scheduler scheduler = DEFAULT_SCHEDULER;
    const char *optstring = "Dtsvo:";
    bool verbose = false;
    char opt;

    // Use the arguments provided as coil and input file names
    std::string coil_file = argv[argc - 2];
    std::string input_file = argv[argc - 1];

    // Construct output file name
    std::string output_file = input_file.substr(0, input_file.find_last_of('.'));
    output_file.append(".out");

    while ((opt = getopt(argc, argv, optstring)) != -1) {
        // Set the right scheduler/execution method
        if (opt == 'D' || opt == 't' || opt == 's') {
            if (scheduler != DEFAULT_SCHEDULER) {
                std::cout << "Scheduler set twice." << std::endl;
                std::cout << usage_string << std::endl;
                std::cout << "Exiting." << std::endl;
                std::terminate();
            }
        }
        switch (opt) {
            case 'D':
                scheduler = debug;
                break;
            case 't':
                scheduler = thrust;
                break;
            case 's':
                scheduler = sequential;
                break;
            case 'v':
                // Set verbose mode
                verbose = true;
                break;
            case 'o':
                // Set custom outfile
                output_file = optarg;
                break;
            case '?':
                if (optopt == 'o') {
                    std::cout << "Option \'o\' requires an option." << std::endl;
                } else {
                    std::cout << "Unknown option \'" << optopt << "\'." << std::endl;
                }
                std::cout << usage_string << std::endl;
                std::cout << "Exiting." << std::endl;
                std::terminate();
            default:
                assert(false);
        }
    }


    InstrInterface interface;

    // Thread out
    std::thread decoder_th(decode, std::ref(interface), coil_file, verbose);
    std::thread scheduler_th(schedule, std::ref(interface), scheduler, input_file, output_file, verbose);

    // Wait till both threads complete
    decoder_th.join();
    scheduler_th.join();

    return 0;
}
