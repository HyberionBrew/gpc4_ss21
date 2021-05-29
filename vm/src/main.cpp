//
// Created by fabian on 27/05/2021.
//

#include "main.h"
#include "Decode.h"
#include <iostream>
#include <string>

std::string usage_string = "Usage: arc coil_file input_file";

int main(int argc, char **argv) {
    // Check for right number of arguments
    if (argc < 3) {
        std::cout << "Not enough arguments provided." << std::endl;
        std::cout << usage_string << std::endl;
        std::cout << "Exiting." << std::endl;
        exit(1);
    }
    if (argc > 3) {
        std::cout << "Too many arguments provided." << std::endl;
        std::cout << usage_string << std::endl;
        std::cout << "Exiting." << std::endl;
        exit(1);
    }
    // Else use the arguments provided as coil and input file names
    std::string coil_file = argv[1];
    std::string input_file = argv[2];

    // Initialize the decoder
    Decode* decodeprt = nullptr;
    try {
        decodeprt = new Decode(coil_file);
    } catch (std::exception &ex) {
        std::cout << "Ouch! Something went wrong: " << ex.what() << std::endl;
        std::cout << "Exiting." << std::endl;
        exit(1);
    }
    decodeprt->decode_next();
    return 0;
}