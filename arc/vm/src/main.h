//
// Created by fabian on 27/05/2021.
//

#ifndef GPC4_SS21_MAIN_H
#define GPC4_SS21_MAIN_H

#include "InstrInterface.h"
#include "Scheduler.h"
#include <string>

enum scheduler{debug, sequential, gpu, thrust};

int main(int argc, char **argv);
void decode (InstrInterface & interface, std::string coil_file, bool verbose);
void schedule (InstrInterface & interface, scheduler type, std::string in_file, std::string outfile, bool verbose);

#endif //GPC4_SS21_MAIN_H
