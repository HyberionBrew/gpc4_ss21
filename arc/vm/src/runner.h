//
// Created by klaus on 24.06.21.
//

#ifndef ARC_RUNNER_H
#define ARC_RUNNER_H

#include "InstrInterface.h"
#include "Scheduler.h"
#include <string>

enum scheduler{debug, sequential, gpu, thrust};

int run(int argc, char **argv);

void decode (InstrInterface & interface, std::string coil_file, bool verbose);
void schedule (InstrInterface & interface, scheduler type, std::string in_file, std::string outfile, bool verbose);

#endif //ARC_RUNNER_H
