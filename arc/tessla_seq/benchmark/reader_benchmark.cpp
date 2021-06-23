//
// Created by klaus on 16.06.21.
//

#include <Reader.h>
#include <ctime>
#include <iostream>

int main(int argc, char **argv) {
    clock_t start = clock();
    Reader r = Reader("./data/benchmarking8.in");
    clock_t end = clock();
    std::cout << end - start << " us";
}

