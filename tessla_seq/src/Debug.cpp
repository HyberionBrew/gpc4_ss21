//
// Created by daniel on 20.05.21.
//

#include "Debug.h"

void printIntStream(IntStream s){
    std::cout << "Debug IntStream" << std::endl;
    for (std::vector<IntEvent>::iterator it = s.stream.begin() ; it != s.stream.end(); ++it) {
        std::cout << it->value << "=" << it->timestamp;
    }
    std::cout << std::endl;
}

void printUnitStream(UnitStream s){
    std::cout << "Debug UnitStream" << std::endl;
    for (std::vector<UnitEvent>::iterator it = s.stream.begin() ; it != s.stream.end(); ++it) {
        std::cout << it->timestamp << " ";
    }
    std::cout << std::endl;
}