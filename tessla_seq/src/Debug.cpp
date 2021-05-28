//
// Created by daniel on 20.05.21.
//

#include "Debug.h"

void printIntStream(IntStream s){
    std::cout << "Debug IntStream" << std::endl;
    for (auto & e : s.stream) {
        std::cout << e.string_rep("") << std::endl;
    }
    std::cout << std::endl;
}

void printUnitStream(UnitStream s){
    std::cout << "Debug UnitStream" << std::endl;
    for (auto & e : s.stream) {
        std::cout << e.string_rep("") << std::endl;
    }
    std::cout << std::endl;
}