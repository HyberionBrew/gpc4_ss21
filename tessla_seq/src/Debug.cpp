//
// Created by daniel on 20.05.21.
//

#include "Debug.h"

void printIntStream(IntStream s){
    std::cout << "Debug IntStream" << std::endl;
    for (IntStream::iterator it = s.begin() ; it != s.end(); ++it) {
        std::cout << it->value << std::endl;
        std::cout << "-" << std::endl;
        std::cout <<it->timestamp << std::endl;
        std::cout << std::endl;
    }
}

void printUnitStream(UnitStream s){
    std::cout << "Debug UnitStream" << std::endl;
    for (UnitStream::iterator it = s.begin() ; it != s.end(); ++it) {
        std::cout << it->timestamp << " ";
    }
    std::cout << std::endl;
}