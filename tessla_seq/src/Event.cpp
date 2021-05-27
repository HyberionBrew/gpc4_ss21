//
// Created by daniel on 20.05.21.
//

#include "Event.h"
#include <iostream>
#include <sstream>

IntEvent::IntEvent(size_t ts, int32_t v){
    this->timestamp = ts;
    this->value = v;
}

std::string IntEvent::string_rep(std::string name) {
    std::ostringstream str_rep;
    str_rep << this->timestamp << ": " << name << " = " << this->value;
    return str_rep;
}

UnitEvent::UnitEvent(size_t ts){
    this->timestamp = ts;
}

std::string UnitEvent::string_rep(std::string name) {
    std::ostringstream str_rep;
    str_rep << this->timestamp << ": " << name << " = ()";
    return str_rep;
}


