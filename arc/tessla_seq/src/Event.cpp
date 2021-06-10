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


bool operator==(const UnitEvent lhs, const UnitEvent rhs)
{
    return lhs.timestamp == rhs.timestamp;
}

bool operator==(const IntEvent lhs, const IntEvent rhs)
{
    return lhs.timestamp == rhs.timestamp && lhs.value == rhs.value;
}

std::string IntEvent::string_rep(std::string name) {
    std::ostringstream str_rep;
    str_rep << this->timestamp << ": " << name << " = " << this->value;
    return str_rep.str();
}

size_t IntEvent::get_timestamp() {
    return this->timestamp;
}

UnitEvent::UnitEvent(size_t ts){
    this->timestamp = ts;
}

std::string UnitEvent::string_rep(std::string name) {
    std::ostringstream str_rep;
    str_rep << this->timestamp << ": " << name << " = ()";
    return str_rep.str();
}

size_t UnitEvent::get_timestamp() {
    return this->timestamp;
}

