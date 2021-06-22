//
// Created by daniel on 20.05.21.
//

#include "Reader.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <regex>
#include <string>

using namespace std;

Reader::Reader(string inputFile) {
    this->FILENAME = inputFile;
    readStreams();
}

void Reader::readStreams() {
    fstream file;
    file.open(this->FILENAME, ios::in);
    if (file.is_open())  {
        string buf;
        int i = 0;
        while (getline(file, buf)) {
            i++;
            buf.erase(std::remove_if(buf.begin(), buf.end(),::isspace), buf.end());
            size_t colPos = buf.find(':');
            size_t eqPos = buf.find('=');
            if (colPos == std::string::npos || eqPos == std::string::npos) {
                char buff[50];
                std::snprintf(buff, sizeof(buff), "Line %d: invalid pattern", i);
                throw std::runtime_error(buff);
            }
            int timestamp = stoi(buf, nullptr);
            string name = buf.substr(colPos+1, eqPos-colPos-1);

            size_t post_eq = eqPos + 1;
            try {
                int value = stoi(buf.substr(post_eq));
                // create int event
                IntEvent ev = IntEvent(timestamp, value);

                // check if exists in map
                if (this->intStreams.find(name) == this->intStreams.end()) {
                    auto s = std::make_shared<IntStream>();
                    this->intStreams.insert(std::pair<string,shared_ptr<IntStream>>(name, s));
                }

                if (this->intStreams.find(name) != this->intStreams.end()) {
                    this->intStreams.find(name)->second->stream.push_back(ev);
                } else {
                    throw std::runtime_error("Error in IntStream map insertion for Stream \"" + name + "\"");
                }

            } catch (std::invalid_argument &ia) {
                // check unit event validity
                if (buf.substr(post_eq) != "()") {
                    throw std::runtime_error("Invalid string \"" + buf.substr(post_eq) +
                                             "\" at RHS of non-int stream");
                }

                // create unit event
                UnitEvent ev = UnitEvent(timestamp);

                // check if exists in map
                if (this->unitStreams.find(name) == this->unitStreams.end()) {
                    auto s = std::make_shared<UnitStream>();
                    this->unitStreams.insert(std::pair<string, shared_ptr<UnitStream>>(name, s));
                }

                if (this->unitStreams.find(name) != this->unitStreams.end()) {
                    this->unitStreams.find(name)->second->stream.push_back(ev);
                } else {
                    throw std::runtime_error("Error in UnitStream map insertion for Stream \"" + name + "\"");
                }
            }
        }
    }
}

shared_ptr<UnitStream> Reader::getUnitStream(string name) {
    const char* name_ptr = name.c_str();
    if (this->unitStreams.find(name_ptr) != this->unitStreams.end()) {
        return this->unitStreams.find(name_ptr)->second;
    } else {
        throw std::runtime_error("could not find unit stream \"" + name + "\"");
    }
}

shared_ptr<IntStream> Reader::getIntStream(string name) {
    const char* name_ptr = name.c_str();
    if (this->intStreams.find(name_ptr) != this->intStreams.end()) {
        return this->intStreams.find(name_ptr)->second;
    } else {
        throw std::runtime_error("could not find int stream \"" + name + "\"");
    }
}