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
        while (getline(file, buf)) {
            buf.erase(std::remove_if(buf.begin(), buf.end(),::isspace));
            size_t colPos = buf.find(':');
            size_t eqPos = buf.find('=');
            if (colPos == std::string::npos || eqPos == std::string::npos) {
                //TODO throw exception
                std::cerr << "Invalid line";
            }
            int timestamp = stoi(buf, 0);
            string name = buf.substr(colPos+1, eqPos-colPos-1);

            try {
                size_t post_eq = eqPos + 1;
                int value = stoi(buf, &post_eq);
                // create int event
                IntEvent ev = IntEvent(timestamp, value);

                // check if exists in map
                if (this->intStreams.find(name) == this->intStreams.end()) {
                    IntStream s;
                    this->intStreams.insert(std::pair<string,IntStream>(name, s));
                }

                if (this->intStreams.find(name) != this->intStreams.end()) {
                    this->intStreams.find(name)->second.stream.push_back(ev);
                } else {
                    std::cout << "SHOULD NOT BE REACHED";
                }

            } catch (invalid_argument) {
                // create unit event
                UnitEvent ev = UnitEvent(timestamp);

                // check if exists in map
                if (this->unitStreams.find(name) == this->unitStreams.end()) {
                    UnitStream s;
                    this->unitStreams.insert(std::pair<string, UnitStream>(name, s));
                }

                if (this->unitStreams.find(name) != this->unitStreams.end()) {
                    this->unitStreams.find(name)->second.stream.push_back(ev);
                } else {
                    std::cout << "SHOULD NOT BE REACHED";
                }
                // try to create unit event
            }
        }
    }
}

UnitStream Reader::getUnitStream(string name) {
    if (this->unitStreams.find(name) != this->unitStreams.end()) {
        return this->unitStreams.find(name)->second;
    } else {
        // TODO throw exception
    }
}

IntStream Reader::getIntStream(string name) {
    if (this->intStreams.find(name) != this->intStreams.end()) {
        return this->intStreams.find(name)->second;
    } else {
        // TODO throw exception
    }
}