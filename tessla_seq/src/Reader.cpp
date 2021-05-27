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
}

UnitStream Reader::getUnitStream(string name) {
    fstream file;
    file.open(this->FILENAME, ios::in);
    UnitStream readStream;

    if (file.is_open()) {
        string buf;
        regex pattern("([0-9]+):\\s*([A-Za-z][0-9A-Za-z]*)\\s*=\\s*\\(\\)\\s*");
        while (getline(file, buf)) {
            // match each line to regex
            smatch matches;
            if (regex_match(buf, matches, pattern)) {
                int timestamp = stoi(matches[1]);
                if (name.compare(matches[2]) == 0) {
                    UnitEvent ue{timestamp};
                    readStream.stream.push_back(ue);
                }
            }
        }
        file.close();
    }
    if (readStream.stream.size() == 0) {
        cerr << "Stream " << name << " is not present in the input file" << "\n";
        exit(1);
    }
    return readStream;
}

IntStream Reader::getIntStream(string name) {
    fstream file;
    file.open(this->FILENAME, ios::in);
    IntStream readStream;

    if (file.is_open()) {
        string buf;
        // match each line to regex
        regex pattern("([0-9]+):\\s*([A-Za-z][0-9A-Za-z]*)\\s*=\\s*([0-9]+)\\s*");
        while (getline(file, buf)) {
            // match each line to regex
            smatch matches;
            if (regex_match(buf, matches, pattern)) {
                int timestamp = stoi(matches[1]);
                int value = stoi(matches[3]);
                if (name.compare(matches[2]) == 0) {
                    IntEvent ie{timestamp, value};
                    readStream.stream.push_back(ie);
                }
            }
        }
        file.close();
    }
    if (readStream.stream.size() == 0) {
        cerr << "Stream " << name << " is not present in the input file" << "\n";
        exit(1);
    }
    return readStream;
}