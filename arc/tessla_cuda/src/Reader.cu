#include "Reader.cuh"
#include <assert.h>
#include <fstream>
#include <iostream>
#include <regex>
#include <string>
#include <vector>
using namespace std;

Reader::Reader(string inputFile) {
    this->FILENAME = inputFile;
}

UnitStream Reader::getUnitStream(string name) {
    fstream file;
    file.open(this->FILENAME, ios::in);
    vector<int> timestamps;

    if (file.is_open()) {
        string buf;
        regex pattern("([0-9]+):\\s*([A-Za-z][0-9A-Za-z]*)\\s*=\\s*\\(\\)\\s*");
        while (getline(file, buf)) {
            // match each line to regex
            smatch matches;
            if (regex_match(buf, matches, pattern)) {
                if (name.compare(matches[2]) == 0) {
                    int timestamp = stoi(matches[1]);
                    timestamps.push_back(timestamp);
                }
            }
        }
        file.close();
    }

    size_t timestampsCnt = timestamps.size();
    if (timestampsCnt == 0) {
        cerr << "Stream " << name << " is not present in the input file" << "\n";
        exit(1);
    }

    int *timestampsA = (int*) malloc(timestampsCnt * sizeof(int));
    memset(timestampsA, 0, timestampsCnt * sizeof(int));
    copy(timestamps.begin(), timestamps.end(), timestampsA);

    UnitStream readStream = UnitStream(timestampsA, timestampsCnt);
    return readStream;
}

IntStream Reader::getIntStream(string name) {
    fstream file;
    file.open(this->FILENAME, ios::in);
    vector<int> timestamps;
    vector<int> values;

    if (file.is_open()) {
        string buf;
        // match each line to regex
        regex pattern("([0-9]+):\\s*([A-Za-z][0-9A-Za-z]*)\\s*=\\s*(-?[0-9]+)\\s*");
        while (getline(file, buf)) {
            // match each line to regex
            smatch matches;
            if (regex_match(buf, matches, pattern)) {
                if (name.compare(matches[2]) == 0) {
                    int timestamp = stoi(matches[1]);
                    int value = stoi(matches[3]);
                    timestamps.push_back(timestamp);
                    values.push_back(value);
                }
            }
        }
        file.close();
    }

    assert(timestamps.size() == values.size());
    size_t timestampsCnt = timestamps.size();
    if (timestampsCnt == 0) {
        cerr << "Stream " << name << " is not present in the input file" << "\n";
        exit(1);
    }

    size_t mallocSize = timestampsCnt * sizeof(int);
    int *timestampsA = (int*) malloc(mallocSize);
    int *valuesA = (int*) malloc(mallocSize);
    memset(timestampsA, 0, mallocSize);
    memset(valuesA, 0, mallocSize);
    copy(timestamps.begin(), timestamps.end(), timestampsA);
    copy(values.begin(), values.end(), valuesA);

    IntStream readStream = IntStream(timestampsA, valuesA, timestampsCnt);
    return readStream;
}