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

IntInStream::IntInStream() {
}

UnitInStream::UnitInStream() {
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

                // check if exists in map
                if (this->intStreams.find(name) == this->intStreams.end()) {
                    shared_ptr<IntInStream> s = make_shared<IntInStream>();
                    this->intStreams.insert(std::pair<string,shared_ptr<IntInStream>>(name, s));
                }

                if (this->intStreams.find(name) != this->intStreams.end()) {
                    this->intStreams.find(name)->second->timestamps.push_back(timestamp);
                    this->intStreams.find(name)->second->values.push_back(value);
                } else {
                    throw std::runtime_error("Error in IntStream map insertion for Stream \"" + name + "\"");
                }

            } catch (std::invalid_argument &ia) {
                // check unit event validity
                if (buf.substr(post_eq) != "()") {
                    throw std::runtime_error("Invalid string \"" + buf.substr(post_eq) +
                                             "\" at RHS of non-int stream");
                }

                // check if exists in map
                if (this->unitStreams.find(name) == this->unitStreams.end()) {
                    shared_ptr<UnitInStream> s = make_shared<UnitInStream>();
                    this->unitStreams.insert(std::pair<string,shared_ptr<UnitInStream>>(name, s));
                }

                if (this->unitStreams.find(name) != this->unitStreams.end()) {
                    this->intStreams.find(name)->second->timestamps.push_back(timestamp);
                } else {
                    throw std::runtime_error("Error in UnitStream map insertion for Stream \"" + name + "\"");
                }
            }
        }
    }
}

/*
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
*/

UnitStream Reader::getUnitStream(string name) {
    if (this->unitStreams.find(name) != this->unitStreams.end()) {
        vector<int> timestamps = this->unitStreams.find(name)->second->timestamps;
        timestamps.shrink_to_fit();
        return UnitStream(&timestamps.front(), timestamps.size());
    } else {
        throw std::runtime_error("could not find unit stream \"" + std::string(name) + "\"");
    }
}

IntStream Reader::getIntStream(string name) {
    if (this->intStreams.find(name) != this->intStreams.end()) {
        vector<int> tsv = this->intStreams.find(name)->second->timestamps;
        vector<int> vsv = this->intStreams.find(name)->second->values;
        tsv.shrink_to_fit();
        vsv.shrink_to_fit();
        int* ts = &tsv.front();
        int* vs = &vsv.front();
        size_t s = tsv.size();
        return IntStream(ts, vs, s);
    } else {
        throw std::runtime_error("could not find int stream \"" + std::string(name) + "\"");
    }
}