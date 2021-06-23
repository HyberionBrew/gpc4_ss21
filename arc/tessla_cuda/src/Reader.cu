#include "GPUReader.cuh"
#include <assert.h>
#include <fstream>
#include <iostream>
#include <regex>
#include <string>
#include <vector>
#include <time.h>
using namespace std;

#define NEW_READER

GPUReader::GPUReader(string inputFile) {
    this->FILENAME = inputFile;
#ifdef NEW_READER
    readStreams();
#endif
}

IntInStream::IntInStream() {
}

UnitInStream::UnitInStream() {
}

void printArray(int* array, size_t len, string name) {
    printf("%s : [", name.c_str());
    for (int i=0; i < len - 1; i++) {
        printf("%d, ", array[i]);
    }
    printf("%d]\n", array[len-1]);
}

void GPUReader::readStreams() {
    fstream file;
    clock_t start = clock();
    file.open(this->FILENAME, ios::in);
    printf("read file %s\n", this->FILENAME.c_str());
    if (file.is_open())  {
        string buf;
        int i = 0;
        while (getline(file, buf)) {
            //printf("LINE %d\n", i);
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
                if (this->GPUIntStreams.find(name) == this->GPUIntStreams.end()) {
                    //printf("Create int stream %s\n", name.c_str());
                    shared_ptr<IntInStream> s = make_shared<IntInStream>();
                    this->GPUIntStreams.insert(std::pair<string,shared_ptr<IntInStream>>(name, s));
                }

                if (this->GPUIntStreams.find(name) != this->GPUIntStreams.end()) {
                    //printf("Insert (%d, %d) int stream %s\n", timestamp, value, name.c_str());
                    this->GPUIntStreams.find(name)->second->timestamps.push_back(timestamp);
                    this->GPUIntStreams.find(name)->second->values.push_back(value);
                } else {
                    throw std::runtime_error("Error in GPUIntStream map insertion for Stream \"" + name + "\"");
                }

            } catch (std::invalid_argument &ia) {
                // check unit event validity
                if (buf.substr(post_eq) != "()") {
                    throw std::runtime_error("Invalid string \"" + buf.substr(post_eq) +
                                             "\" at RHS of non-int stream");
                }

                // check if exists in map
                if (this->GPUUnitStreams.find(name) == this->GPUUnitStreams.end()) {
                    //printf("Create unit stream %s\n", name.c_str());
                    shared_ptr<UnitInStream> s = make_shared<UnitInStream>();
                    this->GPUUnitStreams.insert(std::pair<string,shared_ptr<UnitInStream>>(name, s));
                }

                if (this->GPUUnitStreams.find(name) != this->GPUUnitStreams.end()) {
                    //printf("Insert %d in unit stream %s\n", timestamp, name.c_str());
                    this->GPUUnitStreams.find(name)->second->timestamps.push_back(timestamp);
                    //printf("last elem in %s: %d\n", name.c_str(), this->GPUUnitStreams.find(name)->second->timestamps.back());
                    //printf("Post insert unit stream %s\n", name.c_str());
                } else {
                    throw std::runtime_error("Error in GPUUnitStream map insertion for Stream \"" + name + "\"");
                }
            }
        }
    }
    clock_t dur = clock() - start;
    printf("READING TOOK %ld us\n", dur*1000000/CLOCKS_PER_SEC);
}

#ifndef NEW_READER
GPUUnitStream Reader::getGPUUnitStream(string name) {
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
        cout << "Warning: Stream " << name << " is not present in the input file but was read!" << "\n";
        //exit(1);
    }

    int *timestampsA = (int*) malloc(timestampsCnt * sizeof(int));
    memset(timestampsA, 0, timestampsCnt * sizeof(int));
    copy(timestamps.begin(), timestamps.end(), timestampsA);

    /*
    printf("%s: size=%d\n", name.c_str(), timestampsCnt);
    if (timestampsCnt < 10000) {
        printArray(timestampsA, timestampsCnt, "ts (" + name + ")");
    }
     */

    GPUUnitStream readStream = GPUUnitStream(timestampsA, timestampsCnt);
    return readStream;
}

GPUIntStream Reader::getGPUIntStream(string name) {
    fstream file;
    file.open(this->FILENAME, ios::in);
    vector<int> timestamps;
    vector<int> values;

    printf("read file %s\n", this->FILENAME.c_str());
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
        cout << "Warning: Stream " << name << " is not present in the input file but was read!" << "\n";
        //exit(1);
    }

    size_t mallocSize = timestampsCnt * sizeof(int);
    int *timestampsA = (int*) malloc(mallocSize);
    int *valuesA = (int*) malloc(mallocSize);
    memset(timestampsA, 0, mallocSize);
    memset(valuesA, 0, mallocSize);
    copy(timestamps.begin(), timestamps.end(), timestampsA);
    copy(values.begin(), values.end(), valuesA);

    /*
    printf("%s: size=%d\n", name.c_str(), timestampsCnt);
    if (timestampsCnt < 10000) {
        printArray(timestampsA, timestampsCnt, "ts (" + name + ")");
        printArray(valuesA, timestampsCnt, "vs (" + name + ")");
    }
    */

    GPUIntStream readStream = GPUIntStream(timestampsA, valuesA, timestampsCnt);
    return readStream;
}
#endif

#ifdef NEW_READER
shared_ptr<GPUUnitStream> GPUReader::getUnitStream(string name) {
    if (this->GPUUnitStreams.find(name) != this->GPUUnitStreams.end()) {
        vector<int> *timestamps = &this->GPUUnitStreams.find(name)->second->timestamps;
        size_t mallocSize = timestamps->size() * sizeof(int);
        size_t size = timestamps->size();
        int *timestampsA = (int*) malloc(mallocSize);
        copy(timestamps->begin(), timestamps->end(), timestampsA);
        /*
        printf("%s: size=%zu\n", name.c_str(), timestamps->size());
        if (timestamps->size() < 10000) {
            printArray(&(*timestamps)[0], timestamps->size(), "ts (" + name + ")");
        }
         */
        return std::make_shared<GPUUnitStream>(GPUUnitStream{timestampsA, size});
    } else {
        throw std::runtime_error("could not find unit stream \"" + std::string(name) + "\"");
    }
}

shared_ptr<GPUIntStream> GPUReader::getIntStream(string name) {
    if (this->GPUIntStreams.find(name) != this->GPUIntStreams.end()) {
        vector<int> *timestamps = &this->GPUIntStreams.find(name)->second->timestamps;
        vector<int> *values = &this->GPUIntStreams.find(name)->second->values;
        size_t mallocSize = timestamps->size() * sizeof(int);
        size_t size = timestamps->size();
        int *timestampsA = (int*) malloc(mallocSize);
        int *valuesA = (int*) malloc(mallocSize);
        clock_t start = clock();
        copy(timestamps->begin(), timestamps->end(), timestampsA);
        copy(values->begin(), values->end(), valuesA);
        clock_t time = clock() - start;
        printf("MEMCPY TIME USED:: %ld\n", time*1000000/CLOCKS_PER_SEC);
        /*
        printf("%s: size=%zu\n", name.c_str(), size);
        if (size < 10000) {
            printArray(&(*timestamps)[0], timestamps->size(), "ts (" + name + ")");
            printArray(&(*values)[0], values->size(), "vs (" + name + ")");
        }
         */
        return make_shared<GPUIntStream>(GPUIntStream{timestampsA, valuesA, size});
    } else {
        throw std::runtime_error("could not find int stream \"" + std::string(name) + "\"");
    }
}
#endif