//
// Created by klaus on 12.07.21.
//

#include "GPUWriter.cuh"
#include <iostream>
#include <fstream>
#include <string>
#include <ios>
#include <queue>

GPUWriter::GPUWriter(std::string outputFile){
    this->FILENAME = outputFile;
}

void GPUWriter::addUnitStream(std::string name, std::shared_ptr<GPUUnitStream> unitStream) {
    this->unit_streams.push_back(unitStream);
    this->unit_names.push_back(name);
}

void GPUWriter::addIntStream(std::string name, std::shared_ptr<GPUIntStream> intStream) {
    this->int_streams.push_back(intStream);
    this->int_names.push_back(name);
}

void GPUWriter::writeOutputFile() {

    // Open output file
    std::ofstream f;
    f.open(FILENAME);

    // Priority queue of streams to be put in

    std::priority_queue<std::tuple<int, int, bool>, std::deque<std::tuple<int, int, bool>>, std::greater<std::tuple<int, int, bool>>> sorter;

    // Current positions in stream
    int* int_positions = (int*) calloc(this->int_streams.size(), sizeof (int));
    int* unit_positions = (int*) calloc(this->unit_streams.size(), sizeof (int));
    int* int_sizes = (int*) malloc(this->int_streams.size()*sizeof (int));
    int* unit_sizes = (int*) malloc(this->unit_streams.size()*sizeof (int));

    if (int_positions == nullptr || unit_positions == nullptr || int_sizes == nullptr || unit_sizes == nullptr) {
        throw std::runtime_error("Out of memory. Cannot sort output streams.");
    }

    // Populate data structures
    int i = 0;
    for (auto & stream : int_streams) {
        // Save position
        int_sizes[i] = sizeof(stream->host_timestamp)/sizeof(int);

        sorter.push(std::make_tuple(stream->host_timestamp[0], i, true));
        i++;
    }

    i = 0;
    for (auto & stream : unit_streams) {
        // Save position
        unit_sizes[i] = sizeof(stream->host_timestamp)/sizeof(int);

        sorter.push(std::make_tuple(stream->host_timestamp[0], i, false));
        i++;
    }

    std::tuple<int, int, bool> current;

    // Traverse the priority queue
    while (true) {

        // Check if queue is empty
        if (sorter.empty()) break;

        // Get the first element and write it to the file
        current = sorter.top();
        int timestamp = std::get<0>(current);
        int index = std::get<1>(current);
        bool isIntStream = std::get<2>(current);

        // switch depending on stream type
        if (isIntStream) {
            auto s = int_streams[index];
            int p = int_positions[index];
            f << s->host_timestamp[p] << ": " << int_names[index] << " = " << s->host_values[p] << "\n";
            p = ++int_positions[index];

            if (int_positions[index] < int_sizes[index]) {
                sorter.push(std::make_tuple(s->host_timestamp[p], index, true));
            }
        } else {
            auto s = unit_streams[index];
            int p = unit_positions[index];
            f << s->host_timestamp[p] << ": " << unit_names[index] << " = ()\n";
            p = ++unit_positions[index];

            if (unit_positions[index] < unit_sizes[index]) {
                sorter.push(std::make_tuple(s->host_timestamp[p], index, false));
            }
        }

        // Remove the first element
        sorter.pop();
    }

    // Close the file and free the position array
    f.close();
    delete int_positions;
    delete unit_positions;
    delete int_sizes;
    delete unit_sizes;
}
