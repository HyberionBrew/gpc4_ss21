//
// Created by daniel on 20.05.21.
//

#include "Writer.h"
#include <iostream>
#include <fstream>
#include <string>
#include <ios>
#include <queue>

Writer::Writer(std::string outputFile){
    this->FILENAME = outputFile;
}

void Writer::addStream(std::string name, std::shared_ptr<Stream> intStream) {
    this->streams.push_back(intStream);
    this->stream_names.push_back(name);
}

void Writer::writeOutputFile() {

    // Open output file
    std::ofstream f;
    f.open(FILENAME);

    // Priority queue of streams to be put in
    std::priority_queue<std::pair<int, int>, std::deque<std::pair<int,int>>, std::greater<std::pair<int, int>>> sorter;

    // Current positions in stream
    int* positions = (int*) calloc(streams.size(), sizeof (int));

    // Format events for quick execution
    Event*** event_streams = (Event***) malloc(streams.size()*sizeof(Event**));

    // Event array sizes
    size_t* sizes = (size_t*) malloc(streams.size() * sizeof (size_t));

    if (positions == nullptr || event_streams == nullptr || sizes == nullptr) {
        throw std::runtime_error("Out of memory. Cannot sort output streams.");
    }

    // Populate data structures
    int i = 0;
    for (auto & stream : streams) {
        auto ev = stream->get_event_stream();

        // Save position
        sizes[i] = ev.size();

        // Copy events
        event_streams[i] = (Event**) malloc(ev.size() * sizeof (Event**));
        if (event_streams[i] == nullptr) {
            throw std::runtime_error("Out of memory. Cannot sort output streams.");
        }
        std::copy(ev.begin(), ev.end(), event_streams[i]);

        sorter.push(std::make_pair(event_streams[i][0]->get_timestamp(), i));
        i++;
    }

    std::pair<int, int> current;

    // Traverse the priority queue
    while (true) {

        // Check if queue is empty
        if (sorter.empty()) break;

        // Pop the first element and write it to the file
        current = sorter.top();
        f << event_streams[current.second][positions[current.second]]->string_rep(this->stream_names[current.second]) << "\n";
        positions[current.second]++;

        // Check if end is reached and sort it back in
        if (positions[current.second] < sizes[current.second]) {
            sorter.push(std::make_pair(event_streams[current.second][positions[current.second]]->get_timestamp(), current.second));
        }

        // Remove the first element
        sorter.pop();
    }

    // Close the file and free the position array
    f.close();
    delete positions;
    delete event_streams;
}