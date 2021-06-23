//
// Created by daniel on 20.05.21.
//

#include "Writer.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <ios>

Writer::Writer(std::string outputFile){
    this->FILENAME = outputFile;
    this->unitPos = 0;
}

void Writer::addIntStream(std::string name, std::shared_ptr<Stream> intStream) {
    this->streams.push_back(intStream);
    this->stream_names.push_back(name);
}

void Writer::addUnitStream(std::string name, std::shared_ptr<Stream> unitStream) {
    this->streams.insert((this->streams.begin() + unitPos), unitStream);
    this->stream_names.insert((this->stream_names.begin() + unitPos), name);
    this->unitPos++;
}

void Writer::writeOutputFile() {
    //traverse streams

    int value = 0;
    //IntStream bestIntStream;
    //UnitStream bestUnitStream;

    std::ofstream f;
    f.open(FILENAME);
    std::string streamName = "";

    std::vector<Event*>* currStream;
    std::vector<Event*>* bestStream;

    // termination condition
    bool finished = false;

    // holds indices that were already written for each stream
    std::vector<int> ev_cnt(this->streams.size(), 0);

    // indices of current prioritized streams ?
    int best_stream_idx = 0;
    int stream_idx = 0;

    // PRINT ALL STREAMS
    /*
    for (auto it = streams.begin(); it != streams.end(); it++) {
        std::vector<Event*> event_vector = (*it)->get_event_stream();
        for (auto ev : event_vector) {
            int idx = (it-streams.begin());
            std::cout << "STREAM | "<< ev->string_rep(stream_names[idx]) << std::endl;
        }
    }
    */

    std::vector<Event*> tmp;
    while (!finished) {
        // std::cout << best_stream_idx << std::endl; // Debug print

        finished = true; // termination condition
        uint32_t lowest_timestamp = UINT32_MAX; // set to maximum and check for lower (timestamp is size_t)
        for (auto it = streams.begin(); it != streams.end(); it++) {
            tmp = (*it)->get_event_stream();
            currStream = &tmp;
            stream_idx = it - streams.begin();
            // If all events of this stream have been added, go to the next stream
            if (currStream->begin() + ev_cnt[stream_idx] >= currStream->end()) {
                continue;
            }
            // If this streams timestamp is better than the best so far, save it for later
            if (lowest_timestamp > (*(currStream->begin() + ev_cnt[stream_idx]))->get_timestamp()) {
                finished = false;
                best_stream_idx = stream_idx;
                lowest_timestamp = ((*(currStream->begin() + ev_cnt[stream_idx]))->get_timestamp());
            }
        }
        // Add the best event ot the output
        if (!finished){
            std::vector<Event*> bs = this->streams[best_stream_idx]->get_event_stream();
            Event* ev = *(bs.begin() + ev_cnt[best_stream_idx]);
            f << ev->string_rep(this->stream_names[best_stream_idx]) << "\n";
            (ev_cnt[best_stream_idx])++; // only increment ev_cnt here
        }
    }
    f.close();

}