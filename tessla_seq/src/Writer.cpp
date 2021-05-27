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
};

/*
void Writer::addUnitStream(std::string name,UnitStream stream){
    Stream streamUnit(name,stream);
    Writer::streams.push_back(streamUnit);
};
void Writer::addIntStream(std::string name,IntStream stream){
    Stream streamInt(name,stream);
    Writer::streams.push_back(streamInt);
};
 */

void Writer::addStream(std::string name, Stream& stream) {
    this->streams.push_back(OutputStream(name, stream));
}

void Writer::writeOutputFile() {
    //traverse streams

    int value = 0;
    //IntStream bestIntStream;
    //UnitStream bestUnitStream;

    std::ofstream f;
    f.open(FILENAME);
    std::string streamName = "";

    std::vector<Event>* currStream;
    std::vector<Event>* bestStream;

    bool finished = false;
    std::vector<int> ev_cnt(this->streams.size(), 0);
    int best_stream_idx = 0;
    int stream_idx = 0;
    while (!finished) {
        finished = true;
        uint32_t lowest_timestamp = UINT32_MAX;
        for (std::vector<OutputStream>::iterator it = this->streams.begin(); it != streams.end(); it++) {
            std::vector<Event> tmp = it->stream.get_event_stream();
            currStream = &tmp;
            stream_idx = it - streams.begin();
            if (currStream->begin() + ev_cnt[stream_idx] >= currStream->end()) {
                continue;
            }
            if (lowest_timestamp > (currStream->begin() + ev_cnt[stream_idx])->timestamp) {
                bestStream = currStream;
                streamName = it->name;
                finished = false;
                best_stream_idx = stream_idx;
                lowest_timestamp = ((currStream->begin() + ev_cnt[stream_idx])->timestamp);
            }
        }
        if (!finished){
            Event& ev = *(bestStream->begin() + ev_cnt[best_stream_idx]);

            f << ev.string_rep(streamName) << "\n";
            ev_cnt[best_stream_idx]++;
        }
    }
    f.close();

};



/*
void writeStream(IntStream s, UnitStream) {

    for (IntStream::iterator event = s.begin(); event != s.end(); ++event) {

    }


}
 */