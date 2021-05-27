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

void Writer::addUnitStream(std::string name,UnitStream stream){
    Stream streamUnit(name,stream);
    Writer::streams.push_back(streamUnit);
};
void Writer::addIntStream(std::string name,IntStream stream){
    Stream streamInt(name,stream);
    Writer::streams.push_back(streamInt);
};
void Writer::writeOutputFile() {
    //traverse streams

    int value = 0;
    IntStream bestIntStream;
    UnitStream bestUnitStream;
    bool isIntStream;
    bool finished = false;

    std::ofstream f;
    f.open(FILENAME);
    std::string IntName = "a";
    std::string UnitName = "a";

    IntStream currStreamInt;
    UnitStream currStream;
    std::vector<int> ev_cnt(this->streams.size(), 0);
    int best_stream_idx = 0;
    int stream_idx = 0;
    while (!finished) {
        finished = true;
        uint32_t lowest_timestamp = UINT32_MAX;
        for (std::vector<Stream>::iterator it = this->streams.begin(); it != streams.end(); it++) {
            if (it->get_type() == INT_STREAM) {
                currStreamInt = it->get_IntStream();
                stream_idx = it - streams.begin();
                if (currStreamInt.begin() + ev_cnt[stream_idx] >= currStreamInt.end()) {
                    continue;
                }
                if (lowest_timestamp > (currStreamInt.begin() + ev_cnt[stream_idx])->timestamp) {
                    bestIntStream = currStreamInt;
                    IntName = it->name;
                    isIntStream = true;
                    finished = false;
                    best_stream_idx = stream_idx;
                    lowest_timestamp = ((currStreamInt.begin() + ev_cnt[stream_idx])->timestamp);
                }
            }
            else {
                currStream = it->get_UnitStream();
                stream_idx = it - streams.begin();
                if (currStream.begin() + ev_cnt[stream_idx] >= currStream.end()) {
                    continue;
                }
                if (lowest_timestamp > (currStream.begin()+ ev_cnt[stream_idx])->timestamp) {
                    bestUnitStream = currStream;
                    UnitName = it->name;
                    best_stream_idx = stream_idx;
                    isIntStream = false;
                    finished = false;
                    lowest_timestamp = (currStream.begin()+ ev_cnt[stream_idx])->timestamp;
                }
            }
        }
        if (!finished){
            if (isIntStream) {
                IntEvent ev = *(bestIntStream.begin() + ev_cnt[best_stream_idx]);

                f << ev.timestamp << ": " << IntName << " = " << ev.value << "\n";
                ev_cnt[best_stream_idx]++;
                //print some stuff with ev
            } else {

                UnitEvent ev = *(bestUnitStream.begin()+ ev_cnt[best_stream_idx]);
                f << ev.timestamp << ": " << UnitName << " = ()" << "\n";
                ev_cnt[best_stream_idx]++;
                //bestUnitStream->erase(bestUnitStream->begin());

            }
        }
    }
    f.close();

};




void writeStream(IntStream s, UnitStream) {

    for (IntStream::iterator event = s.begin(); event != s.end(); ++event) {

    }


}