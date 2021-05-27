//
// Created by daniel on 20.05.21.
//

#include "Writer.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <string>

Writer::Writer(std::string outputFile){
    this->FILENAME = outputFile;
};

void Writer::addUnitStream(std::string name,UnitStream stream){
    Stream stream(name,stream);
    Writer::streams.push_back();
};
void Writer::addIntStream(std::string name,IntStream stream){
    Stream stream(name,stream);
    Writer::streams.push_back();
};
void Writer::writeOutputFile() {
    //traverse streams
    uint32_t lowest_timestamp = UINTMAX_MAX;
    int value = 0;
    IntStream bestIntStream;
    UnitStream bestUnitStream;
    bool isIntStream;
    bool finished = false;

    std::ofstream f;
    f.open(FILENAME, ios::out);
/*
    while (finished) {
        finished = true;
        for (std::vector<Stream>::iterator it = this->streams.begin(); it != streams.end(); it++) {
            if (it->get_type() == INT_STREAM) {
                std::vector <IntStream> currStreamInt = it->get_IntStream();
                if (lowest_timestamp > currStreamInt.begin().timestamp) {
                    bestIntStream = currStreamInt;
                    isIntStream = true;
                    finished = false;
                }
            } else {
                std::vector <UnitStream> currStream = it->get_UnitStream();
                if (lowest_timestamp > currStream.begin().timestamp) {
                    bestUnitStream = currStream;
                    isIntStream = false;
                    finished = false;
                }
            }
        }
        if (isIntStream == true) {
            IntEvent ev = bestIntStream;

            f << ev.timestamp << ": "<<ev.name <<" = " << ev.value << "\n";
            ev.erase(ev.begin());
            //print some stuff with ev
        } else {
            UnitEvent ev = bestUnitStream;
            f << ev.timestamp << ": "<<ev.name << " = ()" << "\n";
            ev.erase(ev.begin());
        }
    }*/
    f.close();

};




void writeStream(IntStream s, UnitStream) {

    for (IntStream::iterator event = s.begin(); event != s.end(); ++event) {

    }


}