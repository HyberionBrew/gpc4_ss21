//
// Created by daniel on 20.05.21.
//

#include "StreamFunctions.h"

IntStream time(IntStream s){
    IntStream result;
    for (IntStream::iterator it = s.begin() ; it != s.end(); ++it) {
        IntEvent node{it->timestamp,(int32_t)it->timestamp};
        result.push_back(node);
    }
    return result;
}

UnitStream unit(){
    UnitStream result;
    result.push_back( (UnitEvent) {0} );
    return result;
}

IntStream def(int32_t val){
    IntStream result;
    result.push_back( (IntEvent) {0, val} );
    return result;
}

IntStream last(IntStream v, UnitStream r){
    IntStream result;
    IntStream::iterator currEventV = v.begin();
    for (UnitStream::iterator currEventR = r.begin() ; currEventR != r.end(); ++currEventR) {
        if (currEventV->timestamp >= currEventR->timestamp){
               continue;
        }
        else{
            if ((currEventV+1) != v.end()){
                while ((currEventV+1)->timestamp <= currEventR->timestamp){
                    ++currEventV;
                }
            }
            IntEvent node{currEventR->timestamp,currEventV->value};
            result.push_back(node);
        }
    }
    return result;
}

UnitStream delay(IntStream d, UnitStream r){
    UnitStream outstream;
    UnitStream::iterator currEventR = r.begin();
    std::vector<UnitStream>::size_type currEventOutIndex = 0;
    for (IntStream::iterator currEventD = d.begin(); currEventD != d.end(); ++currEventD) {
        while (currEventR->timestamp < currEventD->timestamp && currEventR != r.end()) {
            currEventR++;
        }
        while (currEventOutIndex < outstream.size() && outstream[currEventOutIndex].timestamp < outstream[currEventOutIndex].timestamp) {
            currEventOutIndex++;
        }
        if ((currEventD->timestamp == currEventR->timestamp) ||
            (currEventOutIndex < outstream.size() && (currEventD->timestamp == outstream[currEventOutIndex].timestamp))) {
            size_t target = currEventD->timestamp + currEventD->value;
            currEventR++;
            if (currEventR->timestamp >= target) {
                UnitEvent node{target};
                outstream.push_back(node);
            }
            currEventR--;
        }
    }
    return outstream;
}

IntStream count(UnitStream y) {
    IntStream outstream(y.size());
    int32_t index = 0;
    for(UnitStream::iterator it = y.begin(); it != y.end(); ++it) {
        outstream.push_back( (IntEvent) {it->timestamp,index} );
    }
    return outstream;
}

IntStream merge(IntStream x, IntStream y) {
    IntStream outstream;

    IntStream::iterator x_it = x.begin();
    IntStream::iterator y_it = y.begin();

    bool x_end = x.size() > 0 ? false : true;
    bool y_end = y.size() > 0 ? false : true;

    while(!x_end || !y_end){
        if (x_end){
            outstream.push_back((IntEvent){y_it->timestamp, y_it->value});
            y_it++;
        }
        else if (y_end) {
            outstream.push_back((IntEvent) {x_it->timestamp, x_it->value});
            x_it++;
        }
        else if (x_it->timestamp <= y_it->timestamp){
            outstream.push_back((IntEvent) {x_it->timestamp, x_it->value});
            if (x_it->timestamp == y_it->timestamp) {
                y_it++;
            }
            x_it++;
        }
        else {
            outstream.push_back((IntEvent) {y_it->timestamp, y_it->value});
            y_it++;
        }

        if (y_it == y.end()) y_end = true;
        if (x_it == x.end()) x_end = true;
    }
    return outstream;
}

UnitStream merge(UnitStream x, UnitStream y) {
    UnitStream outstream;

    UnitStream::iterator x_it = x.begin();
    UnitStream::iterator y_it = y.begin();

    bool x_end = x.size() > 0 ? false : true;
    bool y_end = y.size() > 0 ? false : true;

    while(!x_end || !y_end){
        if (x_end){
            outstream.push_back((UnitEvent){y_it->timestamp});
            y_it++;
        }
        else if (y_end) {
            outstream.push_back((UnitEvent) {x_it->timestamp});
            x_it++;
        }
        else if (x_it->timestamp <= y_it->timestamp){
            outstream.push_back(( UnitEvent){x_it->timestamp});
            x_it++;
        }
        else{
            outstream.push_back((UnitEvent){y_it->timestamp});
            y_it++;
        }

        if (y_it == y.end()) y_end = true;
        if (x_it == x.end()) x_end = true;
    }
    return outstream;
}