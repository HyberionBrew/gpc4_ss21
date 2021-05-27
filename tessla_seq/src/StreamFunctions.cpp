//
// Created by daniel on 20.05.21.
//

#include "StreamFunctions.h"

IntStream time(Stream& s){
    IntStream result;
    std::vector<Event> es = s.get_event_stream();
    for (std::vector<Event>::iterator it = es.begin() ; it != es.end(); ++it) {
        IntEvent node(it->timestamp,(int32_t)it->timestamp);
        result.stream.push_back(node);
    }
    return result;
}

UnitStream unit(){
    UnitStream result;
    result.stream.push_back( (UnitEvent) {0} );
    return result;
}

IntStream def(int32_t val){
    IntStream result;
    result.stream.push_back( (IntEvent) {0, val} );
    return result;
}

IntStream last(IntStream v, UnitStream r){
    IntStream result;
    std::vector<IntEvent>::iterator currEventV = v.stream.begin();
    for (std::vector<UnitEvent>::iterator currEventR = r.stream.begin() ; currEventR != r.stream.end(); ++currEventR) {
        if (currEventV->timestamp >= currEventR->timestamp){
               continue;
        }
        else{
            if ((currEventV+1) != v.stream.end()){
                while ((currEventV+1)->timestamp <= currEventR->timestamp && (currEventV+1) != v.stream.end()){
                    currEventV++;
                }
            }
            IntEvent node{currEventR->timestamp,currEventV->value};
            result.stream.push_back(node);
        }
    }
    return result;
}

UnitStream delay(IntStream d, UnitStream r){
    std::vector<UnitEvent> outstream;
    std::vector<UnitEvent>::iterator currEventR = r.stream.begin();
    std::vector<UnitStream>::size_type currEventOutIndex = 0;
    for (std::vector<IntEvent>::iterator currEventD = d.stream.begin(); currEventD != d.stream.end(); ++currEventD) {
        while (currEventR->timestamp < currEventD->timestamp && currEventR != r.stream.end()) {
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
    UnitStream delayed(outstream);
    return delayed;
}

IntStream count(UnitStream y) {
    IntStream outstream(y.stream.size());
    int32_t index = 0;
    for(std::vector<UnitEvent>::iterator it = y.stream.begin(); it != y.stream.end(); ++it) {
        outstream.stream.push_back((IntEvent){it->timestamp,index});
    }
    return outstream;
}

IntStream merge(IntStream s1, IntStream s2) {
    std::vector<IntEvent> outstream;

    std::vector<IntEvent> x = s1.stream;
    std::vector<IntEvent> y = s2.stream;

    std::vector<IntEvent>::iterator x_it = x.begin();
    std::vector<IntEvent>::iterator y_it = y.begin();

    size_t last_ts = -1;

    bool x_end = x.size() == 0;
    bool y_end = y.size() == 0;

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
            if (x_it->timestamp == y_it->timestamp){
                std::cout << "Double Timestamp: " << x_it->timestamp << std::endl;
                y_it++;
            }
            x_it++;
        }
        else {
            outstream.push_back((IntEvent) {y_it->timestamp, y_it->value});
            y_it++;
        }

        if (y_it >= y.end()) y_end = true;
        if (x_it >= x.end()) x_end = true;
    }
    IntStream merged(outstream);
    return merged;
}

UnitStream merge(UnitStream s1, UnitStream s2) {
    std::vector<UnitEvent> outstream;
    std::vector<UnitEvent> x = s1.stream;
    std::vector<UnitEvent> y = s2.stream;

    std::vector<UnitEvent>::iterator x_it = x.begin();
    std::vector<UnitEvent>::iterator y_it = y.begin();

    bool x_end = x.size() == 0;
    bool y_end = y.size() == 0;

    while(!x_end || !y_end){
        if (x_end){
            outstream.push_back((UnitEvent){y_it->timestamp});
            y_it++;
        }
        else if (y_end) {
            outstream.push_back((UnitEvent){x_it->timestamp});
            x_it++;
        }
        else if (x_it->timestamp <= y_it->timestamp){
            outstream.push_back((UnitEvent){x_it->timestamp});
            if (x_it->timestamp == y_it->timestamp){
                std::cout << "Double Timestamp: " << x_it->timestamp << std::endl;
                y_it++;
            }
            x_it++;
        }
        else{
            outstream.push_back((UnitEvent){y_it->timestamp});
            y_it++;
        }

        if (y_it >= y.end()) y_end = true;
        if (x_it >= x.end()) x_end = true;
    }
    UnitStream merged(outstream);
    return merged;
}
/*
std::tuple<IntStream, IntStream> slift_streams(IntStream x, IntStream y) {
    // IntStream xp = merge(x,last(x,y));
    // IntStream yp = merge(y,last(x,y));
    return NULL;
}

// stream + value
IntStream add(IntStream x, int value){
    //UnitStream::iterator x_it = x.begin();
    //UnitStream::iterator y_it = y.begin();
    //while(!x_end || !y_end){

    //}
}

// stream - value
IntStream sub1(IntStream x, int value){

};

// value - stream
IntStream sub2(IntStream x, int value){

};

// stream * value
IntStream mul(IntStream x, int value){

};

// stream / value
IntStream div1(IntStream x, int value){

};

// value / stream
IntStream div2(IntStream x, int value){

};

// stream % value
IntStream mod1(IntStream x, int value){

};

// value % stream
IntStream mod2(IntStream x, int value){

};

// x + y
IntStream add(IntStream x, IntStream y){

};

// x - y
IntStream sub(IntStream x, IntStream y){

};

// x * y
IntStream mul(IntStream x, IntStream y){

};

// x / y
IntStream div(IntStream x, IntStream y){

};

// x % y
IntStream mod(IntStream x, IntStream y){

}

*/