//
// Created by daniel on 20.05.21.
//

#include "StreamFunctions.h"
#include <tuple>
#include "Debug.h"
#include <memory>

using namespace std;

shared_ptr<IntStream> time(Stream& s){
    shared_ptr<IntStream> result = make_shared<IntStream>();
    std::vector<Event*> es = s.get_event_stream();
    for (auto & elem : es) {
        Event* e = elem;
        IntEvent node(e->get_timestamp(),(int32_t)e->get_timestamp());
        result->stream.push_back(node);
    }
    return result;
}

shared_ptr<UnitStream> unit(){
    shared_ptr<UnitStream> result = make_shared<UnitStream>();
    result->stream.push_back( (UnitEvent) {0} );
    return result;
}

shared_ptr<IntStream> def(int32_t val){
    shared_ptr<IntStream> result = make_shared<IntStream>();
    result->stream.push_back( (IntEvent) {0, val} );
    return result;
}

shared_ptr<IntStream> last(IntStream& v, Stream& r){
    shared_ptr<IntStream> result = make_shared<IntStream>();
    std::vector<IntEvent>::iterator currEventV = v.stream.begin();
    std::vector<Event*> es = r.get_event_stream();
    for (std::vector<Event*>::iterator currEventR = es.begin() ; currEventR != es.end(); ++currEventR) {
        if (currEventV->get_timestamp() >= (*currEventR)->get_timestamp()){
           continue;
        }
        else{
            if ((currEventV+1) != v.stream.end()){
                while ((currEventV+1) != v.stream.end() && (currEventV+1)->get_timestamp() <= (*currEventR)->get_timestamp()){
                    currEventV++;
                }
            }
            IntEvent node{(*currEventR)->get_timestamp(),currEventV->value};
            result->stream.push_back(node);
        }
    }
    return result;
}

shared_ptr<UnitStream> delay(IntStream& d, Stream& r){
    std::vector<UnitEvent> outstream;
    std::vector<Event*> rstream = r.get_event_stream();
    std::vector<Event*>::iterator currEventR = rstream.begin();
    std::vector<UnitStream>::size_type currEventOutIndex = 0;
    for (std::vector<IntEvent>::iterator currEventD = d.stream.begin(); currEventD != d.stream.end(); ++currEventD) {
        while ((*currEventR)->get_timestamp() < currEventD->timestamp && currEventR != rstream.end()) {
            currEventR++;
        }
        while (currEventOutIndex < outstream.size() && outstream[currEventOutIndex].timestamp < outstream[currEventOutIndex].timestamp) {
            currEventOutIndex++;
        }
        if ((currEventD->timestamp == (*currEventR)->get_timestamp()) ||
            (currEventOutIndex < outstream.size() && (currEventD->timestamp == outstream[currEventOutIndex].timestamp))) {

            size_t target = currEventD->timestamp + currEventD->value;
            currEventR++;
            if (currEventR->get_timestamp() >= target) {
                UnitEvent node{target};
                outstream.push_back(node);
            }
            currEventR--;
        }
    }
    return make_shared<UnitStream>(outstream);
}

shared_ptr<IntStream> count(UnitStream& y) {
    shared_ptr<IntStream> outstream = make_shared<IntStream>();
    int32_t index = 0;
    for(std::vector<UnitEvent>::iterator it = y.stream.begin(); it != y.stream.end(); ++it) {
        outstream->stream.push_back((IntEvent){it->timestamp,index});
    }
    return outstream;
}


shared_ptr<UnitStream> merge(UnitStream& s1, UnitStream& s2) {
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
    return make_shared<UnitStream>(outstream);
}

// helpers for arithmetic functions

std::tuple<shared_ptr<IntStream>, shared_ptr<IntStream>> slift_streams(IntStream& x, IntStream& y) {
    shared_ptr<IntStream> lx = last(x,y);
    shared_ptr<IntStream> xp = merge(x,*lx);
    shared_ptr<IntStream> ly = last(y,x);
    shared_ptr<IntStream> yp = merge(y,*ly);
    return std::make_tuple(xp, yp);
}

shared_ptr<IntStream> const_lift(IntStream& x, int value, bool reverse, int(*op)(int, int)) {
    shared_ptr<IntStream> res = make_shared<IntStream>();
    for (auto &elem : x.stream) {
        int val = reverse? op(value, elem.value) : op(elem.value, value);
        res->stream.push_back(IntEvent(elem.timestamp, val));
    }
    return res;
}

shared_ptr<IntStream> lift(IntStream& x, IntStream& y, IntEvent*(*f)(IntEvent*, IntEvent*)) {
    std::vector<IntEvent>::iterator xs = x.stream.begin();
    std::vector<IntEvent>::iterator ys = y.stream.begin();

    std::vector<std::tuple<IntEvent*, IntEvent*>> lift_stream;
    while (xs != x.stream.end() || ys != y.stream.end()) {
        if (xs == x.stream.end()) {
            // tuple with x empty
            lift_stream.push_back(std::make_tuple(nullptr, &*ys));
            ys++;
        } else if (ys == y.stream.end()) {
            // tuple with y empty
            lift_stream.push_back(std::make_tuple(&*xs, nullptr));
            xs++;
        } else {
            int x_ts = xs->get_timestamp();
            int y_ts = ys->get_timestamp();
            if (x_ts == y_ts) {
                // tuple with both events
                lift_stream.push_back(std::make_tuple(&*xs, &*ys));
                xs++;
                ys++;
            } else if (x_ts < y_ts) {
                // tuple with y empty
                lift_stream.push_back(std::make_tuple(&*xs, nullptr));
                xs++;
            } else {
                // tuple with x empty
                lift_stream.push_back(std::make_tuple(nullptr, &*ys));
                ys++;
            }
        }
    }

    std::vector<IntEvent> ret_stream;
    IntEvent* x_ev = nullptr;
    IntEvent* y_ev = nullptr;
    for (auto & ev : lift_stream) {
        std::tie(x_ev, y_ev) = ev;
        IntEvent* ret_ev = f(x_ev, y_ev);
        if (ret_ev != nullptr) {
            ret_stream.push_back(*ret_ev);
        }
    }
    return make_shared<IntStream>(ret_stream);
}

shared_ptr<IntStream> slift(IntStream& x, IntStream& y, IntEvent*(*f)(IntEvent*, IntEvent*)) {
    shared_ptr<IntStream> xs, ys;
    std::tie(xs, ys) = slift_streams(x,y);
    return lift(*xs,*ys,f);
}

IntEvent* int_add(IntEvent* x, IntEvent* y) {
    if (x != nullptr && y != nullptr) { // assert x->timestamp == y->timestamp
        return new IntEvent(x->timestamp, x->value+y->value);
    } else {
        return nullptr;
    }
}
IntEvent* int_sub(IntEvent* x, IntEvent* y) {
    if (x != nullptr && y != nullptr) { // assert x->timestamp == y->timestamp
        return new IntEvent(x->timestamp, x->value-y->value);
    } else {
        return nullptr;
    }
}
IntEvent* int_mul(IntEvent* x, IntEvent* y) {
    if (x != nullptr && y != nullptr) { // assert x->timestamp == y->timestamp
        return new IntEvent(x->timestamp, x->value*y->value);
    } else {
        return nullptr;
    }
}
IntEvent* int_div(IntEvent* x, IntEvent* y) {
    if (x != nullptr && y != nullptr) { // assert x->timestamp == y->timestamp
        return new IntEvent(x->timestamp, x->value/y->value);
    } else {
        return nullptr;
    }
}
IntEvent* int_mod(IntEvent* x, IntEvent* y) {
    if (x != nullptr && y != nullptr) { // assert x->timestamp == y->timestamp
        return new IntEvent(x->timestamp, x->value%y->value);
    } else {
        return nullptr;
    }
}

IntEvent* int_merge(IntEvent* x, IntEvent* y) {
    if (x != nullptr) {
        return new IntEvent(x->timestamp, x->value);
    } else {
        return new IntEvent(y->timestamp, y->value);
    }
}

int int_add(int x, int y) { return x + y; }
int int_sub(int x, int y) { return x - y; }
int int_mul(int x, int y) { return x * y; }
int int_div(int x, int y) { return x / y; }
int int_mod(int x, int y) { return x % y; }

shared_ptr<IntStream> merge(IntStream& s1, IntStream& s2) {
    return lift(s1,s2,int_merge);
}

// stream + value
shared_ptr<IntStream> add(IntStream& x, int value){ return const_lift(x,value,false,int_add); }

// stream - value
shared_ptr<IntStream> sub1(IntStream& x, int value){
    return const_lift(x,value,false,int_sub);

}

// value - stream
shared_ptr<IntStream> sub2(IntStream& x, int value){
    return const_lift(x,value,true,int_sub);
}

// stream * value
shared_ptr<IntStream> mul(IntStream& x, int value){
    return const_lift(x,value,false,int_mul);
}

// stream / value
shared_ptr<IntStream> div1(IntStream& x, int value){
    return const_lift(x,value,false,int_div);
}

// value / stream
shared_ptr<IntStream> div2(IntStream& x, int value){
    return const_lift(x,value,true,int_div);
}

// stream % value
shared_ptr<IntStream> mod1(IntStream& x, int value){
    return const_lift(x,value,false,int_mod);
}

// value % stream
shared_ptr<IntStream> mod2(IntStream& x, int value){
    return const_lift(x,value,true,int_mod);
}

// x + y
shared_ptr<IntStream> add(IntStream& x, IntStream& y){
    return slift(x,y,int_add);
}

// x - y
shared_ptr<IntStream> sub(IntStream& x, IntStream& y){
    return slift(x,y,int_sub);
}

// x * y
shared_ptr<IntStream> mul(IntStream& x, IntStream& y){
    return slift(x,y,int_mul);
}

// x / y
shared_ptr<IntStream> div(IntStream& x, IntStream& y){
    return slift(x,y,int_div);
}

// x % y
shared_ptr<IntStream> mod(IntStream& x, IntStream& y){
    return slift(x,y,int_mod);
}
