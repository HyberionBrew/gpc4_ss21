//
// Created by daniel on 20.05.21.
//

#include "Stream.h"

IntStream::IntStream(std::vector<IntEvent> stream) {
    this->stream = stream;
}

IntStream::IntStream() {}

std::vector<Event*> IntStream::get_event_stream() {
    std::vector<Event*> vec;
    for (auto & elem : this->stream) {
        vec.push_back(&elem);
    }
    return vec;
}

UnitStream::UnitStream(std::vector<UnitEvent> stream) {
    this->stream = stream;
}

UnitStream::UnitStream() {}

std::vector<Event*> UnitStream::get_event_stream() {
    std::vector<Event*> vec;
    for (auto & elem : this->stream) {
        vec.push_back(&elem);
    }
    return vec;
}

OutputStream::OutputStream(std::string name, Stream& stream) : name(name), stream(stream) { }

Stream::~Stream() {};
//LockQueue& operator=(const LockQueue & q);  // Assignment Operator

/*
 * // Assignment Operator
LockQueue& LockQueue::operator=( const LockQueue & q ) {
    head=q.head;
    tail=q.tail;
    size=q.size;
    items = std::unique_ptr<int[]>(new int[q.size]);
    for (int i = 0; i < q.size; i++){
        items.get()[i] = q.items.get()[i];
    }
}
 */
bool operator==(const IntStream& lhs, const IntStream& rhs)
{
    return lhs.stream == rhs.stream;
}

bool operator==(const UnitStream& lhs, const UnitStream& rhs)
{
    return lhs.stream == rhs.stream;
}

/*
bool Stream::is_equal_to(emtream other) {
    if (this->get_type != other->get_type()) {
        return false;
    }
    if (tllllllhis->get_type() == INT_STREAM) {
        return this->get_IntStream() == other.get_IntStream();
    } else if (this->get_type() == UNIT_STREAM) {
        return this->get_UnitStream() == other.get_UnitStream();
    }
}*/