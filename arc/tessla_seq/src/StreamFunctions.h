//
// Created by daniel on 20.05.21.
//

#ifndef GPU_TESSLA_STREAM_FUNCTIONS_H
#define GPU_TESSLA_STREAM_FUNCTIONS_H

#include "Stream.h"
#include "Event.h"
#include <vector>
#include <iterator>
#include <cstdint>
#include <iostream>
#include <memory>

// === BASIC OPERATIONS ===

// returns a new stream with unit event at time 0
std::shared_ptr<UnitStream> unit();

// returns a new stream with int event at time 0
std::shared_ptr<IntStream> def(int32_t val);

// time operation
std::shared_ptr<IntStream> time(Stream& s);

// last operation (sample v at ticks of r)
std::shared_ptr<IntStream> last(IntStream& v, Stream& r);

// delay operation (delay events of d on ticks of r and close transitively)
std::shared_ptr<UnitStream> delay(IntStream& d, Stream& r);

// count operation (running counter of unit events on x)
std::shared_ptr<IntStream> count(UnitStream& x);

/**
 * Merges 2 input streams x and y, if oth events contain an event at the same timestamp,
 * x takes precedence
 * @param x
 * @param y
 * @return a new merged stream
 **/
std::shared_ptr<IntStream> merge(IntStream& x, IntStream& y);

// merge two unit streams
std::shared_ptr<UnitStream> merge(UnitStream& x, UnitStream& y);

// === ARITHMETIC OPERATIONS ===

// stream + value
std::shared_ptr<IntStream> add(IntStream& x, int value);

// stream - value
std::shared_ptr<IntStream> sub1(IntStream& x, int value);

// value - stream
std::shared_ptr<IntStream> sub2(IntStream& x, int value);

// stream * value
std::shared_ptr<IntStream> mul(IntStream& x, int value);

// stream / value
std::shared_ptr<IntStream> div1(IntStream& x, int value);

// value / stream
std::shared_ptr<IntStream> div2(IntStream& x, int value);

// stream % value
std::shared_ptr<IntStream> mod1(IntStream& x, int value);

// value % stream
std::shared_ptr<IntStream> mod2(IntStream& x, int value);

// all two stream operations are SLIFT

// x + y
std::shared_ptr<IntStream> add(IntStream& x, IntStream& y);

// x - y
std::shared_ptr<IntStream> sub(IntStream& x, IntStream& y);

// x * y
std::shared_ptr<IntStream> mul(IntStream& x, IntStream& y);

// x / y
std::shared_ptr<IntStream> div(IntStream& x, IntStream& y);

// x % y
std::shared_ptr<IntStream> mod(IntStream& x, IntStream& y);

#endif //GPU_TESSLA_STREAM_FUNCTIONS_H
