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

// === BASIC OPERATIONS ===

// returns a new stream with unit event at time 0
UnitStream unit();

// returns a new stream with int event at time 0
IntStream def(int32_t val);

// time operation
IntStream time(Stream& s);

// last operation (sample v at ticks of r)
IntStream last(IntStream v, Stream& r);

// delay operation (delay events of d on ticks of r and close transitively)
UnitStream delay(IntStream d, Stream& r);

// count operation (running counter of unit events on x)
IntStream count(UnitStream x);

/**
 * Merges 2 input streams x and y, if oth events contain an event at the same timestamp,
 * x takes precedence
 * @param x
 * @param y
 * @return a new merged stream
 **/
IntStream merge(IntStream x, IntStream y);

// merge two unit streams
UnitStream merge(UnitStream x, UnitStream y);

// === ARITHMETIC OPERATIONS ===

// stream + value
IntStream add(IntStream x, int value);

// stream - value
IntStream sub1(IntStream x, int value);

// value - stream
IntStream sub2(IntStream x, int value);

// stream * value
IntStream mul(IntStream x, int value);

// stream / value
IntStream div1(IntStream x, int value);

// value / stream
IntStream div2(IntStream x, int value);

// stream % value
IntStream mod1(IntStream x, int value);

// value % stream
IntStream mod2(IntStream x, int value);

// all two stream operations are SLIFT

// x + y
IntStream add(IntStream x, IntStream y);

// x - y
IntStream sub(IntStream x, IntStream y);

// x * y
IntStream mul(IntStream x, IntStream y);

// x / y
IntStream div(IntStream x, IntStream y);

// x % y
IntStream mod(IntStream x, IntStream y);

#endif //GPU_TESSLA_STREAM_FUNCTIONS_H
