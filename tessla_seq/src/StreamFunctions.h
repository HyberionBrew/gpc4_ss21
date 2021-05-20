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

UnitStream unit();
IntStream unit(int32_t val);

IntStream time(IntStream s);

IntStream last(IntStream v, UnitStream r);

UnitStream delay(IntStream d, UnitStream r);

IntStream count(UnitStream x);

/**
 * Merges 2 input streams x and y, if oth events contain an event at the same timestamp,
 * x takes precedence
 * @param x
 * @param y
 * @return a new merged stream
 */
IntStream merge(IntStream x, IntStream y);

UnitStream merge(UnitStream x, UnitStream y);

IntStream add(IntStream x, int value);

IntStream sub(IntStream x, int value);

IntStream mul(IntStream x, int value);

IntStream div(IntStream x, int value);

IntStream mod(IntStream x, int value);

IntStream add(IntStream x, IntStream y);

IntStream sub(IntStream x, IntStream y);

IntStream mul(IntStream x, IntStream y);

IntStream div(IntStream x, IntStream y);

IntStream mod(IntStream x, IntStream y);

#endif //GPU_TESSLA_STREAM_FUNCTIONS_H
