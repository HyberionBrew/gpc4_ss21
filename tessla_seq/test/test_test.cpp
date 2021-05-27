#include "catch2/catch.hpp"
#include "../src/Event.h"
#include "../src/Stream.h"
#include "../src/Reader.h"
#include "../src/Writer.h"
#include "../src/StreamFunctions.h"

// Usage:
// make test TEST_SRC=test/test_test.cpp

TEST_CASE("Basic Test Cases") {
    SECTION("bt_add1") {
        Reader inReader = Reader("test/data/bt_add1.in");
        IntStream inStream = inReader.getIntStream("x");
        Reader outReader = Reader("test/data/bt_add1.out");
        IntStream intendedResult = outReader.getIntStream("x");

        /* TODO: Implement add
        IntStream result = add(inStream, 1);

        REQUIRE(result == intendedResult);
        */

    }

    SECTION("bt_delay") {
        Reader inReader = Reader("test/data/bt_delay.in");
        IntStream delayStreamIn = inReader.getIntStream("d");
        UnitStream resetStreamIn = inReader.getUnitStream("r");

        Reader outReader = Reader("test/data/bt_delay.out");
        UnitStream intendedResult = outReader.getUnitStream("y");

        UnitStream result = delay(delayStreamIn, resetStreamIn);

        REQUIRE(result == intendedResult);
    }

    SECTION("bt_last") {
        Reader inReader = Reader("test/data/bt_last.in");
        IntStream vStreamIn = inReader.getIntStream("v");
        UnitStream rStreamIn = inReader.getUnitStream("r");

        Reader outReader = Reader("test/data/bt_last.out");
        IntStream intendedResult = outReader.getIntStream("y");

        IntStream result = last(vStreamIn, rStreamIn);

        REQUIRE(result == intendedResult);
    }

    SECTION("bt_merge") {
        Reader inReader = Reader("test/data/bt_merge.in");
        IntStream xStreamIn = inReader.getIntStream("x");
        IntStream yStreamIn = inReader.getIntStream("y");

        Reader outReader = Reader("test/data/bt_merge.out");
        IntStream intendedResult = outReader.getIntStream("z");

        IntStream result = merge(xStreamIn, yStreamIn);

        REQUIRE(result == intendedResult);
    }

    SECTION("bt_time") {
        Reader inReader = Reader("test/data/bt_time.in");
        IntStream xStreamIn = inReader.getIntStream("x");

        Reader outReader = Reader("test/data/bt_time.out");
        IntStream intendedResult = outReader.getIntStream("x");

        IntStream result = time(xStreamIn);

        REQUIRE(result == intendedResult);
    }

}