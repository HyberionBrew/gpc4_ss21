#include "catch2/catch.hpp"
#include "../src/Event.h"
#include "../src/Stream.h"
#include "../src/Reader.h"
#include "../src/Writer.h"
#include "../src/StreamFunctions.h"
#include "../src/Debug.h"

// Usage:
// make test TEST_SRC=test/test_test.cpp

TEST_CASE("Basic Stream Operations") {

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

TEST_CASE("Arithmetic Test Cases") {

    SECTION("bt_addc") {
        Reader inReader = Reader("test/data/bt_addc.in");
        IntStream inStream = inReader.getIntStream("x");
        Reader outReader = Reader("test/data/bt_addc.out");
        IntStream intendedResult = outReader.getIntStream("x");

        IntStream result = add(inStream, 1);
        REQUIRE(result == intendedResult);
    }

    /*
    SECTION("bt_subc") {
        Reader inReader = Reader("test/data/bt_subc.in");
        IntStream inStream = inReader.getIntStream("x");
        Reader outReader = Reader("test/data/bt_subc.out");
        IntStream intendedResult = outReader.getIntStream("x");

        IntStream result = sub1(inStream, 2);
        REQUIRE(result == intendedResult);
    }

    SECTION("bt_mulc") {
        Reader inReader = Reader("test/data/bt_subc.in");
        IntStream inStream = inReader.getIntStream("x");
        Reader outReader = Reader("test/data/bt_subc.out");
        IntStream intendedResult = outReader.getIntStream("x");

        IntStream result = mul(inStream, 3);
        REQUIRE(result == intendedResult);
    }
     */
}