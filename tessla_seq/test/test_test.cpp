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
        Reader inReader = Reader("../test/data/bt_delay.in");
        IntStream delayStreamIn = inReader.getIntStream("d");
        UnitStream resetStreamIn = inReader.getUnitStream("r");

        Reader outReader = Reader("../test/data/bt_delay.out");
        UnitStream intendedResult = outReader.getUnitStream("y");

        UnitStream result = delay(delayStreamIn, resetStreamIn);

        REQUIRE(result == intendedResult);
    }

    SECTION("bt_last") {
        Reader inReader = Reader("../test/data/bt_last.in");
        IntStream vStreamIn = inReader.getIntStream("v");
        UnitStream rStreamIn = inReader.getUnitStream("r");

        Reader outReader = Reader("../test/data/bt_last.out");
        IntStream intendedResult = outReader.getIntStream("y");

        IntStream result = last(vStreamIn, rStreamIn);

        REQUIRE(result == intendedResult);
    }

    SECTION("bt_merge") {
        Reader inReader = Reader("../test/data/bt_merge.in");
        IntStream xStreamIn = inReader.getIntStream("x");
        IntStream yStreamIn = inReader.getIntStream("y");

        Reader outReader = Reader("../test/data/bt_merge.out");
        IntStream intendedResult = outReader.getIntStream("z");

        IntStream result = merge(xStreamIn, yStreamIn);

        REQUIRE(result == intendedResult);
    }

    SECTION("bt_time") {
        SECTION("bt_time small dataset") {
            Reader inReader = Reader("../test/data/bt_time.in");
            IntStream xStreamIn = inReader.getIntStream("x");

            Reader outReader = Reader("../test/data/bt_time.out");
            IntStream intendedResult = outReader.getIntStream("x");

            IntStream result = time(xStreamIn);

            REQUIRE(result == intendedResult);
        }

        SECTION("bt_time bigger dataset") {
            Reader inReader = Reader("../test/data/bt_time.bigger.in");
            IntStream xStreamIn = inReader.getIntStream("z");

            Reader outReader = Reader("../test/data/bt_time.bigger.out");
            IntStream intendedResult = outReader.getIntStream("y");

            IntStream result = time(xStreamIn);

            REQUIRE(result == intendedResult);
        }
    }

}

TEST_CASE("Constant Test Cases") {

    SECTION("bt_addc") {
        Reader inReader = Reader("../test/data/bt_addc.in");
        IntStream inStream = inReader.getIntStream("x");
        Reader outReader = Reader("../test/data/bt_addc.out");
        IntStream intendedResult = outReader.getIntStream("y");

        IntStream result = add(inStream, 1);
        REQUIRE(result == intendedResult);
    }

    SECTION("bt_adds") {
        Reader inReader = Reader("../test/data/bt_adds.in");
        IntStream inStream1 = inReader.getIntStream("x");
        IntStream inStream2 = inReader.getIntStream("y");
        Reader outReader = Reader("../test/data/bt_adds.out");
        IntStream intendedResult = outReader.getIntStream("z");

        IntStream result = add(inStream1, inStream2);
        REQUIRE(result == intendedResult);
    }

    SECTION("bt_subc") {
        Reader inReader = Reader("../test/data/bt_subc.in");
        IntStream inStream = inReader.getIntStream("x");
        Reader outReader = Reader("../test/data/bt_subc.out");
        IntStream intendedResult = outReader.getIntStream("y");

        IntStream result = sub1(inStream, 3);
        REQUIRE(result == intendedResult);
    }

    SECTION("bt_mulc") {
        Reader inReader = Reader("../test/data/bt_mulc.in");
        IntStream inStream = inReader.getIntStream("x");
        Reader outReader = Reader("../test/data/bt_mulc.out");
        IntStream intendedResult = outReader.getIntStream("y");

        IntStream result = mul(inStream, 4);
        REQUIRE(result == intendedResult);
    }

    SECTION("bt_divc") {
        Reader inReader = Reader("../test/data/bt_divc.in");
        IntStream inStream = inReader.getIntStream("x");
        Reader outReader = Reader("../test/data/bt_divc.out");
        IntStream intendedResult = outReader.getIntStream("y");

        IntStream result = div1(inStream, 3);
        REQUIRE(result == intendedResult);
    }

    SECTION("bt_modc") {
        Reader inReader = Reader("../test/data/bt_modc.in");
        IntStream inStream = inReader.getIntStream("x");
        Reader outReader = Reader("../test/data/bt_modc.out");
        IntStream intendedResult = outReader.getIntStream("y");

        IntStream result = mod1(inStream, 2);
        REQUIRE(result == intendedResult);
    }
}

TEST_CASE("Stream Arithmetic Test Cases (slift)") {

    SECTION("bt_adds") {
        Reader inReader = Reader("../test/data/bt_adds.in");
        IntStream inStream1 = inReader.getIntStream("x");
        IntStream inStream2 = inReader.getIntStream("y");
        Reader outReader = Reader("../test/data/bt_adds.out");
        IntStream intendedResult = outReader.getIntStream("z");

        IntStream result = add(inStream1, inStream2);
        REQUIRE(result == intendedResult);
    }

    SECTION("bt_subs") {
        Reader inReader = Reader("../test/data/bt_subs.in");
        IntStream inStream1 = inReader.getIntStream("x");
        IntStream inStream2 = inReader.getIntStream("y");
        Reader outReader = Reader("../test/data/bt_subs.out");
        IntStream intendedResult = outReader.getIntStream("z");

        IntStream result = sub(inStream1, inStream2);
        REQUIRE(result == intendedResult);
    }

    SECTION("bt_muls") {
        Reader inReader = Reader("../test/data/bt_muls.in");
        IntStream inStream1 = inReader.getIntStream("x");
        IntStream inStream2 = inReader.getIntStream("y");
        Reader outReader = Reader("../test/data/bt_muls.out");
        IntStream intendedResult = outReader.getIntStream("z");

        IntStream result = mul(inStream1, inStream2);
        REQUIRE(result == intendedResult);
    }

    SECTION("bt_divs") {
        Reader inReader = Reader("../test/data/bt_divs.in");
        IntStream inStream1 = inReader.getIntStream("x");
        IntStream inStream2 = inReader.getIntStream("y");
        Reader outReader = Reader("../test/data/bt_divs.out");
        IntStream intendedResult = outReader.getIntStream("z");

        IntStream result = div(inStream1, inStream2);
        REQUIRE(result == intendedResult);
    }

    SECTION("bt_mods") {
        Reader inReader = Reader("../test/data/bt_mods.in");
        IntStream inStream1 = inReader.getIntStream("x");
        IntStream inStream2 = inReader.getIntStream("y");
        Reader outReader = Reader("../test/data/bt_mods.out");
        IntStream intendedResult = outReader.getIntStream("z");

        IntStream result = mod(inStream1, inStream2);
        REQUIRE(result == intendedResult);
    }
}
