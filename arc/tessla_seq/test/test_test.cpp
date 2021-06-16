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
        std::shared_ptr<IntStream> delayStreamIn = inReader.getIntStream("d");
        std::shared_ptr<UnitStream> resetStreamIn = inReader.getUnitStream("r");

        Reader outReader = Reader("../test/data/bt_delay.out");
        std::shared_ptr<UnitStream> intendedResult = outReader.getUnitStream("y");

        std::shared_ptr<UnitStream> result = delay(*delayStreamIn, *resetStreamIn);

        REQUIRE(*result == *intendedResult);
    }

    SECTION("bt_last") {
        Reader inReader = Reader("../test/data/bt_last.in");
        std::shared_ptr<IntStream> vStreamIn = inReader.getIntStream("v");
        std::shared_ptr<UnitStream> rStreamIn = inReader.getUnitStream("r");

        Reader outReader = Reader("../test/data/bt_last.out");
        std::shared_ptr<IntStream> intendedResult = outReader.getIntStream("y");

        std::shared_ptr<IntStream> result = last(*vStreamIn, *rStreamIn);

        REQUIRE(*result == *intendedResult);
    }

    SECTION("bt_merge") {
        Reader inReader = Reader("../test/data/bt_merge.in");
        std::shared_ptr<IntStream> xStreamIn = inReader.getIntStream("x");
        std::shared_ptr<IntStream> yStreamIn = inReader.getIntStream("y");

        Reader outReader = Reader("../test/data/bt_merge.out");
        std::shared_ptr<IntStream> intendedResult = outReader.getIntStream("z");

        std::shared_ptr<IntStream> result = merge(*xStreamIn, *yStreamIn);

        REQUIRE(*result == *intendedResult);
    }

    SECTION("bt_time") {
        SECTION("bt_time small dataset") {
            Reader inReader = Reader("../test/data/bt_time.in");
            std::shared_ptr<IntStream> xStreamIn = inReader.getIntStream("x");

            Reader outReader = Reader("../test/data/bt_time.out");
            std::shared_ptr<IntStream> intendedResult = outReader.getIntStream("x");

            std::shared_ptr<IntStream> result = time(*xStreamIn);

            REQUIRE(*result == *intendedResult);
        }

        SECTION("bt_time bigger dataset") {
            Reader inReader = Reader("../test/data/bt_time.bigger.in");
            std::shared_ptr<IntStream> xStreamIn = inReader.getIntStream("z");

            Reader outReader = Reader("../test/data/bt_time.bigger.out");
            std::shared_ptr<IntStream> intendedResult = outReader.getIntStream("y");

            std::shared_ptr<IntStream> result = time(*xStreamIn);

            REQUIRE(*result == *intendedResult);
        }
    }

}

TEST_CASE("Constant Test Cases") {

    SECTION("bt_addc") {
        Reader inReader = Reader("../test/data/bt_addc.in");
        std::shared_ptr<IntStream> inStream = inReader.getIntStream("x");
        Reader outReader = Reader("../test/data/bt_addc.out");
        std::shared_ptr<IntStream> intendedResult = outReader.getIntStream("y");

        std::shared_ptr<IntStream> result = add(*inStream, 1);
        REQUIRE(*result == *intendedResult);
    }

    SECTION("bt_adds") {
        Reader inReader = Reader("../test/data/bt_adds.in");
        std::shared_ptr<IntStream> inStream1 = inReader.getIntStream("x");
        std::shared_ptr<IntStream> inStream2 = inReader.getIntStream("y");
        Reader outReader = Reader("../test/data/bt_adds.out");
        std::shared_ptr<IntStream> intendedResult = outReader.getIntStream("z");

        std::shared_ptr<IntStream> result = add(*inStream1, *inStream2);
        REQUIRE(*result == *intendedResult);
    }

    SECTION("bt_subc") {
        Reader inReader = Reader("../test/data/bt_subc.in");
        std::shared_ptr<IntStream> inStream = inReader.getIntStream("x");
        Reader outReader = Reader("../test/data/bt_subc.out");
        std::shared_ptr<IntStream> intendedResult = outReader.getIntStream("y");

        std::shared_ptr<IntStream> result = sub1(*inStream, 3);
        REQUIRE(*result == *intendedResult);
    }

    SECTION("bt_mulc") {
        Reader inReader = Reader("../test/data/bt_mulc.in");
        std::shared_ptr<IntStream> inStream = inReader.getIntStream("x");
        Reader outReader = Reader("../test/data/bt_mulc.out");
        std::shared_ptr<IntStream> intendedResult = outReader.getIntStream("y");

        std::shared_ptr<IntStream> result = mul(*inStream, 4);
        REQUIRE(*result == *intendedResult);
    }

    SECTION("bt_divc") {
        Reader inReader = Reader("../test/data/bt_divc.in");
        std::shared_ptr<IntStream> inStream = inReader.getIntStream("x");
        Reader outReader = Reader("../test/data/bt_divc.out");
        std::shared_ptr<IntStream> intendedResult = outReader.getIntStream("y");

        std::shared_ptr<IntStream> result = div1(*inStream, 3);
        REQUIRE(*result == *intendedResult);
    }

    SECTION("bt_modc") {
        Reader inReader = Reader("../test/data/bt_modc.in");
        std::shared_ptr<IntStream> inStream = inReader.getIntStream("x");
        Reader outReader = Reader("../test/data/bt_modc.out");
        std::shared_ptr<IntStream> intendedResult = outReader.getIntStream("y");

        std::shared_ptr<IntStream> result = mod1(*inStream, 2);
        REQUIRE(*result == *intendedResult);
    }
}

TEST_CASE("Stream Arithmetic Test Cases (slift)") {

    SECTION("bt_adds") {
        Reader inReader = Reader("../test/data/bt_adds.in");
        std::shared_ptr<IntStream> inStream1 = inReader.getIntStream("x");
        std::shared_ptr<IntStream> inStream2 = inReader.getIntStream("y");
        Reader outReader = Reader("../test/data/bt_adds.out");
        std::shared_ptr<IntStream> intendedResult = outReader.getIntStream("z");

        std::shared_ptr<IntStream> result = add(*inStream1, *inStream2);
        REQUIRE(*result == *intendedResult);
    }

    SECTION("bt_subs") {
        Reader inReader = Reader("../test/data/bt_subs.in");
        std::shared_ptr<IntStream> inStream1 = inReader.getIntStream("x");
        std::shared_ptr<IntStream> inStream2 = inReader.getIntStream("y");
        Reader outReader = Reader("../test/data/bt_subs.out");
        std::shared_ptr<IntStream> intendedResult = outReader.getIntStream("z");

        std::shared_ptr<IntStream> result = sub(*inStream1, *inStream2);
        REQUIRE(*result == *intendedResult);
    }

    SECTION("bt_muls") {
        Reader inReader = Reader("../test/data/bt_muls.in");
        std::shared_ptr<IntStream> inStream1 = inReader.getIntStream("x");
        std::shared_ptr<IntStream> inStream2 = inReader.getIntStream("y");
        Reader outReader = Reader("../test/data/bt_muls.out");
        std::shared_ptr<IntStream> intendedResult = outReader.getIntStream("z");

        std::shared_ptr<IntStream> result = mul(*inStream1, *inStream2);
        REQUIRE(*result == *intendedResult);
    }

    SECTION("bt_divs") {
        Reader inReader = Reader("../test/data/bt_divs.in");
        std::shared_ptr<IntStream> inStream1 = inReader.getIntStream("x");
        std::shared_ptr<IntStream> inStream2 = inReader.getIntStream("y");
        Reader outReader = Reader("../test/data/bt_divs.out");
        std::shared_ptr<IntStream> intendedResult = outReader.getIntStream("z");

        std::shared_ptr<IntStream> result = div(*inStream1, *inStream2);
        REQUIRE(*result == *intendedResult);
    }

    SECTION("bt_mods") {
        Reader inReader = Reader("../test/data/bt_mods.in");
        std::shared_ptr<IntStream> inStream1 = inReader.getIntStream("x");
        std::shared_ptr<IntStream> inStream2 = inReader.getIntStream("y");
        Reader outReader = Reader("../test/data/bt_mods.out");
        std::shared_ptr<IntStream> intendedResult = outReader.getIntStream("z");

        std::shared_ptr<IntStream> result = mod(*inStream1, *inStream2);
        REQUIRE(*result == *intendedResult);
    }
}
