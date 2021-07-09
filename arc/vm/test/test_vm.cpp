#include "catch2/catch.hpp"
#include "../src/InstrInterface.h"
#include "../src/runner.h"
#include <getopt.h>
#include "../src/Decode.h"

// To run this test, use ```make test TEST_SRC=test/test_vm.cpp```

TEST_CASE("Decode") {
    SECTION("Decode basictest.coil correctly") {
        /*
        InstrInterface instrInterface = InstrInterface();
        Decode decoder = Decode("test/basictest.coil", instrInterface);

        REQUIRE(decoder.majorV == 1);
        REQUIRE(decoder.minorV == 0);
        REQUIRE(decoder.wideAddresses == false);

        REQUIRE(decoder.in_streams[0].name.compare("x"));
        REQUIRE(decoder.in_streams[0].regname == 0);
        REQUIRE(decoder.in_streams[0].type != unit);
        REQUIRE(decoder.in_streams[1].name.compare("e"));
        REQUIRE(decoder.in_streams[1].regname == 1);
        REQUIRE(decoder.in_streams[1].type != unit);
        REQUIRE(decoder.in_streams[2].name.compare("g"));
        REQUIRE(decoder.in_streams[2].regname == 2);
        REQUIRE(decoder.in_streams[2].type != unit);

        REQUIRE(decoder.out_streams[0].name.compare("z"));
        REQUIRE(decoder.out_streams[0].regname == 4);
        REQUIRE(decoder.out_streams[1].name.compare("d"));
        REQUIRE(decoder.out_streams[1].regname == 0);
        REQUIRE(decoder.out_streams[2].name.compare("h"));
        REQUIRE(decoder.out_streams[2].regname == 3);
         */
    }
}

TEST_CASE("Random Tests") {
    const char* testf_path = "arc/vm/test/arc_input/";
#define TESTF_PATH "arc/vm/test/arc_input/"
#define TF1 "test_1337_sl10_tl100"
#define TF2 "test_lel_sl100_tl2500"
#define TF3 "default"
#define TF4 "test_peda_sl200_tl250_benchmark"
#define COIL_F ".coil"
#define IN_F ".in"
#define concat(f,s,t) f s t

    int argc = 5;
    const char* a1 = "arc";
    const char* a2 = "-s";
    const char* a3 = "-v";

    SECTION("test_1337_sl_10") {
        const char* a4 = concat(TESTF_PATH, TF1, COIL_F);
        const char* a5 = concat(TESTF_PATH, TF1, IN_F);
        const char* argv[] = {a1, a2, a3, a4, a5};
        run(argc, const_cast<char **>(argv));
    }

    SECTION("test_lel_sl10_tl2500") {
        optind = 0;
        const char* a4 = concat(TESTF_PATH, TF2, COIL_F);
        const char* a5 = concat(TESTF_PATH, TF2, IN_F);
        const char* argv[] = {a1, a2, a3, a4, a5};
        run(argc, const_cast<char **>(argv));
    }

    SECTION("default (assertion violation after default)") {
        optind = 0;
        const char* a4 = concat(TESTF_PATH, TF3, COIL_F);
        const char* a5 = concat(TESTF_PATH, TF3, IN_F);
        const char* argv[] = {a1, a2, a3, a4, a5};
        run(argc, const_cast<char **>(argv));
    }

    SECTION("test_peda_sl200_tl250 (SIGSEGV)") {
        optind = 0;
        const char* a4 = concat(TESTF_PATH, TF4, COIL_F);
        const char* a5 = concat(TESTF_PATH, TF4, IN_F);
        const char* argv[] = {a1, a2, a3, a4, a5};
        run(argc, const_cast<char **>(argv));
    }
}