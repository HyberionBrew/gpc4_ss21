#include "catch2/catch.hpp"
#include "../src/InstrInterface.h"
#include "../src/Decode.h"

// To run this test, use ```make test TEST_SRC=test/test_vm.cpp```

TEST_CASE("Decode") {
    SECTION("Decode basictest.coil correctly") {
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
    }
}