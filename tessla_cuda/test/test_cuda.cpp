#include <string.h>
#include "../../test/catch2/catch.hpp"
#include "../src/Stream.cuh"
#include "../src/StreamFunctions.cuh"


TEST_CASE("Basic tests") {
    SECTION("time()") {
        int size = 4;
        int sizeAllocated = (size_t) size * sizeof(int);

        int inTimestamps[4];
        int inInts[4];
        int outTimestamps[4];
        int outInts[4];

        inTimestamps[0] = 2;
        inTimestamps[1] = 5;
        inTimestamps[2] = 6;
        inTimestamps[3] = 9;

        inInts[0] = 1;
        inInts[1] = 3;
        inInts[2] = 9;
        inInts[3] = 4;

        outTimestamps[0] = 2;
        outTimestamps[1] = 5;
        outTimestamps[2] = 6;
        outTimestamps[3] = 9;

        outInts[0] = 2;
        outInts[1] = 5;
        outInts[2] = 6;
        outInts[3] = 9;

        int *host_timestampOut = (int *) malloc(size * sizeof(int));
        int *host_valueOut = (int*) malloc(size * sizeof(int));
        memset(host_timestampOut, 0, sizeAllocated);
        memset(host_valueOut, 0, sizeAllocated);

        IntStream inputStream(inTimestamps, inInts, size);
        IntStream outputStream(host_timestampOut, host_valueOut, size);

        inputStream.copy_to_device();
        outputStream.copy_to_device();

        time(&inputStream, &outputStream, 0);

        outputStream.copy_to_host();

        REQUIRE(outputStream.host_timestamp[0] == outTimestamps[0]);
        REQUIRE(outputStream.host_values[0] == outInts[0]);
        REQUIRE(outputStream.host_timestamp[1] == outTimestamps[1]);
        REQUIRE(outputStream.host_values[1] == outInts[1]);
        REQUIRE(outputStream.host_timestamp[2] == outTimestamps[2]);
        REQUIRE(outputStream.host_values[2] == outInts[2]);
        REQUIRE(outputStream.host_timestamp[3] == outTimestamps[3]);
        REQUIRE(outputStream.host_values[3] == outInts[3]);

        // Cleanup
        inputStream.free_device();
        outputStream.free_device();
        free(host_valueOut);
        free(host_timestampOut);
    }
    
    SECTION("last()") {
        int size = 5;
        int sizeAllocated = (size_t) size * sizeof(int);

        int inTimestamps[3];
        int inInts[3];
        int unitInTimestamps[5];
        int outTimestamps[4];
        int outInts[4];

        inTimestamps[0] = 3;
        inTimestamps[1] = 6;
        inTimestamps[2] = 8;

        inInts[0] = 1;
        inInts[1] = 3;
        inInts[2] = 6;

        unitInTimestamps[0] = 0;
        unitInTimestamps[1] = 2;
        unitInTimestamps[2] = 4;
        unitInTimestamps[3] = 5;
        unitInTimestamps[4] = 9;

        outTimestamps[0] = 4;
        outTimestamps[1] = 5;
        outTimestamps[2] = 9;

        outInts[0] = 1;
        outInts[1] = 1;
        outInts[2] = 6;

        int *host_timestampOut = (int *) malloc(size * sizeof(int));
        int *host_valueOut = (int*) malloc(size * sizeof(int));
        memset(host_timestampOut, 0, sizeAllocated);
        memset(host_valueOut, 0, sizeAllocated);

        IntStream inputStream(inTimestamps, inInts, size);
        UnitStream unitInputStream(unitInTimestamps, size);
        IntStream outputStream(host_timestampOut, host_valueOut, size);

        inputStream.copy_to_device();
        unitInputStream.copy_to_device();
        outputStream.copy_to_device();

        //time(&inputStream, &outputStream, 0);
        last(&inputStream, &unitInputStream, &outputStream, 0);

        outputStream.copy_to_host();

        REQUIRE(outputStream.host_timestamp[0] == outTimestamps[0]);
        REQUIRE(outputStream.host_values[0] == outInts[0]);
        REQUIRE(outputStream.host_timestamp[1] == outTimestamps[1]);
        REQUIRE(outputStream.host_values[1] == outInts[1]);
        REQUIRE(outputStream.host_timestamp[2] == outTimestamps[2]);
        REQUIRE(outputStream.host_values[2] == outInts[2]);

        // Cleanup
        inputStream.free_device();
        unitInputStream.free_device();
        outputStream.free_device();
        free(host_valueOut);
        free(host_timestampOut);
    }
}
