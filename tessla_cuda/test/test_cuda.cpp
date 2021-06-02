#include <string.h>
#include "../../test/catch2/catch.hpp"
#include "../src/Reader.cuh"
#include "../src/Stream.cuh"
#include "../src/StreamFunctions.cuh"


TEST_CASE("Basic tests") {
    SECTION("time()") {
        // Read input and correct output data
        Reader inReader = Reader("../test/data/bt_time.in");
        IntStream inputStream = inReader.getIntStream("x");
        Reader outReader = Reader("../test/data/bt_time.out");
        IntStream CORRECT_STREAM = outReader.getIntStream("x");

        // Prepare empty output stream to fill
        int size = CORRECT_STREAM.size;
        int sizeAllocated = (size_t) size * sizeof(int);
        int *host_timestampOut = (int *) malloc(size * sizeof(int));
        int *host_valueOut = (int*) malloc(size * sizeof(int));
        memset(host_timestampOut, 0, sizeAllocated);
        memset(host_valueOut, 0, sizeAllocated);
        IntStream outputStream(host_timestampOut, host_valueOut, size);

        // Run kernel
        inputStream.copy_to_device();
        outputStream.copy_to_device();
        time(&inputStream, &outputStream, 0);
        outputStream.copy_to_host();

        // Compare kernel result to correct data
        for (int i = 0; i < size; i++) {
            REQUIRE(outputStream.host_timestamp[i] == CORRECT_STREAM.host_timestamp[i]);
            REQUIRE(outputStream.host_values[i] == CORRECT_STREAM.host_values[i]);
        }

        // Cleanup
        inputStream.free_device();
        outputStream.free_device();
        free(host_valueOut);
        free(host_timestampOut);
    }
    
    SECTION("last()") {
        // Read input and correct output data
        Reader inReader = Reader("../test/data/bt_last.in");
        IntStream inputStreamV = inReader.getIntStream("v");
        UnitStream inputStreamR = inReader.getUnitStream("r");
        Reader outReader = Reader("../test/data/bt_last.out");
        IntStream CORRECT_STREAM = outReader.getIntStream("y");

        // Prepare empty output stream to fill
        int size = CORRECT_STREAM.size;
        int sizeAllocated = (size_t) size * sizeof(int);
        int *host_timestampOut = (int *) malloc(size * sizeof(int));
        int *host_valueOut = (int*) malloc(size * sizeof(int));
        memset(host_timestampOut, 0, sizeAllocated);
        memset(host_valueOut, 0, sizeAllocated);
        IntStream outputStream(host_timestampOut, host_valueOut, size);

        // Run kernel
        inputStreamV.copy_to_device();
        inputStreamR.copy_to_device();
        outputStream.copy_to_device();
        //last(&inputStreamV, &inputStreamR, &outputStream, 0); // TODO: Not working right now
        outputStream.copy_to_host();

        // Compare kernel result to correct data
        for (int i = 0; i < size; i++) {
            REQUIRE(outputStream.host_timestamp[i] == CORRECT_STREAM.host_timestamp[i]);
            REQUIRE(outputStream.host_values[i] == CORRECT_STREAM.host_values[i]);
        }

        // Cleanup
        inputStreamV.free_device();
        inputStreamR.free_device();
        outputStream.free_device();
        free(host_valueOut);
        free(host_timestampOut);
    }
}

TEST_CASE("Reader (bt_delay.in)") {
    Reader reader = Reader("../test/data/bt_delay.in");

    SECTION("Read existing IntStream correctly") {
        int CORRECT_STREAM_SIZE = 6;
        int CORRECT_STREAM_TIMESTAMPS[CORRECT_STREAM_SIZE] = {1, 2, 3, 4, 6, 7};
        int CORRECT_STREAM_INTS[CORRECT_STREAM_SIZE] = {1, 2, 4, 1, 2, 3};

        IntStream dStream = reader.getIntStream("d");
        REQUIRE(dStream.size == CORRECT_STREAM_SIZE);
        for (int i = 0; i < CORRECT_STREAM_SIZE; i++) {
            REQUIRE(dStream.host_timestamp[i] == CORRECT_STREAM_TIMESTAMPS[i]);
            REQUIRE(dStream.host_values[i] == CORRECT_STREAM_INTS[i]);
        }
    }

    SECTION("Read existing UnitStream correctly") {
        int CORRECT_STREAM_SIZE = 3;
        int CORRECT_STREAM_TIMESTAMPS[CORRECT_STREAM_SIZE] = {2, 7, 8};

        UnitStream dStream = reader.getUnitStream("r");
        REQUIRE(dStream.size == CORRECT_STREAM_SIZE);
        for (int i = 0; i < CORRECT_STREAM_SIZE; i++) {
            REQUIRE(dStream.host_timestamp[i] == CORRECT_STREAM_TIMESTAMPS[i]);
        }
    }

}
