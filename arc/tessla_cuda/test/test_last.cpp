//
// Created by fabian on 24/06/2021.
//

#include <iostream>
#include <string>
#include <string.h>
#include <vector>
#include "../../test/catch2/catch.hpp"
#include <GPUReader.cuh>
#include <GPUStream.cuh>

#include <chrono>
#include <fstream>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <cuda_profiler_api.h>

#include <StreamFunctions.cuh>
#include <StreamFunctionsThrust.cuh>

static std::string TESTDATA_PATH = "test/data/";

TEST_CASE("last_thrust()") {
    SECTION("last_thrust() tuwel example") {
        GPUReader inReader = GPUReader(TESTDATA_PATH + "bt_last.in");
        GPUReader outReader = GPUReader(TESTDATA_PATH + "bt_last.out");
        std::shared_ptr<GPUIntStream> inputStreamV = inReader.getIntStream("v");
        std::shared_ptr<GPUUnitStream> inputStreamR = inReader.getUnitStream("r");
        std::shared_ptr<GPUIntStream> CORRECT_STREAM = outReader.getIntStream("y");

        inputStreamV->copy_to_device();
        inputStreamR->copy_to_device();

        std::shared_ptr<GPUIntStream> outputStream = last_thrust(inputStreamV, inputStreamR, 0);
        outputStream->copy_to_host();

        // Compare kernel result to correct data
        std::vector<int> kernelTimestamps(outputStream->host_timestamp + *(outputStream->host_offset),
                                          outputStream->host_timestamp + outputStream->size);
        std::vector<int> kernelValues(outputStream->host_values + *(outputStream->host_offset),
                                      outputStream->host_values + outputStream->size);
        std::vector<int> correctTimestamps(CORRECT_STREAM->host_timestamp,
                                           CORRECT_STREAM->host_timestamp + CORRECT_STREAM->size);
        std::vector<int> correctValues(CORRECT_STREAM->host_values, CORRECT_STREAM->host_values + CORRECT_STREAM->size);
        REQUIRE(kernelTimestamps == correctTimestamps);
        REQUIRE(kernelValues == correctValues);

        // Cleanup
        inputStreamV->free_device();
        inputStreamR->free_device();
        outputStream->free_device();
        inputStreamV->free_host();
        outputStream->free_host();
        inputStreamR->free_host();
    }

    SECTION("last_thrust() large random example") {
        // Read input and correct output data
        GPUReader inReader = GPUReader(TESTDATA_PATH + "last_test2.in");
        GPUReader outReader = GPUReader(TESTDATA_PATH + "last_test2.out");
        std::shared_ptr<GPUIntStream> inputStreamV = inReader.getIntStream("z");
        std::shared_ptr<GPUUnitStream> inputStreamR = inReader.getUnitStream("a");
        std::shared_ptr<GPUIntStream> CORRECT_STREAM = outReader.getIntStream("y");
        std::shared_ptr<GPUIntStream> outputStream;

        inputStreamV->copy_to_device();
        inputStreamR->copy_to_device();

        outputStream = last_thrust(inputStreamV, inputStreamR, 0);
        outputStream->copy_to_host();

        // Compare kernel result to correct data
        std::vector<int> kernelTimestamps(outputStream->host_timestamp+*(outputStream->host_offset), outputStream->host_timestamp+outputStream->size);
        std::vector<int> kernelValues(outputStream->host_values+*(outputStream->host_offset), outputStream->host_values+outputStream->size);
        std::vector<int> correctTimestamps(CORRECT_STREAM->host_timestamp, CORRECT_STREAM->host_timestamp + CORRECT_STREAM->size);
        std::vector<int> correctValues(CORRECT_STREAM->host_values, CORRECT_STREAM->host_values + CORRECT_STREAM->size);
        REQUIRE(kernelTimestamps == correctTimestamps);
        REQUIRE(kernelValues == correctValues);

        // Cleanup
        inputStreamV->free_device();
        inputStreamR->free_device();
        outputStream->free_device();
        inputStreamV->free_host();
        inputStreamR->free_host();
        outputStream->free_host();
    }

    SECTION("last() twice test with no invalids") {
        GPUReader inReader = GPUReader(TESTDATA_PATH + "last_test3.in");
        GPUReader outReader = GPUReader(TESTDATA_PATH + "last_test3.out");
        std::shared_ptr<GPUIntStream> inputStreamV = inReader.getIntStream("z");
        std::shared_ptr<GPUUnitStream> inputStream2 = inReader.getUnitStream("a");
        std::shared_ptr<GPUUnitStream> inputStreamR = inReader.getUnitStream("x");
        std::shared_ptr<GPUIntStream> CORRECT_STREAM = outReader.getIntStream("o");

        inputStreamV->copy_to_device();
        inputStreamR->copy_to_device();
        inputStream2->copy_to_device();

        // Run kernel
        std::shared_ptr<GPUIntStream> intermediateStream = last_thrust(inputStreamV, inputStreamR, 0);
        intermediateStream->copy_to_host();
        std::shared_ptr<GPUIntStream> outputStream = last_thrust(intermediateStream, inputStream2, 0);
        outputStream->copy_to_host();

        // Compare kernel result to correct data
        std::vector<int> kernelTimestamps(outputStream->host_timestamp+*(outputStream->host_offset), outputStream->host_timestamp+outputStream->size);
        std::vector<int> kernelValues(outputStream->host_values+*(outputStream->host_offset), outputStream->host_values+outputStream->size);
        std::vector<int> correctTimestamps(CORRECT_STREAM->host_timestamp, CORRECT_STREAM->host_timestamp + CORRECT_STREAM->size);
        std::vector<int> correctValues(CORRECT_STREAM->host_values, CORRECT_STREAM->host_values + CORRECT_STREAM->size);
        REQUIRE(kernelTimestamps == correctTimestamps);
        REQUIRE(kernelValues == correctValues);

        // Cleanup
        inputStreamV->free_device();
        inputStreamR->free_device();
        intermediateStream->free_device();
        outputStream->free_device();
        inputStream2->free_device();
        inputStreamV->free_host();
        inputStreamR->free_host();
        intermediateStream->free_host();
        outputStream->free_host();
        inputStream2->free_host();
    }

    SECTION("last() twice test with invalids in Unit Stream") {
        // Read input and correct output data
        GPUReader inReader = GPUReader(TESTDATA_PATH + "last_test4.in");
        GPUReader outReader = GPUReader(TESTDATA_PATH + "last_test4.out");
        std::shared_ptr<GPUIntStream> inputStreamV = inReader.getIntStream("z");
        std::shared_ptr<GPUUnitStream> inputStream2 = inReader.getUnitStream("x");
        std::shared_ptr<GPUUnitStream> inputStreamR = inReader.getUnitStream("a");
        std::shared_ptr<GPUIntStream> CORRECT_STREAM = outReader.getIntStream("o");

        inputStreamV->copy_to_device();
        inputStreamR->copy_to_device();
        inputStream2->copy_to_device();

        std::shared_ptr<GPUIntStream> intermediateStream = last_thrust(inputStreamV, inputStreamR,  0);
        std::shared_ptr<GPUIntStream> outputStream = last_thrust(intermediateStream, inputStream2, 0);
        intermediateStream->copy_to_host();
        outputStream->copy_to_host();

        // Compare kernel result to correct data
        std::vector<int> kernelTimestamps(outputStream->host_timestamp+*(outputStream->host_offset), outputStream->host_timestamp+outputStream->size);
        std::vector<int> kernelValues(outputStream->host_values+*(outputStream->host_offset), outputStream->host_values+outputStream->size);
        std::vector<int> correctTimestamps(CORRECT_STREAM->host_timestamp, CORRECT_STREAM->host_timestamp + CORRECT_STREAM->size);
        std::vector<int> correctValues(CORRECT_STREAM->host_values, CORRECT_STREAM->host_values + CORRECT_STREAM->size);
        REQUIRE(kernelTimestamps == correctTimestamps);
        REQUIRE(kernelValues == correctValues);

        // Cleanup
        inputStreamV->free_device();
        inputStreamR->free_device();
        intermediateStream->free_device();
        outputStream->free_device();
        inputStream2->free_device();
        inputStreamV->free_host();
        inputStreamR->free_host();
        intermediateStream->free_host();
        outputStream->free_host();
        inputStream2->free_host();
    }

    SECTION("last empty"){
        //reading only invalid streams (they are empty)
        GPUReader inReader = GPUReader(TESTDATA_PATH + "bt_last.in");
        GPUReader outReader = GPUReader(TESTDATA_PATH + "bt_last.out");
        std::shared_ptr<GPUIntStream> inputStreamV = inReader.getIntStream("v");
        std::shared_ptr<GPUUnitStream> inputStreamR = inReader.getUnitStream("r");
        std::shared_ptr<GPUIntStream> CORRECT_STREAM = outReader.getIntStream("y");

        // Run kernel
        inputStreamV->copy_to_device();
        inputStreamR->copy_to_device();
        std::shared_ptr<GPUIntStream> outputStream = last_thrust(inputStreamV, inputStreamR, 0);
        outputStream->copy_to_host();

        // Compare kernel result to correct data
        std::vector<int> kernelTimestamps(outputStream->host_timestamp+*(outputStream->host_offset), outputStream->host_timestamp+outputStream->size);
        std::vector<int> kernelValues(outputStream->host_values+*(outputStream->host_offset), outputStream->host_values+outputStream->size);
        std::vector<int> correctTimestamps(CORRECT_STREAM->host_timestamp, CORRECT_STREAM->host_timestamp + CORRECT_STREAM->size);
        std::vector<int> correctValues(CORRECT_STREAM->host_values, CORRECT_STREAM->host_values + CORRECT_STREAM->size);
        REQUIRE(kernelTimestamps == correctTimestamps);
        REQUIRE(kernelValues == correctValues);

        // Cleanup
        inputStreamV->free_device();
        inputStreamR->free_device();
        outputStream->free_device();
        inputStreamV->free_host();
        inputStreamR->free_host();
        outputStream->free_host();
    }

}

TEST_CASE("last()") {
    SECTION("last() small random example") {
        // Read input and correct output data
        GPUReader inReader = GPUReader(TESTDATA_PATH + "last_test1.in");
        GPUReader outReader = GPUReader(TESTDATA_PATH + "last_test1.out");
        std::shared_ptr<GPUIntStream> inputStreamV = inReader.getIntStream("z");
        std::shared_ptr<GPUUnitStream> inputStreamR = inReader.getUnitStream("a");
        std::shared_ptr<GPUIntStream> CORRECT_STREAM = outReader.getIntStream("y");

        // Run kernel
        inputStreamV->copy_to_device();
        inputStreamR->copy_to_device();
        std::shared_ptr<GPUIntStream> outputStream = last(inputStreamV, inputStreamR, 0);
        outputStream->copy_to_host();

        // Compare kernel result to correct data
        std::vector<int> kernelTimestamps(outputStream->host_timestamp + *(outputStream->host_offset),
                                          outputStream->host_timestamp + outputStream->size);
        std::vector<int> kernelValues(outputStream->host_values + *(outputStream->host_offset),
                                      outputStream->host_values + outputStream->size);
        std::vector<int> correctTimestamps(CORRECT_STREAM->host_timestamp,
                                           CORRECT_STREAM->host_timestamp + CORRECT_STREAM->size);
        std::vector<int> correctValues(CORRECT_STREAM->host_values, CORRECT_STREAM->host_values + CORRECT_STREAM->size);
        REQUIRE(kernelTimestamps == correctTimestamps);
        REQUIRE(kernelValues == correctValues);

        // Cleanup
        inputStreamV->free_device();
        inputStreamR->free_device();
        outputStream->free_device();
        inputStreamV->free_host();
        inputStreamR->free_host();
        outputStream->free_host();
    }

    SECTION("last empty") {
        //reading only invalid streams (they are empty)
        GPUReader inReader = GPUReader(TESTDATA_PATH + "bt_last.in");
        GPUReader outReader = GPUReader(TESTDATA_PATH + "bt_last.out");
        std::shared_ptr<GPUIntStream> inputStreamV = inReader.getIntStream("v");
        std::shared_ptr<GPUUnitStream> inputStreamR = inReader.getUnitStream("r");
        std::shared_ptr<GPUIntStream> CORRECT_STREAM = outReader.getIntStream("y");
        //int size = CORRECT_STREAM->size;
        //std::shared_ptr<GPUIntStream> outputStream;

        // Run kernel
        inputStreamV->copy_to_device();
        inputStreamR->copy_to_device();

        std::shared_ptr<GPUIntStream> outputStream = last(inputStreamV, inputStreamR, 0);
        outputStream->copy_to_host();

        // Compare kernel result to correct data
        std::vector<int> kernelTimestamps(outputStream->host_timestamp + *(outputStream->host_offset),
                                          outputStream->host_timestamp + outputStream->size);
        std::vector<int> kernelValues(outputStream->host_values + *(outputStream->host_offset),
                                      outputStream->host_values + outputStream->size);
        std::vector<int> correctTimestamps(CORRECT_STREAM->host_timestamp,
                                           CORRECT_STREAM->host_timestamp + CORRECT_STREAM->size);
        std::vector<int> correctValues(CORRECT_STREAM->host_values, CORRECT_STREAM->host_values + CORRECT_STREAM->size);
        REQUIRE(kernelTimestamps == correctTimestamps);
        REQUIRE(kernelValues == correctValues);

        // Cleanup
        inputStreamV->free_device();
        inputStreamR->free_device();
        outputStream->free_device();
        inputStreamV->free_host();
        inputStreamR->free_host();
        outputStream->free_host();
    }

    SECTION("last() tuwel example") {
        // Read input and correct output data
        GPUReader inReader = GPUReader(TESTDATA_PATH + "bt_last.in");
        GPUReader outReader = GPUReader(TESTDATA_PATH + "bt_last.out");
        std::shared_ptr<GPUIntStream> inputStreamV = inReader.getIntStream("v");
        std::shared_ptr<GPUUnitStream> inputStreamR = inReader.getUnitStream("r");
        std::shared_ptr<GPUIntStream> CORRECT_STREAM = outReader.getIntStream("y");

        // Prepare empty output stream to fill

        // Run kernel
        inputStreamV->copy_to_device();
        inputStreamR->copy_to_device();

        std::shared_ptr<GPUIntStream> outputStream = last(inputStreamV, inputStreamR, 0);
        outputStream->copy_to_host();

        // Compare kernel result to correct data
        std::vector<int> kernelTimestamps(outputStream->host_timestamp + *(outputStream->host_offset),
                                          outputStream->host_timestamp + outputStream->size);
        std::vector<int> kernelValues(outputStream->host_values + *(outputStream->host_offset),
                                      outputStream->host_values + outputStream->size);
        std::vector<int> correctTimestamps(CORRECT_STREAM->host_timestamp,
                                           CORRECT_STREAM->host_timestamp + CORRECT_STREAM->size);
        std::vector<int> correctValues(CORRECT_STREAM->host_values, CORRECT_STREAM->host_values + CORRECT_STREAM->size);
        REQUIRE(kernelTimestamps == correctTimestamps);
        REQUIRE(kernelValues == correctValues);

        // Cleanup
        inputStreamV->free_device();
        inputStreamR->free_device();
        outputStream->free_device();
        inputStreamV->free_host();
        inputStreamR->free_host();
        outputStream->free_host();
    }

    SECTION("last() small random example") {
        // Read input and correct output data
        GPUReader inReader = GPUReader(TESTDATA_PATH + "last_test1.in");
        GPUReader outReader = GPUReader(TESTDATA_PATH + "last_test1.out");
        std::shared_ptr<GPUIntStream> inputStreamV = inReader.getIntStream("z");
        std::shared_ptr<GPUUnitStream> inputStreamR = inReader.getUnitStream("a");
        std::shared_ptr<GPUIntStream> CORRECT_STREAM = outReader.getIntStream("y");

        // Run kernel
        inputStreamV->copy_to_device();
        inputStreamR->copy_to_device();

        std::shared_ptr<GPUIntStream> outputStream = last(inputStreamV, inputStreamR, 0);
        outputStream->copy_to_host();

        std::vector<int> kernelTimestamps(outputStream->host_timestamp + *(outputStream->host_offset),
                                          outputStream->host_timestamp + outputStream->size);
        std::vector<int> kernelValues(outputStream->host_values + *(outputStream->host_offset),
                                      outputStream->host_values + outputStream->size);
        std::vector<int> correctTimestamps(CORRECT_STREAM->host_timestamp,
                                           CORRECT_STREAM->host_timestamp + CORRECT_STREAM->size);
        std::vector<int> correctValues(CORRECT_STREAM->host_values, CORRECT_STREAM->host_values + CORRECT_STREAM->size);
        REQUIRE(kernelTimestamps == correctTimestamps);
        REQUIRE(kernelValues == correctValues);

        // Cleanup
        inputStreamV->free_device();
        inputStreamR->free_device();
        outputStream->free_device();
        inputStreamV->free_host();
        inputStreamR->free_host();
        outputStream->free_host();
    }

    SECTION("last() large random example") {
        // Read input and correct output data
        GPUReader inReader = GPUReader(TESTDATA_PATH + "last_test2.in");
        GPUReader outReader = GPUReader(TESTDATA_PATH + "last_test2.out");
        std::shared_ptr<GPUIntStream> inputStreamV = inReader.getIntStream("z");
        std::shared_ptr<GPUUnitStream> inputStreamR = inReader.getUnitStream("a");
        std::shared_ptr<GPUIntStream> CORRECT_STREAM = outReader.getIntStream("y");
        //std::shared_ptr<GPUIntStream> outputStream;

        // Run kernel
        inputStreamV->copy_to_device();
        inputStreamR->copy_to_device();

        std::shared_ptr<GPUIntStream> outputStream = last(inputStreamV, inputStreamR, 0);
        outputStream->copy_to_host();

        // Compare kernel result to correct data
        std::vector<int> kernelTimestamps(outputStream->host_timestamp + *(outputStream->host_offset),
                                          outputStream->host_timestamp + outputStream->size);
        std::vector<int> kernelValues(outputStream->host_values + *(outputStream->host_offset),
                                      outputStream->host_values + outputStream->size);
        std::vector<int> correctTimestamps(CORRECT_STREAM->host_timestamp,
                                           CORRECT_STREAM->host_timestamp + CORRECT_STREAM->size);
        std::vector<int> correctValues(CORRECT_STREAM->host_values, CORRECT_STREAM->host_values + CORRECT_STREAM->size);
        REQUIRE(kernelTimestamps == correctTimestamps);
        REQUIRE(kernelValues == correctValues);

        // Cleanup
        inputStreamV->free_device();
        inputStreamR->free_device();
        outputStream->free_device();
        inputStreamV->free_host();
        inputStreamR->free_host();
        outputStream->free_host();
    }


    SECTION("last() twice test with no invalids") {
        // Read input and correct output data
        GPUReader inReader = GPUReader(TESTDATA_PATH + "last_test3.in");
        GPUReader outReader = GPUReader(TESTDATA_PATH + "last_test3.out");
        std::shared_ptr<GPUIntStream> inputStreamV = inReader.getIntStream("z");
        std::shared_ptr<GPUUnitStream> inputStream2 = inReader.getUnitStream("a");
        std::shared_ptr<GPUUnitStream> inputStreamR = inReader.getUnitStream("x");
        std::shared_ptr<GPUIntStream> CORRECT_STREAM = outReader.getIntStream("o");

        // Run kernel
        inputStreamV->copy_to_device();
        inputStreamR->copy_to_device();
        inputStream2->copy_to_device();
        std::shared_ptr<GPUIntStream> intermediateStream = last(inputStreamV, inputStreamR, 0);
        std::shared_ptr<GPUIntStream> outputStream = last(intermediateStream, inputStream2, 0);
        intermediateStream->copy_to_host();
        outputStream->copy_to_host();

        // Compare kernel result to correct data
        std::vector<int> kernelTimestamps(outputStream->host_timestamp + *(outputStream->host_offset),
                                          outputStream->host_timestamp + outputStream->size);
        std::vector<int> kernelValues(outputStream->host_values + *(outputStream->host_offset),
                                      outputStream->host_values + outputStream->size);
        std::vector<int> correctTimestamps(CORRECT_STREAM->host_timestamp,
                                           CORRECT_STREAM->host_timestamp + CORRECT_STREAM->size);
        std::vector<int> correctValues(CORRECT_STREAM->host_values, CORRECT_STREAM->host_values + CORRECT_STREAM->size);
        REQUIRE(kernelTimestamps == correctTimestamps);
        REQUIRE(kernelValues == correctValues);

        // Cleanup
        inputStreamV->free_device();
        inputStreamR->free_device();
        intermediateStream->free_device();
        outputStream->free_device();
        inputStream2->free_device();
        inputStreamV->free_host();
        inputStreamR->free_host();
        intermediateStream->free_host();
        outputStream->free_host();
        inputStream2->free_host();
    }

    SECTION("last() twice test with invalids in Unit Stream") {
        GPUReader inReader = GPUReader(TESTDATA_PATH + "last_test4.in");
        GPUReader outReader = GPUReader(TESTDATA_PATH + "last_test4.out");
        std::shared_ptr<GPUIntStream> inputStreamV = inReader.getIntStream("z");
        std::shared_ptr<GPUUnitStream> inputStream2 = inReader.getUnitStream("x");
        std::shared_ptr<GPUUnitStream> inputStreamR = inReader.getUnitStream("a");
        std::shared_ptr<GPUIntStream> CORRECT_STREAM = outReader.getIntStream("o");

        inputStreamV->copy_to_device();
        inputStreamR->copy_to_device();
        inputStream2->copy_to_device();

        std::shared_ptr<GPUIntStream> intermediateStream = last(inputStreamV, inputStreamR, 0);
        std::shared_ptr<GPUIntStream> outputStream = last(intermediateStream, inputStream2, 0);
        outputStream->copy_to_host();

        // Compare kernel result to correct data
        std::vector<int> kernelTimestamps(outputStream->host_timestamp + *(outputStream->host_offset),
                                          outputStream->host_timestamp + outputStream->size);
        std::vector<int> kernelValues(outputStream->host_values + *(outputStream->host_offset),
                                      outputStream->host_values + outputStream->size);
        std::vector<int> correctTimestamps(CORRECT_STREAM->host_timestamp,
                                           CORRECT_STREAM->host_timestamp + CORRECT_STREAM->size);
        std::vector<int> correctValues(CORRECT_STREAM->host_values, CORRECT_STREAM->host_values + CORRECT_STREAM->size);
        REQUIRE(kernelTimestamps == correctTimestamps);
        REQUIRE(kernelValues == correctValues);

        // Cleanup
        inputStreamV->free_device();
        inputStreamR->free_device();
        intermediateStream->free_device();
        outputStream->free_device();
        inputStream2->free_device();
        inputStreamV->free_host();
        inputStreamR->free_host();
        intermediateStream->free_host();
        outputStream->free_host();
        inputStream2->free_host();
    }
}