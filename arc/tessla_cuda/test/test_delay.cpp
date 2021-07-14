#include <iostream>
#include <string>
#include <string.h>
#include <vector>
#include "../../test/catch2/catch.hpp"
#include <GPUReader.cuh>
#include <GPUStream.cuh>
#include <StreamFunctions.cuh>

#include <chrono>
#include <fstream>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <cuda_profiler_api.h>

#include <StreamFunctions.cuh>
#include <StreamFunctionsThrust.cuh>

static std::string TESTDATA_PATH = "test/data/";

TEST_CASE("delay_thrust()") {
    SECTION("delay_thrust() tuwel example") {
        GPUReader inReader = GPUReader(TESTDATA_PATH + "bt_delay.in");
        GPUReader outReader = GPUReader(TESTDATA_PATH + "bt_delay.out");
        std::shared_ptr<GPUIntStream> inputStreamD = inReader.getIntStream("d");
        std::shared_ptr<GPUUnitStream> inputStreamR = inReader.getUnitStream("r");
        std::shared_ptr<GPUUnitStream> CORRECT_STREAM = outReader.getUnitStream("y");

        inputStreamD->copy_to_device();
        inputStreamR->copy_to_device();

        std::shared_ptr<GPUUnitStream> outputStream = delay_thrust(inputStreamD, inputStreamR, 0);
        outputStream->copy_to_host();

        std::vector<int> kernelTimestamps(outputStream->host_timestamp + *(outputStream->host_offset),
                                          outputStream->host_timestamp + outputStream->size);
        std::vector<int> correctTimestamps(CORRECT_STREAM->host_timestamp,
                                           CORRECT_STREAM->host_timestamp + CORRECT_STREAM->size);
        REQUIRE(kernelTimestamps == correctTimestamps);

        // Cleanup
        inputStreamD->free_device();
        inputStreamR->free_device();
        inputStreamD->free_host();
        inputStreamR->free_host();
        outputStream->free_host();
    }

    SECTION("delay_thrust() middle") {
        GPUReader inReader = GPUReader(TESTDATA_PATH + "bt_delay.middle.in");
        GPUReader outReader = GPUReader(TESTDATA_PATH + "bt_delay.middle.out");
        std::shared_ptr<GPUIntStream> inputStreamZ = inReader.getIntStream("z");
        std::shared_ptr<GPUUnitStream> inputStreamA = inReader.getUnitStream("a");
        std::shared_ptr<GPUUnitStream> CORRECT_STREAM = outReader.getUnitStream("y");

        inputStreamZ->copy_to_device();
        inputStreamA->copy_to_device();

        std::shared_ptr<GPUUnitStream> outputStream = delay_thrust(inputStreamZ, inputStreamA, 0);
        outputStream->copy_to_host();

        std::vector<int> kernelTimestamps(outputStream->host_timestamp + *(outputStream->host_offset),
                                          outputStream->host_timestamp + outputStream->size);
        std::vector<int> correctTimestamps(CORRECT_STREAM->host_timestamp,
                                           CORRECT_STREAM->host_timestamp + CORRECT_STREAM->size);
        REQUIRE(kernelTimestamps == correctTimestamps);

        // Cleanup
        inputStreamZ->free_device();
        inputStreamA->free_device();
        inputStreamZ->free_host();
        inputStreamA->free_host();
        outputStream->free_host();
    }

    SECTION("delay_thrust() bigger") {
        GPUReader inReader = GPUReader(TESTDATA_PATH + "bt_delay.bigger.in");
        GPUReader outReader = GPUReader(TESTDATA_PATH + "bt_delay.bigger.out");
        std::shared_ptr<GPUIntStream> inputStreamZ = inReader.getIntStream("z");
        std::shared_ptr<GPUUnitStream> inputStreamA = inReader.getUnitStream("a");
        std::shared_ptr<GPUUnitStream> CORRECT_STREAM = outReader.getUnitStream("y");

        inputStreamZ->copy_to_device();
        inputStreamA->copy_to_device();

        std::shared_ptr<GPUUnitStream> outputStream = delay_thrust(inputStreamZ, inputStreamA, 0);
        outputStream->copy_to_host();

        std::vector<int> kernelTimestamps(outputStream->host_timestamp + *(outputStream->host_offset),
                                          outputStream->host_timestamp + outputStream->size);
        std::vector<int> correctTimestamps(CORRECT_STREAM->host_timestamp,
                                           CORRECT_STREAM->host_timestamp + CORRECT_STREAM->size);
        REQUIRE(kernelTimestamps == correctTimestamps);

        // Cleanup
        inputStreamZ->free_device();
        inputStreamA->free_device();
        inputStreamZ->free_host();
        inputStreamA->free_host();
        outputStream->free_host();
    }
}