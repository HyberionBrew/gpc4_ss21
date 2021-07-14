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

TEST_CASE("time()") {
    SECTION("time() with small dataset") {
        GPUReader inReader = GPUReader(TESTDATA_PATH + "bt_time.in");
        GPUReader outReader = GPUReader(TESTDATA_PATH + "bt_time.out");
        std::shared_ptr<GPUIntStream> inputStream = inReader.getIntStreamDebug("x");
        std::shared_ptr<GPUIntStream> CORRECT_STREAM = outReader.getIntStreamDebug("x");
        inputStream->copy_to_device();

        std::shared_ptr<GPUIntStream> outputStream = time(inputStream, 0);
        outputStream->copy_to_host();

        // Compare kernel result to correct data
        std::vector<int> kernelTimestamps(outputStream->host_timestamp, outputStream->host_timestamp +
                                                                        sizeof(outputStream->host_timestamp) /
                                                                        sizeof(int));
        std::vector<int> kernelValues(outputStream->host_values,
                                      outputStream->host_values + sizeof(outputStream->host_values) / sizeof(int));
        std::vector<int> correctTimestamps(CORRECT_STREAM->host_timestamp, CORRECT_STREAM->host_timestamp +
                                                                           sizeof(CORRECT_STREAM->host_timestamp) /
                                                                           sizeof(int));
        std::vector<int> correctValues(CORRECT_STREAM->host_values, CORRECT_STREAM->host_values +
                                                                    sizeof(CORRECT_STREAM->host_values) /
                                                                    sizeof(int));
        REQUIRE(kernelTimestamps == correctTimestamps);
        REQUIRE(kernelValues == correctValues);

        // Cleanup
        inputStream->free_device();
        outputStream->free_device();
        inputStream->free_host();
        outputStream->free_host();
    }

    SECTION("time() with bigger dataset (~109k/250k events)") {
        GPUReader inReader = GPUReader(TESTDATA_PATH + "bt_time.bigger.in");
        GPUReader outReader = GPUReader(TESTDATA_PATH + "bt_time.bigger.out");
        std::shared_ptr<GPUIntStream> inputStream = inReader.getIntStreamDebug("z");
        std::shared_ptr<GPUIntStream> CORRECT_STREAM = outReader.getIntStreamDebug("y");
        inputStream->copy_to_device();

        std::shared_ptr<GPUIntStream> outputStream = time(inputStream, 0);
        outputStream->copy_to_host();

        // Compare kernel result to correct data
        std::vector<int> kernelTimestamps(outputStream->host_timestamp, outputStream->host_timestamp +
                                                                        sizeof(outputStream->host_timestamp) /
                                                                        sizeof(int));
        std::vector<int> kernelValues(outputStream->host_values,
                                      outputStream->host_values + sizeof(outputStream->host_values) / sizeof(int));
        std::vector<int> correctTimestamps(CORRECT_STREAM->host_timestamp, CORRECT_STREAM->host_timestamp +
                                                                           sizeof(CORRECT_STREAM->host_timestamp) /
                                                                           sizeof(int));
        std::vector<int> correctValues(CORRECT_STREAM->host_values, CORRECT_STREAM->host_values +
                                                                    sizeof(CORRECT_STREAM->host_values) /
                                                                    sizeof(int));
        REQUIRE(kernelTimestamps == correctTimestamps);
        REQUIRE(kernelValues == correctValues);

        // Cleanup
        inputStream->free_device();
        outputStream->free_device();
        inputStream->free_host();
        outputStream->free_host();
    }
}

TEST_CASE("time_thrust()") {
    SECTION("time_thrust() with small dataset") {
        GPUReader inReader = GPUReader(TESTDATA_PATH + "bt_time.in");
        GPUReader outReader = GPUReader(TESTDATA_PATH + "bt_time.out");
        std::shared_ptr<GPUIntStream> inputStream = inReader.getIntStreamDebug("x");
        std::shared_ptr<GPUIntStream> CORRECT_STREAM = outReader.getIntStreamDebug("x");
        inputStream->copy_to_device();

        std::shared_ptr<GPUIntStream> outputStream = time_thrust(inputStream);
        outputStream->copy_to_host();

        // Compare kernel result to correct data
        std::vector<int> kernelTimestamps(outputStream->host_timestamp, outputStream->host_timestamp +
                                                                        sizeof(outputStream->host_timestamp) /
                                                                        sizeof(int));
        std::vector<int> kernelValues(outputStream->host_values,
                                      outputStream->host_values + sizeof(outputStream->host_values) / sizeof(int));
        std::vector<int> correctTimestamps(CORRECT_STREAM->host_timestamp, CORRECT_STREAM->host_timestamp +
                                                                           sizeof(CORRECT_STREAM->host_timestamp) /
                                                                           sizeof(int));
        std::vector<int> correctValues(CORRECT_STREAM->host_values, CORRECT_STREAM->host_values +
                                                                    sizeof(CORRECT_STREAM->host_values) /
                                                                    sizeof(int));
        REQUIRE(kernelTimestamps == correctTimestamps);
        REQUIRE(kernelValues == correctValues);

        // Cleanup
        inputStream->free_device();
        outputStream->free_device();
        inputStream->free_host();
        outputStream->free_host();
    }

    SECTION("time_thrust() with bigger dataset (~109k/250k events)") {
        GPUReader inReader = GPUReader(TESTDATA_PATH + "bt_time.bigger.in");
        GPUReader outReader = GPUReader(TESTDATA_PATH + "bt_time.bigger.out");
        std::shared_ptr<GPUIntStream> inputStream = inReader.getIntStreamDebug("z");
        std::shared_ptr<GPUIntStream> CORRECT_STREAM = outReader.getIntStreamDebug("y");
        inputStream->copy_to_device();

        std::shared_ptr<GPUIntStream> outputStream = time_thrust(inputStream);
        outputStream->copy_to_host();

        // Compare kernel result to correct data
        std::vector<int> kernelTimestamps(outputStream->host_timestamp, outputStream->host_timestamp +
                                                                        sizeof(outputStream->host_timestamp) /
                                                                        sizeof(int));
        std::vector<int> kernelValues(outputStream->host_values,
                                      outputStream->host_values + sizeof(outputStream->host_values) / sizeof(int));
        std::vector<int> correctTimestamps(CORRECT_STREAM->host_timestamp, CORRECT_STREAM->host_timestamp +
                                                                           sizeof(CORRECT_STREAM->host_timestamp) /
                                                                           sizeof(int));
        std::vector<int> correctValues(CORRECT_STREAM->host_values, CORRECT_STREAM->host_values +
                                                                    sizeof(CORRECT_STREAM->host_values) /
                                                                    sizeof(int));
        REQUIRE(kernelTimestamps == correctTimestamps);
        REQUIRE(kernelValues == correctValues);

        // Cleanup
        inputStream->free_device();
        outputStream->free_device();
        inputStream->free_host();
        outputStream->free_host();
    }
}