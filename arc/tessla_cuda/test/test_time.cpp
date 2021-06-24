//
// Created by fabian on 24/06/2021.
//

#include <iostream>
#include <string>
#include <string.h>
#include <vector>
#include "../../test/catch2/catch.hpp"
#include <GPUReader.cuh>
#include <Stream.cuh>
#include <StreamFunctions.cuh>

#include <chrono>
#include <fstream>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <cuda_profiler_api.h>

#include <StreamFunctionsThrust.cuh>


TEST_CASE("time()") {
    SECTION("time() with small dataset") {
// Read input and correct output data
        GPUReader inReader = GPUReader("../test/data/bt_time.in");
        std::shared_ptr<GPUIntStream> inputStream = inReader.getIntStreamDebug("x");
        GPUReader outReader = GPUReader("../test/data/bt_time.out");
        std::shared_ptr<GPUIntStream> CORRECT_STREAM = outReader.getIntStreamDebug("x");

// Prepare empty output stream to fill
        int size = CORRECT_STREAM->size;

// outputStream; //(host_timestampOut, host_valueOut, size);

// Run kernel
        inputStream->copy_to_device();
//outputStream.copy_to_device();
        std::shared_ptr<GPUIntStream> outputStream = time(inputStream, 0);
        printf("Result size outputstream %d \n", outputStream->size);
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
    }

    SECTION("time() with bigger dataset (~109k/250k events)") {
// Read input and correct output data
        GPUReader inReader = GPUReader("../test/data/bt_time.bigger.in");
        std::shared_ptr<GPUIntStream> inputStream = inReader.getIntStreamDebug("z");
        GPUReader outReader = GPUReader("../test/data/bt_time.bigger.out");
        std::shared_ptr<GPUIntStream> CORRECT_STREAM = outReader.getIntStreamDebug("y");

// Prepare empty output stream to fill
        int size = CORRECT_STREAM->size;

//std::shared_ptr<GPUIntStream> outputStream;
// Run kernel
        inputStream->copy_to_device();
//outputStream->copy_to_device();
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
    }
}