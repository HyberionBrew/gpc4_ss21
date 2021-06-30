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

TEST_CASE("last_thrust()") {
    SECTION("last_thrust() tuwel example") {
        // Read input and correct output data

        GPUReader inReader = GPUReader("test/data/bt_last.in");
        std::shared_ptr<GPUIntStream> inputStreamV = inReader.getIntStreamDebug("v");
        std::shared_ptr<GPUUnitStream> inputStreamR = inReader.getUnitStreamDebug("r");
        GPUReader outReader = GPUReader("test/data/bt_last.out");
        std::shared_ptr<GPUIntStream> CORRECT_STREAM = outReader.getIntStreamDebug("y");

        // Prepare empty output stream to fill
        int size = CORRECT_STREAM->size;
        std::shared_ptr<GPUIntStream> outputStream;
        // Run kernel
        inputStreamV->copy_to_device();
        inputStreamR->copy_to_device();
        // inputStreamV->print();
        //  inputStreamR->print();
        outputStream = last_thrust(inputStreamV, inputStreamR, 0);
        //inputStreamR->print();
        outputStream->copy_to_host();
        //outputStream->print();
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
        //outputStream->print();
        // Cleanup
        inputStreamV->free_device();

        inputStreamR->free_device();
        outputStream->free_device();

        inputStreamV->free_host();
        outputStream->free_host();
        inputStreamR->free_host();
    }

    SECTION("last() large random example") {
        // Read input and correct output data
        GPUReader inReader = GPUReader("test/data/last_test2.in");
        std::shared_ptr<GPUIntStream> inputStreamV = inReader.getIntStreamDebug("z");
        std::shared_ptr<GPUUnitStream> inputStreamR = inReader.getUnitStreamDebug("a");
        GPUReader outReader = GPUReader("test/data/last_test2.out");
        std::shared_ptr<GPUIntStream> CORRECT_STREAM = outReader.getIntStreamDebug("y");
        // Prepare empty output stream to fill
        std::shared_ptr<GPUIntStream> outputStream;

        // Run kernel
        inputStreamV->copy_to_device();
        inputStreamR->copy_to_device();
        //inputStreamV->print();
        //inputStreamR->print();

        outputStream = last_thrust(inputStreamV, inputStreamR, 0);
        outputStream->copy_to_host();

        //outputStream->print();

        // Compare kernel result to correct data
        std::vector<int> kernelTimestamps(outputStream->host_timestamp+*(outputStream->host_offset), outputStream->host_timestamp+outputStream->size);
        std::vector<int> kernelValues(outputStream->host_values+*(outputStream->host_offset), outputStream->host_values+outputStream->size);
        std::vector<int> correctTimestamps(CORRECT_STREAM->host_timestamp, CORRECT_STREAM->host_timestamp + CORRECT_STREAM->size);
        std::vector<int> correctValues(CORRECT_STREAM->host_values, CORRECT_STREAM->host_values + CORRECT_STREAM->size);
        for (int i = 0; i< CORRECT_STREAM->size; i++){
            REQUIRE(kernelTimestamps[i] == correctTimestamps[i]);
        }

        REQUIRE(kernelValues == correctValues);
        //outputStream->print();
        // Cleanup
        inputStreamV->free_device();
        inputStreamR->free_device();
        outputStream->free_device();
    }

    SECTION("last() twice test with no invalids") {
        // Read input and correct output data
        GPUReader inReader = GPUReader("test/data/last_test3.in");
        std::shared_ptr<GPUIntStream> inputStreamV = inReader.getIntStreamDebug("z");
        std::shared_ptr<GPUUnitStream> inputStream2 = inReader.getUnitStreamDebug("a");
        std::shared_ptr<GPUUnitStream> inputStreamR = inReader.getUnitStreamDebug("x");
        GPUReader outReader = GPUReader("test/data/last_test3.out");
        std::shared_ptr<GPUIntStream> CORRECT_STREAM = outReader.getIntStreamDebug("o");

        // Prepare empty output stream to fill
        int size = CORRECT_STREAM->size;

        std::shared_ptr<GPUIntStream> intermediateStream;
        std::shared_ptr<GPUIntStream> outputStream;

        // Run kernel
        inputStreamV->copy_to_device();
        inputStreamR->copy_to_device();
        inputStream2->copy_to_device();
        intermediateStream = last_thrust(inputStreamV, inputStreamR, 0);
        intermediateStream->copy_to_host();
        outputStream = last_thrust(intermediateStream, inputStream2, 0);

        outputStream->copy_to_host();
//        outputStream.print();

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
    }

    SECTION("last() twice test with invalids in Unit Stream") {
        //printf("-------------------------\n");
        // Read input and correct output data
        GPUReader inReader = GPUReader("test/data/last_test4.in");
        std::shared_ptr<GPUIntStream> inputStreamV = inReader.getIntStreamDebug("z");
        std::shared_ptr<GPUUnitStream> inputStream2 = inReader.getUnitStreamDebug("x");
        std::shared_ptr<GPUUnitStream> inputStreamR = inReader.getUnitStreamDebug("a");
        GPUReader outReader = GPUReader("test/data/last_test4.out");
        std::shared_ptr<GPUIntStream> CORRECT_STREAM = outReader.getIntStreamDebug("o");

        // Prepare empty output stream to fill
        int size = CORRECT_STREAM->size;
        int sizeAllocated = (size_t)  inputStreamR->size * sizeof(int);
        int *host_timestampOut = (int *) malloc( inputStreamR->size * sizeof(int));
        int *host_valueOut = (int*) malloc( inputStreamR->size * sizeof(int));

        memset(host_timestampOut, 0, sizeAllocated);
        memset(host_valueOut, 0, sizeAllocated);

        int *host_timestampOut2 = (int *) malloc( inputStream2->size * sizeof(int));
        int *host_valueOut2 = (int*) malloc( inputStream2->size * sizeof(int));

        memset(host_timestampOut2, 0, inputStream2->size * sizeof(int));
        memset(host_valueOut2, 0, inputStream2->size * sizeof(int));

        std::shared_ptr<GPUIntStream> intermediateStream = std::make_shared<GPUIntStream>(host_timestampOut,
                                                                                          host_valueOut,
                                                                                          inputStreamR->size);
        std::shared_ptr<GPUIntStream> outputStream = std::make_shared<GPUIntStream>(host_timestampOut2, host_valueOut2,
                                                                                    inputStream2->size);

        // Run kernel
        intermediateStream->copy_to_device();
        inputStreamV->copy_to_device();
        inputStreamR->copy_to_device();
        outputStream->copy_to_device();
        inputStream2->copy_to_device();

        intermediateStream = last_thrust(inputStreamV, inputStreamR,  0);
        outputStream = last_thrust(intermediateStream, inputStream2, 0);

        outputStream->copy_to_host();
        //outputStream->print();

        // Compare kernel result to correct data
        std::vector<int> kernelTimestamps(outputStream->host_timestamp+*(outputStream->host_offset), outputStream->host_timestamp+outputStream->size);
        std::vector<int> kernelValues(outputStream->host_values+*(outputStream->host_offset), outputStream->host_values+outputStream->size);
        std::vector<int> correctTimestamps(CORRECT_STREAM->host_timestamp, CORRECT_STREAM->host_timestamp + CORRECT_STREAM->size);
        std::vector<int> correctValues(CORRECT_STREAM->host_values, CORRECT_STREAM->host_values + CORRECT_STREAM->size);

        REQUIRE(kernelTimestamps == correctTimestamps);

        REQUIRE(kernelValues == correctValues);
        //outputStream->print();
        // Cleanup
        inputStreamV->free_device();
        inputStreamR->free_device();
        intermediateStream->free_device();
        outputStream->free_device();
        inputStream2->free_device();
        free(host_valueOut);
        free(host_timestampOut);
        free(host_valueOut2);
        free(host_timestampOut2);
    }
    SECTION("last empty"){
        //reading only invalid streams (they are empty)
        GPUReader inReader = GPUReader("test/data/bt_last.in");
        std::shared_ptr<GPUIntStream> inputStreamV = inReader.getIntStreamDebug("v2");
        std::shared_ptr<GPUUnitStream> inputStreamR = inReader.getUnitStreamDebug("r2");
        GPUReader outReader = GPUReader("test/data/bt_last.out");
        std::shared_ptr<GPUIntStream> CORRECT_STREAM = outReader.getIntStreamDebug("y2");
        int size = CORRECT_STREAM->size;
        std::shared_ptr<GPUIntStream> outputStream;

        // Run kernel
        inputStreamV->copy_to_device();
        inputStreamR->copy_to_device();
        // inputStreamV->print();
        //  inputStreamR->print();
        outputStream = last_thrust(inputStreamV, inputStreamR, 0);
        //inputStreamR->print();
        outputStream->copy_to_host();

        //outputStream->print();
        // Compare kernel result to correct data
        std::vector<int> kernelTimestamps(outputStream->host_timestamp+*(outputStream->host_offset), outputStream->host_timestamp+outputStream->size);
        std::vector<int> kernelValues(outputStream->host_values+*(outputStream->host_offset), outputStream->host_values+outputStream->size);
        std::vector<int> correctTimestamps(CORRECT_STREAM->host_timestamp, CORRECT_STREAM->host_timestamp + CORRECT_STREAM->size);
        std::vector<int> correctValues(CORRECT_STREAM->host_values, CORRECT_STREAM->host_values + CORRECT_STREAM->size);

        REQUIRE(kernelTimestamps == correctTimestamps);
        REQUIRE(kernelValues == correctValues);
        //outputStream->print();
        // Cleanup
        inputStreamV->free_device();
        inputStreamR->free_device();
        outputStream->free_device();
    }

}

TEST_CASE("last()") {
    SECTION("last() small random example") {
        // Read input and correct output data
        GPUReader inReader = GPUReader("test/data/last_test1.in");
        std::shared_ptr<GPUIntStream> inputStreamV = inReader.getIntStreamDebug("z");
        std::shared_ptr<GPUUnitStream> inputStreamR = inReader.getUnitStreamDebug("a");
        GPUReader outReader = GPUReader("test/data/last_test1.out");
        std::shared_ptr<GPUIntStream> CORRECT_STREAM = outReader.getIntStreamDebug("y");

        // Prepare empty output stream to fill
        int size = CORRECT_STREAM->size;
        //printf("%d \n\n",  CORRECT_STREAM.size);

        // Run kernel
        inputStreamV->copy_to_device();
        inputStreamR->copy_to_device();
        //inputStreamV->print();
        //inputStreamR->print();
        //inputStreamV.print();
        //inputStreamR.print();
        std::shared_ptr<GPUIntStream> outputStream = last(inputStreamV, inputStreamR, 0);
        outputStream->copy_to_host();
        //outputStream->print();
        //printf("xx");
        // outputStream.print();
        //outputStream.print();
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
    }

    SECTION("last empty") {

        //reading only invalid streams (they are empty)
        GPUReader inReader = GPUReader("test/data/bt_last.in");
        std::shared_ptr<GPUIntStream> inputStreamV = inReader.getIntStreamDebug("v2");
        std::shared_ptr<GPUUnitStream> inputStreamR = inReader.getUnitStreamDebug("r2");
        GPUReader outReader = GPUReader("test/data/bt_last.out");
        std::shared_ptr<GPUIntStream> CORRECT_STREAM = outReader.getIntStreamDebug("y2");
        int size = CORRECT_STREAM->size;
        std::shared_ptr<GPUIntStream> outputStream;

        // Run kernel
        inputStreamV->copy_to_device();
        inputStreamR->copy_to_device();
        // inputStreamV->print();
        //  inputStreamR->print();
        outputStream = last(inputStreamV, inputStreamR, 0);
        //inputStreamR->print();
        outputStream->copy_to_host();

        //outputStream->print();
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
        //outputStream->print();
        // Cleanup
        inputStreamV->free_device();
        inputStreamR->free_device();
        outputStream->free_device();
    }

    SECTION("last() tuwel example") {
        // Read input and correct output data

        GPUReader inReader = GPUReader("test/data/bt_last.in");
        std::shared_ptr<GPUIntStream> inputStreamV = inReader.getIntStreamDebug("v");
        std::shared_ptr<GPUUnitStream> inputStreamR = inReader.getUnitStreamDebug("r");
        GPUReader outReader = GPUReader("test/data/bt_last.out");
        std::shared_ptr<GPUIntStream> CORRECT_STREAM = outReader.getIntStreamDebug("y");

        // Prepare empty output stream to fill
        int size = CORRECT_STREAM->size;
        std::shared_ptr<GPUIntStream> outputStream;

        // Run kernel
        inputStreamV->copy_to_device();
        inputStreamR->copy_to_device();
        // inputStreamV->print();
        //  inputStreamR->print();
        outputStream = last(inputStreamV, inputStreamR, 0);
        //inputStreamR->print();
        outputStream->copy_to_host();

        //outputStream->print();
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
        //outputStream->print();
        // Cleanup
        inputStreamV->free_device();
        inputStreamR->free_device();
        outputStream->free_device();
    }

    SECTION("last() small random example") {
        // Read input and correct output data
        GPUReader inReader = GPUReader("test/data/last_test1.in");
        std::shared_ptr<GPUIntStream> inputStreamV = inReader.getIntStreamDebug("z");
        std::shared_ptr<GPUUnitStream> inputStreamR = inReader.getUnitStreamDebug("a");
        GPUReader outReader = GPUReader("test/data/last_test1.out");
        std::shared_ptr<GPUIntStream> CORRECT_STREAM = outReader.getIntStreamDebug("y");

        // Prepare empty output stream to fill
        int size = CORRECT_STREAM->size;
        //printf("%d \n\n",  CORRECT_STREAM->size);

        std::shared_ptr<GPUIntStream> outputStream;

        // Run kernel
        inputStreamV->copy_to_device();
        inputStreamR->copy_to_device();
        //inputStreamV->print();
        //inputStreamR->print();
        outputStream = last(inputStreamV, inputStreamR, 0);
        outputStream->copy_to_host();
        //printf("xx");
        // outputStream->print();
        //outputStream->print();
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
    }

    SECTION("last() large random example") {
        // Read input and correct output data
        GPUReader inReader = GPUReader("test/data/last_test2.in");
        std::shared_ptr<GPUIntStream> inputStreamV = inReader.getIntStreamDebug("z");
        std::shared_ptr<GPUUnitStream> inputStreamR = inReader.getUnitStreamDebug("a");
        GPUReader outReader = GPUReader("test/data/last_test2.out");
        std::shared_ptr<GPUIntStream> CORRECT_STREAM = outReader.getIntStreamDebug("y");
        // Prepare empty output stream to fill
        std::shared_ptr<GPUIntStream> outputStream;

        // Run kernel
        inputStreamV->copy_to_device();
        inputStreamR->copy_to_device();
        //inputStreamV->print();
        //inputStreamR->print();

        outputStream = last(inputStreamV, inputStreamR, 0);
        outputStream->copy_to_host();

        //outputStream->print();

        // Compare kernel result to correct data
        std::vector<int> kernelTimestamps(outputStream->host_timestamp + *(outputStream->host_offset),
                                          outputStream->host_timestamp + outputStream->size);
        std::vector<int> kernelValues(outputStream->host_values + *(outputStream->host_offset),
                                      outputStream->host_values + outputStream->size);
        std::vector<int> correctTimestamps(CORRECT_STREAM->host_timestamp,
                                           CORRECT_STREAM->host_timestamp + CORRECT_STREAM->size);
        std::vector<int> correctValues(CORRECT_STREAM->host_values, CORRECT_STREAM->host_values + CORRECT_STREAM->size);
        for (int i = 0; i < CORRECT_STREAM->size; i++) {
            REQUIRE(kernelTimestamps[i] == correctTimestamps[i]);
        }

        REQUIRE(kernelValues == correctValues);
        //outputStream->print();
        // Cleanup
        inputStreamV->free_device();
        inputStreamR->free_device();
        outputStream->free_device();
    }


    SECTION("last() twice test with no invalids") {
        // Read input and correct output data
        GPUReader inReader = GPUReader("test/data/last_test3.in");
        std::shared_ptr<GPUIntStream> inputStreamV = inReader.getIntStreamDebug("z");
        std::shared_ptr<GPUUnitStream> inputStream2 = inReader.getUnitStreamDebug("a");
        std::shared_ptr<GPUUnitStream> inputStreamR = inReader.getUnitStreamDebug("x");
        GPUReader outReader = GPUReader("test/data/last_test3.out");
        std::shared_ptr<GPUIntStream> CORRECT_STREAM = outReader.getIntStreamDebug("o");

        // Prepare empty output stream to fill
        int size = CORRECT_STREAM->size;

        std::shared_ptr<GPUIntStream> intermediateStream;
        std::shared_ptr<GPUIntStream> outputStream;

        // Run kernel
        inputStreamV->copy_to_device();
        inputStreamR->copy_to_device();
        inputStream2->copy_to_device();
        intermediateStream = last(inputStreamV, inputStreamR, 0);
        intermediateStream->copy_to_host();
        outputStream = last(intermediateStream, inputStream2, 0);

        outputStream->copy_to_host();
//        outputStream->print();

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
    }

    SECTION("last() twice test with invalids in Unit Stream") {
        //printf("-------------------------\n");
        // Read input and correct output data
        GPUReader inReader = GPUReader("test/data/last_test4.in");
        std::shared_ptr<GPUIntStream> inputStreamV = inReader.getIntStreamDebug("z");
        std::shared_ptr<GPUUnitStream> inputStream2 = inReader.getUnitStreamDebug("x");
        std::shared_ptr<GPUUnitStream> inputStreamR = inReader.getUnitStreamDebug("a");
        GPUReader outReader = GPUReader("test/data/last_test4.out");
        std::shared_ptr<GPUIntStream> CORRECT_STREAM = outReader.getIntStreamDebug("o");

        // Prepare empty output stream to fill
        int size = CORRECT_STREAM->size;
        int sizeAllocated = (size_t) inputStreamR->size * sizeof(int);
        int *host_timestampOut = (int *) malloc(inputStreamR->size * sizeof(int));
        int *host_valueOut = (int *) malloc(inputStreamR->size * sizeof(int));

        memset(host_timestampOut, 0, sizeAllocated);
        memset(host_valueOut, 0, sizeAllocated);

        int *host_timestampOut2 = (int *) malloc(inputStream2->size * sizeof(int));
        int *host_valueOut2 = (int *) malloc(inputStream2->size * sizeof(int));

        memset(host_timestampOut2, 0, inputStream2->size * sizeof(int));
        memset(host_valueOut2, 0, inputStream2->size * sizeof(int));
        std::shared_ptr<GPUIntStream> intermediateStream = std::make_shared<GPUIntStream>(host_timestampOut,
                                                                                          host_valueOut,
                                                                                          inputStreamR->size);
        std::shared_ptr<GPUIntStream> outputStream = std::make_shared<GPUIntStream>(host_timestampOut2, host_valueOut2,
                                                                                    inputStream2->size);
        // Run kernel
        intermediateStream->copy_to_device();
        inputStreamV->copy_to_device();
        inputStreamR->copy_to_device();
        outputStream->copy_to_device();
        inputStream2->copy_to_device();

        intermediateStream = last(inputStreamV, inputStreamR, 0);
        outputStream = last(intermediateStream, inputStream2, 0);

        outputStream->copy_to_host();
        //outputStream->print();

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
        //outputStream->print();
        // Cleanup
        inputStreamV->free_device();
        inputStreamR->free_device();
        intermediateStream->free_device();
        outputStream->free_device();
        inputStream2->free_device();
        free(host_valueOut);
        free(host_timestampOut);
        free(host_valueOut2);
        free(host_timestampOut2);
    }
}