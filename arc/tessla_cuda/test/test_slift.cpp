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
#include <StreamFunctions.cuh>

#include <chrono>
#include <fstream>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <cuda_profiler_api.h>

#include <StreamFunctionsThrust.cuh>
#include <stdio.h>

static std::string TESTDATA_PATH = "test/data/";

TEST_CASE("slift_regular"){
    SECTION("slift(merge) simple example") {
        GPUReader inReader = GPUReader(TESTDATA_PATH + "slift1.in");
        GPUReader outReader = GPUReader(TESTDATA_PATH + "slift1_merge.out");
        std::shared_ptr<GPUIntStream> inputStreamZ = inReader.getIntStream("z");
        std::shared_ptr<GPUIntStream> inputStreamY = inReader.getIntStream("y");
        std::shared_ptr<GPUIntStream> CORRECT_STREAM = outReader.getIntStream("f");
        inputStreamZ->copy_to_device();
        inputStreamY->copy_to_device();

        std::shared_ptr<GPUIntStream> outputStream = slift(inputStreamZ, inputStreamY, MRG);
        outputStream->copy_to_host();

        // Compare kernel result to correct data
        std::vector<int> kernelTimestamps(outputStream->host_timestamp+*(outputStream->host_offset), outputStream->host_timestamp+outputStream->size);
        std::vector<int> kernelValues(outputStream->host_values+*(outputStream->host_offset), outputStream->host_values+outputStream->size);
        std::vector<int> correctTimestamps(CORRECT_STREAM->host_timestamp, CORRECT_STREAM->host_timestamp + CORRECT_STREAM->size);
        std::vector<int> correctValues(CORRECT_STREAM->host_values, CORRECT_STREAM->host_values + CORRECT_STREAM->size);
        REQUIRE(kernelTimestamps == correctTimestamps);
        REQUIRE(kernelValues == correctValues);
        
        // Cleanup
        inputStreamZ->free_device();
        inputStreamY->free_device();
        outputStream->free_device();
        inputStreamZ->free_host();
        outputStream->free_host();
        inputStreamY->free_host();
    }

    SECTION("slift(+) simple example") {
        // Read input and correct output data
        GPUReader inReader = GPUReader(TESTDATA_PATH + "slift1.in");
        GPUReader outReader = GPUReader(TESTDATA_PATH + "slift1.out");
        std::shared_ptr<GPUIntStream> inputStreamZ = inReader.getIntStream("z");
        std::shared_ptr<GPUIntStream> inputStreamY = inReader.getIntStream("y");
        std::shared_ptr<GPUIntStream> CORRECT_STREAM = outReader.getIntStream("f");

        inputStreamZ->copy_to_device();
        inputStreamY->copy_to_device();

        std::shared_ptr<GPUIntStream> outputStream = slift(inputStreamZ, inputStreamY, ADD);
        outputStream->copy_to_host();

        // Compare kernel result to correct data
        std::vector<int> kernelTimestamps(outputStream->host_timestamp+*(outputStream->host_offset), outputStream->host_timestamp+outputStream->size);
        std::vector<int> kernelValues(outputStream->host_values+*(outputStream->host_offset), outputStream->host_values+outputStream->size);
        std::vector<int> correctTimestamps(CORRECT_STREAM->host_timestamp, CORRECT_STREAM->host_timestamp + CORRECT_STREAM->size);
        std::vector<int> correctValues(CORRECT_STREAM->host_values, CORRECT_STREAM->host_values + CORRECT_STREAM->size);
        REQUIRE(kernelTimestamps == correctTimestamps);
        REQUIRE(kernelValues == correctValues);

        // Cleanup
        inputStreamZ->free_device();
        inputStreamY->free_device();
        outputStream->free_device();
        inputStreamZ->free_host();
        outputStream->free_host();
        inputStreamY->free_host();
    }

    SECTION("slift(-) simple example") {
        // Read input and correct output data
        GPUReader inReader = GPUReader("test/data/slift1.in");
        GPUReader outReader = GPUReader("test/data/slift1_minus.out");
        std::shared_ptr<GPUIntStream> inputStreamZ = inReader.getIntStream("z");
        std::shared_ptr<GPUIntStream> inputStreamY = inReader.getIntStream("y");
        std::shared_ptr<GPUIntStream> CORRECT_STREAM = outReader.getIntStream("f");
        inputStreamZ->copy_to_device();
        inputStreamY->copy_to_device();

        std::shared_ptr<GPUIntStream> outputStream = slift(inputStreamZ, inputStreamY, SUB);
        outputStream->copy_to_host();
        // Compare kernel result to correct data
        std::vector<int> kernelTimestamps(outputStream->host_timestamp+*(outputStream->host_offset), outputStream->host_timestamp+outputStream->size);
        std::vector<int> kernelValues(outputStream->host_values+*(outputStream->host_offset), outputStream->host_values+outputStream->size);
        std::vector<int> correctTimestamps(CORRECT_STREAM->host_timestamp, CORRECT_STREAM->host_timestamp + CORRECT_STREAM->size);
        std::vector<int> correctValues(CORRECT_STREAM->host_values, CORRECT_STREAM->host_values + CORRECT_STREAM->size);
        REQUIRE(kernelTimestamps == correctTimestamps);
        REQUIRE(kernelValues == correctValues);

        // Cleanup
        inputStreamZ->free_device();
        inputStreamY->free_device();
        outputStream->free_device();
        inputStreamZ->free_host();
        outputStream->free_host();
        inputStreamY->free_host();
    }

    SECTION("slift(/) simple example") {
        // Read input and correct output data
        GPUReader inReader = GPUReader("test/data/slift1.in");
        GPUReader outReader = GPUReader("test/data/slift1_div.out");
        std::shared_ptr<GPUIntStream> inputStreamZ = inReader.getIntStream("z");
        std::shared_ptr<GPUIntStream> inputStreamY = inReader.getIntStream("y");
        std::shared_ptr<GPUIntStream> CORRECT_STREAM = outReader.getIntStream("f");
        inputStreamZ->copy_to_device();
        inputStreamY->copy_to_device();

        std::shared_ptr<GPUIntStream> outputStream = slift(inputStreamZ, inputStreamY, DIV);
        outputStream->copy_to_host();

        // Compare kernel result to correct data
        std::vector<int> kernelTimestamps(outputStream->host_timestamp+*(outputStream->host_offset), outputStream->host_timestamp+outputStream->size);
        std::vector<int> kernelValues(outputStream->host_values+*(outputStream->host_offset), outputStream->host_values+outputStream->size);
        std::vector<int> correctTimestamps(CORRECT_STREAM->host_timestamp, CORRECT_STREAM->host_timestamp + CORRECT_STREAM->size);
        std::vector<int> correctValues(CORRECT_STREAM->host_values, CORRECT_STREAM->host_values + CORRECT_STREAM->size);
        REQUIRE(kernelTimestamps == correctTimestamps);
        REQUIRE(kernelValues == correctValues);
        
        // Cleanup
        inputStreamZ->free_device();
        inputStreamY->free_device();
        outputStream->free_device();
        inputStreamZ->free_host();
        outputStream->free_host();
        inputStreamY->free_host();
    }

/*
    SECTION("slift(/) simple example with invalids") {
        GPUReader inReader = GPUReader(TESTDATA_PATH + "slift1_div.in");
        GPUReader outReader = GPUReader(TESTDATA_PATH + "slift1_div_inv.out");
        std::shared_ptr<GPUIntStream> inputStreamZ = inReader.getIntStreamDebug("z");
        std::shared_ptr<GPUIntStream> inputStreamY = inReader.getIntStreamDebug("y");
        std::shared_ptr<GPUIntStream> CORRECT_STREAM = outReader.getIntStreamDebug("f");
        inputStreamZ->copy_to_device();
        inputStreamY->copy_to_device();

        try {
            std::shared_ptr<GPUIntStream> outputStream = slift(inputStreamZ, inputStreamY, DIV);
            outputStream->copy_to_host();
            outputStream->print();

            // Compare kernel result to correct data
            std::vector<int> kernelTimestamps(outputStream->host_timestamp+*(outputStream->host_offset), outputStream->host_timestamp+outputStream->size);
            std::vector<int> kernelValues(outputStream->host_values+*(outputStream->host_offset), outputStream->host_values+outputStream->size);
            std::vector<int> correctTimestamps(CORRECT_STREAM->host_timestamp, CORRECT_STREAM->host_timestamp + CORRECT_STREAM->size);
            std::vector<int> correctValues(CORRECT_STREAM->host_values, CORRECT_STREAM->host_values + CORRECT_STREAM->size);

            //REQUIRE(kernelTimestamps == correctTimestamps);
            //REQUIRE(kernelValues == correctValues);

            // Cleanup
            outputStream->free_device();
            outputStream->free_host();
            REQUIRE(false);
        } catch(std::runtime_error& e){
            printf("Caught Runtime error \n");
        }
        inputStreamZ->free_device();
        inputStreamY->free_device();
        inputStreamZ->free_host();
        inputStreamY->free_host();
    }
*/

    SECTION("slift(+) simple example 2") {
        GPUReader inReader = GPUReader("test/data/slift2.in");
        GPUReader outReader = GPUReader("test/data/slift2.out");
        std::shared_ptr<GPUIntStream> inputStreamZ = inReader.getIntStream("z");
        std::shared_ptr<GPUIntStream> inputStreamY = inReader.getIntStream("y");
        std::shared_ptr<GPUIntStream> CORRECT_STREAM = outReader.getIntStream("f");
        inputStreamZ->copy_to_device();
        inputStreamY->copy_to_device();

        std::shared_ptr<GPUIntStream> outputStream = slift(inputStreamZ, inputStreamY, ADD);
        outputStream->copy_to_host();

        // Compare kernel result to correct data
        std::vector<int> kernelTimestamps(outputStream->host_timestamp+*(outputStream->host_offset), outputStream->host_timestamp+outputStream->size);
        std::vector<int> kernelValues(outputStream->host_values+*(outputStream->host_offset), outputStream->host_values+outputStream->size);
        std::vector<int> correctTimestamps(CORRECT_STREAM->host_timestamp, CORRECT_STREAM->host_timestamp + CORRECT_STREAM->size);
        std::vector<int> correctValues(CORRECT_STREAM->host_values, CORRECT_STREAM->host_values + CORRECT_STREAM->size);
        REQUIRE(kernelTimestamps == correctTimestamps);
        REQUIRE(kernelValues == correctValues);

        // Cleanup
        inputStreamZ->free_device();
        inputStreamY->free_device();
        outputStream->free_device();
        inputStreamZ->free_host();
        outputStream->free_host();
        inputStreamY->free_host();
    }

    SECTION("slift(+) simple example 2 flipped") {
        // Read input and correct output data
        GPUReader inReader = GPUReader("test/data/slift2.in");
        GPUReader outReader = GPUReader("test/data/slift2.out");
        std::shared_ptr<GPUIntStream> inputStreamZ = inReader.getIntStream("z");
        std::shared_ptr<GPUIntStream> inputStreamY = inReader.getIntStream("y");
        std::shared_ptr<GPUIntStream> CORRECT_STREAM = outReader.getIntStream("f");
        inputStreamZ->copy_to_device();
        inputStreamY->copy_to_device();

        std::shared_ptr<GPUIntStream> outputStream = slift(inputStreamY, inputStreamZ, ADD);
        outputStream->copy_to_host();

        // Compare kernel result to correct data
        std::vector<int> kernelTimestamps(outputStream->host_timestamp+*(outputStream->host_offset), outputStream->host_timestamp+outputStream->size);
        std::vector<int> kernelValues(outputStream->host_values+*(outputStream->host_offset), outputStream->host_values+outputStream->size);
        std::vector<int> correctTimestamps(CORRECT_STREAM->host_timestamp, CORRECT_STREAM->host_timestamp + CORRECT_STREAM->size);
        std::vector<int> correctValues(CORRECT_STREAM->host_values, CORRECT_STREAM->host_values + CORRECT_STREAM->size);
        REQUIRE(kernelTimestamps == correctTimestamps);
        REQUIRE(kernelValues == correctValues);
        // Cleanup

        inputStreamZ->free_device();
        inputStreamY->free_device();
        outputStream->free_device();
        inputStreamZ->free_host();
        outputStream->free_host();
        inputStreamY->free_host();
    }

    SECTION("slift(+) empty example") {
        // Read input and correct output data
        GPUReader inReader = GPUReader(TESTDATA_PATH + "slift3.in");
        GPUReader outReader = GPUReader(TESTDATA_PATH + "slift3.out");
        std::shared_ptr<GPUIntStream> inputStreamZ = inReader.getIntStreamDebug("z");
        std::shared_ptr<GPUIntStream> inputStreamY = inReader.getIntStream("y");
        std::shared_ptr<GPUIntStream> CORRECT_STREAM = outReader.getIntStreamDebug("f");
        inputStreamZ->copy_to_device();
        inputStreamY->copy_to_device();

        std::shared_ptr<GPUIntStream> outputStream = slift(inputStreamY, inputStreamZ, ADD);
        outputStream->copy_to_host();

        // Compare kernel result to correct data
        std::vector<int> kernelTimestamps(outputStream->host_timestamp+*(outputStream->host_offset), outputStream->host_timestamp+outputStream->size);
        std::vector<int> kernelValues(outputStream->host_values+*(outputStream->host_offset), outputStream->host_values+outputStream->size);
        std::vector<int> correctTimestamps(CORRECT_STREAM->host_timestamp, CORRECT_STREAM->host_timestamp + CORRECT_STREAM->size);
        std::vector<int> correctValues(CORRECT_STREAM->host_values, CORRECT_STREAM->host_values + CORRECT_STREAM->size);
        REQUIRE(kernelTimestamps == correctTimestamps);
        REQUIRE(kernelValues == correctValues);
        // Cleanup

        inputStreamZ->free_device();
        inputStreamY->free_device();
        outputStream->free_device();
        inputStreamZ->free_host();
        outputStream->free_host();
        inputStreamY->free_host();
    }

    SECTION("slift(+) large example") {
        GPUReader inReader = GPUReader("test/data/slift4.in");
        GPUReader outReader = GPUReader("test/data/slift4.out");
        std::shared_ptr<GPUIntStream> inputStreamZ = inReader.getIntStream("z");
        std::shared_ptr<GPUIntStream> inputStreamY = inReader.getIntStream("y");
        std::shared_ptr<GPUIntStream> CORRECT_STREAM = outReader.getIntStream("f");
        inputStreamZ->copy_to_device();
        inputStreamY->copy_to_device();

        std::shared_ptr<GPUIntStream> outputStream = slift(inputStreamY, inputStreamZ, ADD);
        outputStream->copy_to_host();
        
        // Compare kernel result to correct data
        std::vector<int> kernelTimestamps(outputStream->host_timestamp+*(outputStream->host_offset), outputStream->host_timestamp+outputStream->size);
        std::vector<int> kernelValues(outputStream->host_values+*(outputStream->host_offset), outputStream->host_values+outputStream->size);
        std::vector<int> correctTimestamps(CORRECT_STREAM->host_timestamp, CORRECT_STREAM->host_timestamp + CORRECT_STREAM->size);
        std::vector<int> correctValues(CORRECT_STREAM->host_values, CORRECT_STREAM->host_values + CORRECT_STREAM->size);
        REQUIRE(kernelTimestamps == correctTimestamps);
        REQUIRE(kernelValues == correctValues);
        // Cleanup

        inputStreamZ->free_device();
        inputStreamY->free_device();
        outputStream->free_device();
        inputStreamZ->free_host();
        outputStream->free_host();
        inputStreamY->free_host();
    }
}

//////////////////
// Thrust slift //
//////////////////

TEST_CASE("slift_thrust()"){
    SECTION("slift(merge) simple example") {
        GPUReader inReader = GPUReader("test/data/slift1.in");
        GPUReader outReader = GPUReader("test/data/slift1_merge.out");
        std::shared_ptr<GPUIntStream> inputStreamZ = inReader.getIntStream("z");
        std::shared_ptr<GPUIntStream> inputStreamY = inReader.getIntStream("y");
        std::shared_ptr<GPUIntStream> CORRECT_STREAM = outReader.getIntStream("f");
        inputStreamZ->copy_to_device();
        inputStreamY->copy_to_device();

        std::shared_ptr<GPUIntStream> outputStream = slift_thrust(inputStreamZ, inputStreamY,TH_OP_merge, 0);
        outputStream->copy_to_host();

        // Compare kernel result to correct data
        std::vector<int> kernelTimestamps(outputStream->host_timestamp+*(outputStream->host_offset), outputStream->host_timestamp+outputStream->size);
        std::vector<int> kernelValues(outputStream->host_values+*(outputStream->host_offset), outputStream->host_values+outputStream->size);
        std::vector<int> correctTimestamps(CORRECT_STREAM->host_timestamp, CORRECT_STREAM->host_timestamp + CORRECT_STREAM->size);
        std::vector<int> correctValues(CORRECT_STREAM->host_values, CORRECT_STREAM->host_values + CORRECT_STREAM->size);
        REQUIRE(kernelTimestamps == correctTimestamps);
        REQUIRE(kernelValues == correctValues);

        // Cleanup
        inputStreamZ->free_device();
        inputStreamY->free_device();
        outputStream->free_device();
        inputStreamZ->free_host();
        outputStream->free_host();
        inputStreamY->free_host();
    }

    SECTION("slift(+) simple example") {
        GPUReader inReader = GPUReader("test/data/slift1.in");
        GPUReader outReader = GPUReader("test/data/slift1.out");
        std::shared_ptr<GPUIntStream> inputStreamZ = inReader.getIntStream("z");
        std::shared_ptr<GPUIntStream> inputStreamY = inReader.getIntStream("y");
        std::shared_ptr<GPUIntStream> CORRECT_STREAM = outReader.getIntStream("f");
        inputStreamZ->copy_to_device();
        inputStreamY->copy_to_device();

        std::shared_ptr<GPUIntStream> outputStream = slift_thrust(inputStreamZ, inputStreamY,TH_OP_add, 0);
        outputStream->copy_to_host();

        // Compare kernel result to correct data
        std::vector<int> kernelTimestamps(outputStream->host_timestamp+*(outputStream->host_offset), outputStream->host_timestamp+outputStream->size);
        std::vector<int> kernelValues(outputStream->host_values+*(outputStream->host_offset), outputStream->host_values+outputStream->size);
        std::vector<int> correctTimestamps(CORRECT_STREAM->host_timestamp, CORRECT_STREAM->host_timestamp + CORRECT_STREAM->size);
        std::vector<int> correctValues(CORRECT_STREAM->host_values, CORRECT_STREAM->host_values + CORRECT_STREAM->size);
        REQUIRE(kernelTimestamps == correctTimestamps);
        REQUIRE(kernelValues == correctValues);
        
        // Cleanup
        inputStreamZ->free_device();
        inputStreamY->free_device();
        outputStream->free_device();
        inputStreamZ->free_host();
        outputStream->free_host();
        inputStreamY->free_host();
    }

    SECTION("slift(-) simple example") {
        GPUReader inReader = GPUReader("test/data/slift1.in");
        GPUReader outReader = GPUReader("test/data/slift1_minus.out");
        std::shared_ptr<GPUIntStream> inputStreamZ = inReader.getIntStream("z");
        std::shared_ptr<GPUIntStream> inputStreamY = inReader.getIntStream("y");
        std::shared_ptr<GPUIntStream> CORRECT_STREAM = outReader.getIntStream("f");
        inputStreamZ->copy_to_device();
        inputStreamY->copy_to_device();

        std::shared_ptr<GPUIntStream> outputStream = slift_thrust(inputStreamZ, inputStreamY, TH_OP_subtract, 0);
        outputStream->copy_to_host();

        // Compare kernel result to correct data
        std::vector<int> kernelTimestamps(outputStream->host_timestamp+*(outputStream->host_offset), outputStream->host_timestamp+outputStream->size);
        std::vector<int> kernelValues(outputStream->host_values+*(outputStream->host_offset), outputStream->host_values+outputStream->size);
        std::vector<int> correctTimestamps(CORRECT_STREAM->host_timestamp, CORRECT_STREAM->host_timestamp + CORRECT_STREAM->size);
        std::vector<int> correctValues(CORRECT_STREAM->host_values, CORRECT_STREAM->host_values + CORRECT_STREAM->size);
        REQUIRE(kernelTimestamps == correctTimestamps);
        REQUIRE(kernelValues == correctValues);
        
        // Cleanup
        inputStreamZ->free_device();
        inputStreamY->free_device();
        outputStream->free_device();
        inputStreamZ->free_host();
        outputStream->free_host();
        inputStreamY->free_host();
    }

    SECTION("slift(/) simple example") {
        GPUReader inReader = GPUReader("test/data/slift1.in");
        GPUReader outReader = GPUReader("test/data/slift1_div.out");
        std::shared_ptr<GPUIntStream> inputStreamZ = inReader.getIntStream("z");
        std::shared_ptr<GPUIntStream> inputStreamY = inReader.getIntStream("y");
        std::shared_ptr<GPUIntStream> CORRECT_STREAM = outReader.getIntStream("f");
        inputStreamZ->copy_to_device();
        inputStreamY->copy_to_device();

        std::shared_ptr<GPUIntStream> outputStream = slift_thrust(inputStreamZ, inputStreamY,TH_OP_divide, 0);
        outputStream->copy_to_host();

        // Compare kernel result to correct data
        std::vector<int> kernelTimestamps(outputStream->host_timestamp+*(outputStream->host_offset), outputStream->host_timestamp+outputStream->size);
        std::vector<int> kernelValues(outputStream->host_values+*(outputStream->host_offset), outputStream->host_values+outputStream->size);
        std::vector<int> correctTimestamps(CORRECT_STREAM->host_timestamp, CORRECT_STREAM->host_timestamp + CORRECT_STREAM->size);
        std::vector<int> correctValues(CORRECT_STREAM->host_values, CORRECT_STREAM->host_values + CORRECT_STREAM->size);
        REQUIRE(kernelTimestamps == correctTimestamps);
        REQUIRE(kernelValues == correctValues);

        // Cleanup
        inputStreamZ->free_device();
        inputStreamY->free_device();
        outputStream->free_device();
        inputStreamZ->free_host();
        outputStream->free_host();
        inputStreamY->free_host();
    }


    SECTION("slift(/) simple example with invalids") {
        GPUReader inReader = GPUReader(TESTDATA_PATH + "slift1_div.in");
        GPUReader outReader = GPUReader(TESTDATA_PATH + "slift1_div_inv.out");
        std::shared_ptr<GPUIntStream> inputStreamZ = inReader.getIntStreamDebug("z");
        std::shared_ptr<GPUIntStream> inputStreamY = inReader.getIntStreamDebug("y");
        std::shared_ptr<GPUIntStream> CORRECT_STREAM = outReader.getIntStreamDebug("f");
        inputStreamZ->copy_to_device();
        inputStreamY->copy_to_device();

        try {
            std::shared_ptr<GPUIntStream> outputStream = slift_thrust(inputStreamZ, inputStreamY,TH_OP_divide, 0);
            outputStream->copy_to_host();

            // Compare kernel result to correct data
            std::vector<int> kernelTimestamps(outputStream->host_timestamp+*(outputStream->host_offset), outputStream->host_timestamp+outputStream->size);
            std::vector<int> kernelValues(outputStream->host_values+*(outputStream->host_offset), outputStream->host_values+outputStream->size);
            std::vector<int> correctTimestamps(CORRECT_STREAM->host_timestamp, CORRECT_STREAM->host_timestamp + CORRECT_STREAM->size);
            std::vector<int> correctValues(CORRECT_STREAM->host_values, CORRECT_STREAM->host_values + CORRECT_STREAM->size);
            REQUIRE(kernelTimestamps == correctTimestamps);
            REQUIRE(kernelValues == correctValues);

            // Cleanup
            outputStream->free_device();
            outputStream->free_host();
            REQUIRE(false);
        } catch(std::runtime_error& e) {
            printf("Caught Runtime error \n");
        }

        inputStreamZ->free_device();
        inputStreamY->free_device();
        inputStreamZ->free_host();
        inputStreamY->free_host();
    }

    SECTION("slift(+) simple example 2") {
       //printf("----slift 2-----");
        GPUReader inReader = GPUReader(TESTDATA_PATH + "slift2.in");
        GPUReader outReader = GPUReader(TESTDATA_PATH + "slift2.out");
        std::shared_ptr<GPUIntStream> inputStreamZ = inReader.getIntStream("z");
        std::shared_ptr<GPUIntStream> inputStreamY = inReader.getIntStream("y");
        std::shared_ptr<GPUIntStream> CORRECT_STREAM = outReader.getIntStream("f");
        // Run kernel
        inputStreamZ->copy_to_device();
        inputStreamY->copy_to_device();

        std::shared_ptr<GPUIntStream> outputStream = slift_thrust(inputStreamZ, inputStreamY,TH_OP_add, 0);
        outputStream->copy_to_host();

        // Compare kernel result to correct data
        std::vector<int> kernelTimestamps(outputStream->host_timestamp+*(outputStream->host_offset), outputStream->host_timestamp+outputStream->size);
        std::vector<int> kernelValues(outputStream->host_values+*(outputStream->host_offset), outputStream->host_values+outputStream->size);
        std::vector<int> correctTimestamps(CORRECT_STREAM->host_timestamp, CORRECT_STREAM->host_timestamp + CORRECT_STREAM->size);
        std::vector<int> correctValues(CORRECT_STREAM->host_values, CORRECT_STREAM->host_values + CORRECT_STREAM->size);
        REQUIRE(kernelTimestamps == correctTimestamps);
        REQUIRE(kernelValues == correctValues);

        // Cleanup
        inputStreamZ->free_device();
        inputStreamY->free_device();
        outputStream->free_device();
        inputStreamZ->free_host();
        outputStream->free_host();
        inputStreamY->free_host();
    }

    SECTION("slift(+) simple example 2 flipped") {
        GPUReader inReader = GPUReader("test/data/slift2.in");
        GPUReader outReader = GPUReader("test/data/slift2.out");
        std::shared_ptr<GPUIntStream> inputStreamZ = inReader.getIntStream("z");
        std::shared_ptr<GPUIntStream> inputStreamY = inReader.getIntStream("y");
        std::shared_ptr<GPUIntStream> CORRECT_STREAM = outReader.getIntStream("f");
        inputStreamZ->copy_to_device();
        inputStreamY->copy_to_device();

        std::shared_ptr<GPUIntStream> outputStream = slift_thrust(inputStreamY, inputStreamZ,TH_OP_add, 0);
        outputStream->copy_to_host();

        // Compare kernel result to correct data
        std::vector<int> kernelTimestamps(outputStream->host_timestamp+*(outputStream->host_offset), outputStream->host_timestamp+outputStream->size);
        std::vector<int> kernelValues(outputStream->host_values+*(outputStream->host_offset), outputStream->host_values+outputStream->size);
        std::vector<int> correctTimestamps(CORRECT_STREAM->host_timestamp, CORRECT_STREAM->host_timestamp + CORRECT_STREAM->size);
        std::vector<int> correctValues(CORRECT_STREAM->host_values, CORRECT_STREAM->host_values + CORRECT_STREAM->size);
        REQUIRE(kernelTimestamps == correctTimestamps);
        REQUIRE(kernelValues == correctValues);
        
        // Cleanup
        inputStreamZ->free_device();
        inputStreamY->free_device();
        outputStream->free_device();
        inputStreamZ->free_host();
        outputStream->free_host();
        inputStreamY->free_host();
    }

    SECTION("slift(+) empty example") {
        // Read input and correct output data
        GPUReader inReader = GPUReader(TESTDATA_PATH + "slift3.in");
        GPUReader outReader = GPUReader(TESTDATA_PATH + "slift3.out");
        std::shared_ptr<GPUIntStream> inputStreamZ = inReader.getIntStreamDebug("z");
        std::shared_ptr<GPUIntStream> inputStreamY = inReader.getIntStreamDebug("y");
        std::shared_ptr<GPUIntStream> CORRECT_STREAM = outReader.getIntStreamDebug("f");

        inputStreamZ->copy_to_device();
        inputStreamY->copy_to_device();

        std::shared_ptr<GPUIntStream> outputStream = slift_thrust(inputStreamY, inputStreamZ,TH_OP_add, 0);
        outputStream->copy_to_host();

        // Compare kernel result to correct data
        std::vector<int> kernelTimestamps(outputStream->host_timestamp+*(outputStream->host_offset), outputStream->host_timestamp+outputStream->size);
        std::vector<int> kernelValues(outputStream->host_values+*(outputStream->host_offset), outputStream->host_values+outputStream->size);
        std::vector<int> correctTimestamps(CORRECT_STREAM->host_timestamp, CORRECT_STREAM->host_timestamp + CORRECT_STREAM->size);
        std::vector<int> correctValues(CORRECT_STREAM->host_values, CORRECT_STREAM->host_values + CORRECT_STREAM->size);
        REQUIRE(kernelTimestamps == correctTimestamps);
        REQUIRE(kernelValues == correctValues);

        // Cleanup
        inputStreamZ->free_device();
        inputStreamY->free_device();
        outputStream->free_device();
        inputStreamZ->free_host();
        outputStream->free_host();
        inputStreamY->free_host();
    }

    SECTION("slift(+) large example") {
        GPUReader inReader = GPUReader("test/data/slift4.in");
        GPUReader outReader = GPUReader("test/data/slift4.out");
        std::shared_ptr<GPUIntStream> inputStreamZ = inReader.getIntStream("z");
        std::shared_ptr<GPUIntStream> inputStreamY = inReader.getIntStream("y");
        std::shared_ptr<GPUIntStream> CORRECT_STREAM = outReader.getIntStream("f");
        inputStreamZ->copy_to_device();
        inputStreamY->copy_to_device();

        std::shared_ptr<GPUIntStream> outputStream = slift_thrust(inputStreamY, inputStreamZ,TH_OP_add, 0);
        outputStream->copy_to_host();
        
        // Compare kernel result to correct data
        std::vector<int> kernelTimestamps(outputStream->host_timestamp+*(outputStream->host_offset), outputStream->host_timestamp+outputStream->size);
        std::vector<int> kernelValues(outputStream->host_values+*(outputStream->host_offset), outputStream->host_values+outputStream->size);
        std::vector<int> correctTimestamps(CORRECT_STREAM->host_timestamp, CORRECT_STREAM->host_timestamp + CORRECT_STREAM->size);
        std::vector<int> correctValues(CORRECT_STREAM->host_values, CORRECT_STREAM->host_values + CORRECT_STREAM->size);
        REQUIRE(kernelTimestamps == correctTimestamps);
        REQUIRE(kernelValues == correctValues);

        // Cleanup
        inputStreamZ->free_device();
        inputStreamY->free_device();
        outputStream->free_device();
        inputStreamZ->free_host();
        outputStream->free_host();
        inputStreamY->free_host();
    }
}