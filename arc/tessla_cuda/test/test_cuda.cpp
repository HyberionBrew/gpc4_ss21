
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



/*

TEST_CASE("extensive stream ops"){
    SECTION("last|time"){
        // Read input and correct output data
        GPUReader inReader = GPUReader("../test/data/extensive_benchmark.in");
        GPUIntStream inputStreamZ = inReader.getIntStreamDebug("z");
        GPUUnitStream inputStreamA = inReader.getUnitStreamDebug("a");
        GPUUnitStream inputStreamB = inReader.getUnitStreamDebug("b");
        GPUReader outReader = GPUReader("../test/data/extensive_benchmark.out");
        GPUIntStream CORRECT_STREAM = outReader.getIntStreamDebug("y5");

        // Prepare empty output stream to fill
        int size = CORRECT_STREAM.size;
        GPUIntStream y1,y2,y3,y4,y5;

        // Run kernel
        inputStreamZ.copy_to_device();
        inputStreamA.copy_to_device();
        inputStreamB.copy_to_device();
       // inputStreamV.print();
      //  inputStreamR.print();
        last(&inputStreamZ, &inputStreamA, &y1, 0);
        time(&y1,&y2,0);
        last(&y2, &inputStreamB, &y3, 0);
        time(&y3,&y4,0);
        last(&y4, &inputStreamB, &y5, 0);
        //inputStreamR.print();
        y5.copy_to_host();

        //outputStream.print();
        // Compare kernel result to correct data
        std::vector<int> kernelTimestamps(y5.host_timestamp+*(y5.host_offset), y5.host_timestamp+y5.size);
        std::vector<int> kernelValues(y5.host_values+*(y5.host_offset), y5.host_values+y5.size);
        std::vector<int> correctTimestamps(CORRECT_STREAM.host_timestamp, CORRECT_STREAM.host_timestamp + CORRECT_STREAM.size);
        std::vector<int> correctValues(CORRECT_STREAM.host_values, CORRECT_STREAM.host_values + CORRECT_STREAM.size);

        REQUIRE(kernelTimestamps == correctTimestamps);
        REQUIRE(kernelValues == correctValues);
        // Cleanup
        inputStreamZ.free_device();
        inputStreamA.free_device();
        inputStreamB.free_device();
        inputStreamZ.free_host();
        y1.free_device();
        y2.free_device();
        y3.free_device();
        y4.free_device();
        y5.free_device();
    }

    SECTION("last|time|delay"){
        // Read input and correct output data
        printf("------------- \n");
        GPUReader inReader = GPUReader("../test/data/extensive_benchmark.in");
        GPUIntStream inputStreamZ = inReader.getIntStreamDebug("z");
        GPUUnitStream inputStreamA = inReader.getUnitStreamDebug("a");
        GPUUnitStream inputStreamB = inReader.getUnitStreamDebug("b");
        GPUReader outReader = GPUReader("../test/data/extensive_benchmark2.out");
        GPUIntStream CORRECT_STREAM1 = outReader.getIntStreamDebug("y1");
        GPUIntStream CORRECT_STREAM2 = outReader.getIntStreamDebug("y2");
        //TODO! crate nil streams in output file!
        //otherwise comment out the below
        GPUUnitStream CORRECT_STREAM3 = outReader.getUnitStreamDebug("y3");
        GPUUnitStream CORRECT_STREAM4 = outReader.getUnitStreamDebug("y4");
        GPUIntStream CORRECT_STREAM5 = outReader.getIntStreamDebug("y5");
        GPUUnitStream CORRECT_STREAM6 = outReader.getUnitStreamDebug("y6");
        // Prepare empty output stream to fill
        //int size = CORRECT_STREAM.size;
        GPUIntStream y1,y2,y5;
        GPUUnitStream y3,y4,y6;
        // Run kernel
        inputStreamZ.copy_to_device();
        inputStreamA.copy_to_device();
        inputStreamB.copy_to_device();
       // inputStreamV.print();
      //  inputStreamR.print();
        //printf("before delay \n");
        last(&inputStreamZ, &inputStreamA, &y1, 0);
        time(&y1,&y2,0);
        printf("before delay \n");
        delay(&y2,&inputStreamA,&y3,0);
        delay(&inputStreamZ,&y3,&y4,0);
        last(&y2, &y4, &y5, 0);
        printf("hi!\n");
        delay(&y5,&y4,&y6,0);
        printf("hi!\n");
        //inputStreamR.print();
        y1.copy_to_host();
        printf("hi!\n");
        y2.copy_to_host();
        printf("hi!\n");
        y6.copy_to_host();
        printf("hi!\n");
        //y2.print();
        //outputStream.print();
        // Compare kernel result to correct data
        std::vector<int> kernelTimestamps(y1.host_timestamp+*(y1.host_offset), y1.host_timestamp+y1.size);
        std::vector<int> kernelValues(y1.host_values+*(y1.host_offset), y1.host_values+y1.size);
        std::vector<int> correctTimestamps(CORRECT_STREAM1.host_timestamp, CORRECT_STREAM1.host_timestamp + CORRECT_STREAM1.size);
        std::vector<int> correctValues(CORRECT_STREAM1.host_values, CORRECT_STREAM1.host_values + CORRECT_STREAM1.size);

        REQUIRE(kernelTimestamps == correctTimestamps);
        REQUIRE(kernelValues == correctValues);

        std::vector<int> kernelTimestamps1(y2.host_timestamp+*(y2.host_offset), y2.host_timestamp+y2.size);
        std::vector<int> kernelValues1(y2.host_values+*(y2.host_offset), y2.host_values+y2.size);
        std::vector<int> correctTimestamps1(CORRECT_STREAM2.host_timestamp, CORRECT_STREAM2.host_timestamp + CORRECT_STREAM2.size);
        std::vector<int> correctValues1(CORRECT_STREAM2.host_values, CORRECT_STREAM2.host_values + CORRECT_STREAM2.size);

        REQUIRE(kernelTimestamps1 == correctTimestamps1);
        REQUIRE(kernelValues1 == correctValues1);
        /*
        std::vector<int> kernelTimestamps2(y6.host_timestamp+*(y6.host_offset), y6.host_timestamp+y6.size);
        //std::vector<int> kernelValues(y6.host_values+*(y6.host_offset), y6.host_values+y6.size);
        std::vector<int> correctTimestamps2(CORRECT_STREAM6.host_timestamp, CORRECT_STREAM6.host_timestamp + CORRECT_STREAM6.size);
        //std::vector<int> correctValues(CORRECT_STREAM6.host_values, CORRECT_STREAM6.host_values + CORRECT_STREAM6.size);

        REQUIRE(kernelTimestamps2 == correctTimestamps2);
        //REQUIRE(kernelValues == correctValues);
        *//*
        // Cleanup
        inputStreamZ.free_device();
        inputStreamA.free_device();
        inputStreamB.free_device();
        y1.free_device();
        y2.free_device();
        y3.free_device();
        y4.free_device();
        y5.free_device();
        y6.free_device();


    }

}

TEST_CASE("Basic Stream Operations") {
    SECTION("delay()") {
        SECTION("delay() with small dataset") {
            // Read input and correct output data
            GPUReader inReader = GPUReader("../test/data/bt_delay.in");
            GPUIntStream inputStreamD = inReader.getIntStreamDebug("d");
            GPUUnitStream inputStreamR = inReader.getUnitStreamDebug("r");
            GPUReader outReader = GPUReader("../test/data/bt_delay.out");
            GPUUnitStream CORRECT_STREAM = outReader.getUnitStreamDebug("y");

            // Prepare empty output stream to fill
            int size = inputStreamR.size;
            int sizeAllocated = (size_t) size * sizeof(int);
            int *host_timestampOut = (int *) malloc(size * sizeof(int));
            memset(host_timestampOut, -1, sizeAllocated);
            GPUUnitStream outputStream(host_timestampOut, size);

            // Run kernel
            inputStreamD.copy_to_device();
            inputStreamR.copy_to_device();
            outputStream.copy_to_device();
            delay(&inputStreamD, &inputStreamR, &outputStream, 0);
            outputStream.copy_to_host();

            // Compare kernel result to correct data
            int* resultStart = outputStream.host_timestamp + *outputStream.host_offset;
            std::vector<int> kernelTimestamps(resultStart, resultStart + sizeof(resultStart) / sizeof(int));
            std::vector<int> correctTimestamps(CORRECT_STREAM.host_timestamp, CORRECT_STREAM.host_timestamp + sizeof(CORRECT_STREAM.host_timestamp) / sizeof(int));
            REQUIRE(kernelTimestamps == correctTimestamps);

            // Cleanup
            inputStreamD.free_device();
            inputStreamR.free_device();
            outputStream.free_device();
            free(host_timestampOut);
        }



        SECTION("delay() with middle dataset") {
            // Read input and correct output data
            GPUReader inReader = GPUReader("../test/data/bt_delay.middle.in");
            GPUIntStream inputStreamD = inReader.getIntStreamDebug("z");
            GPUUnitStream inputStreamR = inReader.getUnitStreamDebug("a");
            GPUReader outReader = GPUReader("../test/data/bt_delay.middle.out");
            GPUUnitStream CORRECT_STREAM = outReader.getUnitStreamDebug("y");

            // Prepare empty output stream to fill
            int size = CORRECT_STREAM.size;
            int sizeAllocated = (size_t) size * sizeof(int);
            int *host_timestampOut = (int *) malloc(size * sizeof(int));
            memset(host_timestampOut, 0, sizeAllocated);
            GPUUnitStream outputStream(host_timestampOut, size);

            // Run kernel
            inputStreamD.copy_to_device();
            inputStreamR.copy_to_device();
            outputStream.copy_to_device();
            delay(&inputStreamD, &inputStreamR, &outputStream, 0);
            outputStream.copy_to_host();

            // Compare kernel result to correct data
            int* resultStart = outputStream.host_timestamp + *outputStream.host_offset;
            std::vector<int> kernelTimestamps(resultStart, resultStart + sizeof(resultStart) / sizeof(int));
            std::vector<int> correctTimestamps(CORRECT_STREAM.host_timestamp, CORRECT_STREAM.host_timestamp + sizeof(CORRECT_STREAM.host_timestamp) / sizeof(int));
            REQUIRE(kernelTimestamps == correctTimestamps);

            // Cleanup
            inputStreamD.free_device();
            inputStreamR.free_device();
            outputStream.free_device();
            free(host_timestampOut);
        }

        SECTION("delay() with bigger dataset") {
            // Read input and correct output data
            GPUReader inReader = GPUReader("../test/data/bt_delay.bigger.in");
            GPUIntStream inputStreamD = inReader.getIntStreamDebug("z");
            GPUUnitStream inputStreamR = inReader.getUnitStreamDebug("a");
            GPUReader outReader = GPUReader("../test/data/bt_delay.bigger.out");
            GPUUnitStream CORRECT_STREAM = outReader.getUnitStreamDebug("y");

            // Prepare empty output stream to fill
            int size = CORRECT_STREAM.size;
            int sizeAllocated = (size_t) size * sizeof(int);
            int *host_timestampOut = (int *) malloc(size * sizeof(int));
            memset(host_timestampOut, 0, sizeAllocated);
            GPUUnitStream outputStream(host_timestampOut, size);

            // Run kernel
            inputStreamD.copy_to_device();
            inputStreamR.copy_to_device();
            outputStream.copy_to_device();
            delay(&inputStreamD, &inputStreamR, &outputStream, 0);
            outputStream.copy_to_host();

            // Compare kernel result to correct data
            int* resultStart = outputStream.host_timestamp + *outputStream.host_offset;
            std::vector<int> kernelTimestamps(resultStart, resultStart + sizeof(resultStart) / sizeof(int));
            std::vector<int> correctTimestamps(CORRECT_STREAM.host_timestamp, CORRECT_STREAM.host_timestamp + sizeof(CORRECT_STREAM.host_timestamp) / sizeof(int));
            REQUIRE(kernelTimestamps == correctTimestamps);
            // Cleanup
            inputStreamD.free_device();
            inputStreamR.free_device();
            outputStream.free_device();
            free(host_timestampOut);
        }
    }

    SECTION("merge() (not implemented yet)") {
        // Read input and correct output data
        GPUReader inReader = GPUReader("../test/data/bt_merge.in");
        GPUIntStream inputStreamX = inReader.getIntStreamDebug("x");
        GPUIntStream inputStreamY = inReader.getIntStreamDebug("y");
        GPUReader outReader = GPUReader("../test/data/bt_merge.out");
        GPUIntStream CORRECT_STREAM = outReader.getIntStreamDebug("z");

        // Prepare empty output stream to fill
        int size = CORRECT_STREAM.size;
        int sizeAllocated = (size_t) size * sizeof(int);
        int *host_timestampOut = (int *) malloc(size * sizeof(int));
        int *host_valueOut = (int*) malloc(size * sizeof(int));
        memset(host_timestampOut, 0, sizeAllocated);
        memset(host_valueOut, 0, sizeAllocated);
        GPUIntStream outputStream(host_timestampOut, host_valueOut, size);

        // Run kernel
        inputStreamX.copy_to_device();
        inputStreamY.copy_to_device();
        outputStream.copy_to_device();
        // merge(...) // TODO: Implement
        outputStream.copy_to_host();

        // Compare kernel result to correct data
        std::vector<int> kernelTimestamps(outputStream.host_timestamp, outputStream.host_timestamp + sizeof(outputStream.host_timestamp) / sizeof(int));
        std::vector<int> kernelValues(outputStream.host_values, outputStream.host_values + sizeof(outputStream.host_values) / sizeof(int));
        std::vector<int> correctTimestamps(CORRECT_STREAM.host_timestamp, CORRECT_STREAM.host_timestamp + sizeof(CORRECT_STREAM.host_timestamp) / sizeof(int));
        std::vector<int> correctValues(CORRECT_STREAM.host_values, CORRECT_STREAM.host_values + sizeof(CORRECT_STREAM.host_values) / sizeof(int));
        //REQUIRE(kernelTimestamps == correctTimestamps);
        //REQUIRE(kernelValues == correctValues);

        // Cleanup
        inputStreamX.free_device();
        inputStreamY.free_device();
        outputStream.free_device();
        free(host_valueOut);
        free(host_timestampOut);
    }



}

TEST_CASE("GPUReader (bt_delay.in)") {
    GPUReader reader = GPUReader("../test/data/bt_delay.in");

    SECTION("Read existing GPUIntStream correctly") {
        int CORRECT_STREAM_SIZE = 6;
        int CORRECT_STREAM_TIMESTAMPS[CORRECT_STREAM_SIZE] = {1, 2, 3, 4, 6, 7};
        int CORRECT_STREAM_INTS[CORRECT_STREAM_SIZE] = {1, 2, 4, 1, 2, 3};

        GPUIntStream dStream = reader.getIntStreamDebug("d");
        REQUIRE(dStream.size == CORRECT_STREAM_SIZE);
        for (int i = 0; i < CORRECT_STREAM_SIZE; i++) {
            REQUIRE(dStream.host_timestamp[i] == CORRECT_STREAM_TIMESTAMPS[i]);
            REQUIRE(dStream.host_values[i] == CORRECT_STREAM_INTS[i]);
        }
    }

    SECTION("Read existing GPUUnitStream correctly") {
        int CORRECT_STREAM_SIZE = 3;
        int CORRECT_STREAM_TIMESTAMPS[CORRECT_STREAM_SIZE] = {2, 7, 8};

        GPUUnitStream dStream = reader.getUnitStreamDebug("r");
        REQUIRE(dStream.size == CORRECT_STREAM_SIZE);
        for (int i = 0; i < CORRECT_STREAM_SIZE; i++) {
            REQUIRE(dStream.host_timestamp[i] == CORRECT_STREAM_TIMESTAMPS[i]);
        }
    }

}
*/
