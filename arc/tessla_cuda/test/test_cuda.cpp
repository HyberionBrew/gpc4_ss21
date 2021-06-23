
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

TEST_CASE("last_thrust()"){

    SECTION("last() tuwel example") {
        // Read input and correct output data
        
        GPUReader inReader = GPUReader("../test/data/bt_last.in");
        GPUIntStream inputStreamV = inReader.getGPUIntStream("v");
        GPUUnitStream inputStreamR = inReader.getGPUUnitStream("r");
        GPUReader outReader = GPUReader("../test/data/bt_last.out");
        GPUIntStream CORRECT_STREAM = outReader.getGPUIntStream("y");

        // Prepare empty output stream to fill
        int size = CORRECT_STREAM.size;
        GPUIntStream outputStream;
        printf("??!\n");
        // Run kernel
        inputStreamV.copy_to_device();
        inputStreamR.copy_to_device();
       // inputStreamV.print();
      //  inputStreamR.print();
        printf("??!\n");
        last_thrust(&inputStreamV, &inputStreamR, &outputStream, 0);
        //inputStreamR.print();
        for (int i = 0; i<5; i++){
            printf("%d \n",*(outputStream.host_timestamp+i));
        }
        outputStream.copy_to_host();
        //outputStream.print();
        // Compare kernel result to correct data
        std::vector<int> kernelTimestamps(outputStream.host_timestamp+*(outputStream.host_offset), outputStream.host_timestamp+outputStream.size);
        std::vector<int> kernelValues(outputStream.host_values+*(outputStream.host_offset), outputStream.host_values+outputStream.size);
        std::vector<int> correctTimestamps(CORRECT_STREAM.host_timestamp, CORRECT_STREAM.host_timestamp + CORRECT_STREAM.size);
        std::vector<int> correctValues(CORRECT_STREAM.host_values, CORRECT_STREAM.host_values + CORRECT_STREAM.size);

        REQUIRE(kernelTimestamps == correctTimestamps);
        REQUIRE(kernelValues == correctValues);
        //outputStream.print();
        // Cleanup
        inputStreamV.free_device();

        inputStreamR.free_device();
        outputStream.free_device();
        
        inputStreamV.free_host();
        outputStream.free_host();
        inputStreamR.free_host();
    }


SECTION("last() small random example") {
        // Read input and correct output data
        GPUReader inReader = GPUReader("../test/data/last_test1.in");
        GPUIntStream inputStreamV = inReader.getGPUIntStream("z");
        GPUUnitStream inputStreamR = inReader.getGPUUnitStream("a");
        GPUReader outReader = GPUReader("../test/data/last_test1.out");
        GPUIntStream CORRECT_STREAM = outReader.getGPUIntStream("y");

        // Prepare empty output stream to fill
        int size = CORRECT_STREAM.size;
        //printf("%d \n\n",  CORRECT_STREAM.size);

        GPUIntStream outputStream;

        // Run kernel
        inputStreamV.copy_to_device();
        inputStreamR.copy_to_device();
        //inputStreamV.print();
        //inputStreamR.print();
        last_thrust(&inputStreamV, &inputStreamR, &outputStream, 0);
        outputStream.copy_to_host();
        //printf("xx");
       // outputStream.print();
        //outputStream.print();
        // Compare kernel result to correct data
        std::vector<int> kernelTimestamps(outputStream.host_timestamp+*(outputStream.host_offset), outputStream.host_timestamp+outputStream.size);
        std::vector<int> kernelValues(outputStream.host_values+*(outputStream.host_offset), outputStream.host_values+outputStream.size);
        std::vector<int> correctTimestamps(CORRECT_STREAM.host_timestamp, CORRECT_STREAM.host_timestamp + CORRECT_STREAM.size);
        std::vector<int> correctValues(CORRECT_STREAM.host_values, CORRECT_STREAM.host_values + CORRECT_STREAM.size);

        REQUIRE(kernelTimestamps == correctTimestamps);
        REQUIRE(kernelValues == correctValues);

        // Cleanup
        inputStreamV.free_device();
        inputStreamR.free_device();
        outputStream.free_device();
    }

    SECTION("last() large random example") {
        // Read input and correct output data
        GPUReader inReader = GPUReader("../test/data/last_test2.in");
        GPUIntStream inputStreamV = inReader.getGPUIntStream("z");
        GPUUnitStream inputStreamR = inReader.getGPUUnitStream("a");
        GPUReader outReader = GPUReader("../test/data/last_test2.out");
        GPUIntStream CORRECT_STREAM = outReader.getGPUIntStream("y");
        // Prepare empty output stream to fill
        GPUIntStream outputStream;

        // Run kernel
        inputStreamV.copy_to_device();
        inputStreamR.copy_to_device();
        //inputStreamV.print();
        //inputStreamR.print();

        last_thrust(&inputStreamV, &inputStreamR, &outputStream, 0);
        outputStream.copy_to_host();

        //outputStream.print();

        // Compare kernel result to correct data
        std::vector<int> kernelTimestamps(outputStream.host_timestamp+*(outputStream.host_offset), outputStream.host_timestamp+outputStream.size);
        std::vector<int> kernelValues(outputStream.host_values+*(outputStream.host_offset), outputStream.host_values+outputStream.size);
        std::vector<int> correctTimestamps(CORRECT_STREAM.host_timestamp, CORRECT_STREAM.host_timestamp + CORRECT_STREAM.size);
        std::vector<int> correctValues(CORRECT_STREAM.host_values, CORRECT_STREAM.host_values + CORRECT_STREAM.size);
        for (int i = 0; i< CORRECT_STREAM.size; i++){
            REQUIRE(kernelTimestamps[i] == correctTimestamps[i]);
        }

        REQUIRE(kernelValues == correctValues);
        //outputStream.print();
        // Cleanup
        inputStreamV.free_device();
        inputStreamR.free_device();
        outputStream.free_device();
    }

    SECTION("last() twice test with no invalids") {
        // Read input and correct output data
        GPUReader inReader = GPUReader("../test/data/last_test3.in");
        GPUIntStream inputStreamV = inReader.getGPUIntStream("z");
        GPUUnitStream inputStream2 = inReader.getGPUUnitStream("a");
        GPUUnitStream inputStreamR = inReader.getGPUUnitStream("x");
        GPUReader outReader = GPUReader("../test/data/last_test3.out");
        GPUIntStream CORRECT_STREAM = outReader.getGPUIntStream("o");

        // Prepare empty output stream to fill
        int size = CORRECT_STREAM.size;

        GPUIntStream intermediateStream;
        GPUIntStream outputStream;
          
        // Run kernel
        inputStreamV.copy_to_device();
        inputStreamR.copy_to_device();
        inputStream2.copy_to_device();
        last_thrust(&inputStreamV, &inputStreamR, &intermediateStream, 0);
        intermediateStream.copy_to_host();
        last_thrust(&intermediateStream, &inputStream2, &outputStream, 0);

        outputStream.copy_to_host();
//        outputStream.print();

        // Compare kernel result to correct data
        std::vector<int> kernelTimestamps(outputStream.host_timestamp+*(outputStream.host_offset), outputStream.host_timestamp+outputStream.size);
        std::vector<int> kernelValues(outputStream.host_values+*(outputStream.host_offset), outputStream.host_values+outputStream.size);
        std::vector<int> correctTimestamps(CORRECT_STREAM.host_timestamp, CORRECT_STREAM.host_timestamp + CORRECT_STREAM.size);
        std::vector<int> correctValues(CORRECT_STREAM.host_values, CORRECT_STREAM.host_values + CORRECT_STREAM.size);
        REQUIRE(kernelTimestamps == correctTimestamps);


        REQUIRE(kernelValues == correctValues);
        // Cleanup
        inputStreamV.free_device();
        inputStreamR.free_device();
        intermediateStream.free_device();
        outputStream.free_device();
        inputStream2.free_device();
    }

    SECTION("last() twice test with invalids in Unit Stream") {
        //printf("-------------------------\n");
        // Read input and correct output data
        GPUReader inReader = GPUReader("../test/data/last_test4.in");
        GPUIntStream inputStreamV = inReader.getGPUIntStream("z");
        GPUUnitStream inputStream2 = inReader.getGPUUnitStream("x");
        GPUUnitStream inputStreamR = inReader.getGPUUnitStream("a");
        GPUReader outReader = GPUReader("../test/data/last_test4.out");
        GPUIntStream CORRECT_STREAM = outReader.getGPUIntStream("o");

        // Prepare empty output stream to fill
        int size = CORRECT_STREAM.size;
        int sizeAllocated = (size_t)  inputStreamR.size * sizeof(int);
        int *host_timestampOut = (int *) malloc( inputStreamR.size * sizeof(int));
        int *host_valueOut = (int*) malloc( inputStreamR.size * sizeof(int));

        memset(host_timestampOut, 0, sizeAllocated);
        memset(host_valueOut, 0, sizeAllocated);

        int *host_timestampOut2 = (int *) malloc( inputStream2.size * sizeof(int));
        int *host_valueOut2 = (int*) malloc( inputStream2.size * sizeof(int));

        memset(host_timestampOut2, 0, inputStream2.size * sizeof(int));
        memset(host_valueOut2, 0, inputStream2.size * sizeof(int));
        GPUIntStream intermediateStream(host_timestampOut, host_valueOut, inputStreamR.size);
        GPUIntStream outputStream(host_timestampOut2, host_valueOut2,inputStream2.size);
        // Run kernel
        intermediateStream.copy_to_device();
        inputStreamV.copy_to_device();
        inputStreamR.copy_to_device();
        outputStream.copy_to_device();
        inputStream2.copy_to_device();

        last_thrust(&inputStreamV, &inputStreamR, &intermediateStream, 0);
        last_thrust(&intermediateStream, &inputStream2, &outputStream, 0);

        outputStream.copy_to_host();
        outputStream.print();

        // Compare kernel result to correct data
        std::vector<int> kernelTimestamps(outputStream.host_timestamp+*(outputStream.host_offset), outputStream.host_timestamp+outputStream.size);
        std::vector<int> kernelValues(outputStream.host_values+*(outputStream.host_offset), outputStream.host_values+outputStream.size);
        std::vector<int> correctTimestamps(CORRECT_STREAM.host_timestamp, CORRECT_STREAM.host_timestamp + CORRECT_STREAM.size);
        std::vector<int> correctValues(CORRECT_STREAM.host_values, CORRECT_STREAM.host_values + CORRECT_STREAM.size);

        REQUIRE(kernelTimestamps == correctTimestamps);

        REQUIRE(kernelValues == correctValues);
        //outputStream.print();
        // Cleanup
        inputStreamV.free_device();
        inputStreamR.free_device();
        intermediateStream.free_device();
        outputStream.free_device();
        inputStream2.free_device();
        free(host_valueOut);
        free(host_timestampOut);
        free(host_valueOut2);
        free(host_timestampOut2);
    }
    SECTION("last empty"){
        //reading only invalid streams (they are empty)
        GPUReader inReader = GPUReader("../test/data/bt_last.in");
        GPUIntStream inputStreamV = inReader.getGPUIntStream("v2");
        GPUUnitStream inputStreamR = inReader.getGPUUnitStream("r2");
        GPUReader outReader = GPUReader("../test/data/bt_last.out");
        GPUIntStream CORRECT_STREAM = outReader.getGPUIntStream("y2");
        int size = CORRECT_STREAM.size;
        GPUIntStream outputStream;

        // Run kernel
        inputStreamV.copy_to_device();
        inputStreamR.copy_to_device();
       // inputStreamV.print();
      //  inputStreamR.print();
        last_thrust(&inputStreamV, &inputStreamR, &outputStream, 0);
        //inputStreamR.print();
        outputStream.copy_to_host();

        //outputStream.print();
        // Compare kernel result to correct data
        std::vector<int> kernelTimestamps(outputStream.host_timestamp+*(outputStream.host_offset), outputStream.host_timestamp+outputStream.size);
        std::vector<int> kernelValues(outputStream.host_values+*(outputStream.host_offset), outputStream.host_values+outputStream.size);
        std::vector<int> correctTimestamps(CORRECT_STREAM.host_timestamp, CORRECT_STREAM.host_timestamp + CORRECT_STREAM.size);
        std::vector<int> correctValues(CORRECT_STREAM.host_values, CORRECT_STREAM.host_values + CORRECT_STREAM.size);

        REQUIRE(kernelTimestamps == correctTimestamps);
        REQUIRE(kernelValues == correctValues);
        //outputStream.print();
        // Cleanup
        inputStreamV.free_device();
        inputStreamR.free_device();
        outputStream.free_device();
    }



}



TEST_CASE("last()"){
    
    SECTION("last empty"){
        //reading only invalid streams (they are empty)
        GPUReader inReader = GPUReader("../test/data/bt_last.in");
        GPUIntStream inputStreamV = inReader.getGPUIntStream("v2");
        GPUUnitStream inputStreamR = inReader.getGPUUnitStream("r2");
        GPUReader outReader = GPUReader("../test/data/bt_last.out");
        GPUIntStream CORRECT_STREAM = outReader.getGPUIntStream("y2");
        int size = CORRECT_STREAM.size;
        GPUIntStream outputStream;

        // Run kernel
        inputStreamV.copy_to_device();
        inputStreamR.copy_to_device();
       // inputStreamV.print();
      //  inputStreamR.print();
        last(&inputStreamV, &inputStreamR, &outputStream, 0);
        //inputStreamR.print();
        outputStream.copy_to_host();

        //outputStream.print();
        // Compare kernel result to correct data
        std::vector<int> kernelTimestamps(outputStream.host_timestamp+*(outputStream.host_offset), outputStream.host_timestamp+outputStream.size);
        std::vector<int> kernelValues(outputStream.host_values+*(outputStream.host_offset), outputStream.host_values+outputStream.size);
        std::vector<int> correctTimestamps(CORRECT_STREAM.host_timestamp, CORRECT_STREAM.host_timestamp + CORRECT_STREAM.size);
        std::vector<int> correctValues(CORRECT_STREAM.host_values, CORRECT_STREAM.host_values + CORRECT_STREAM.size);

        REQUIRE(kernelTimestamps == correctTimestamps);
        REQUIRE(kernelValues == correctValues);
        //outputStream.print();
        // Cleanup
        inputStreamV.free_device();
        inputStreamR.free_device();
        outputStream.free_device();
    }

    SECTION("last() tuwel example") {
        // Read input and correct output data
        
        GPUReader inReader = GPUReader("../test/data/bt_last.in");
        GPUIntStream inputStreamV = inReader.getGPUIntStream("v");
        GPUUnitStream inputStreamR = inReader.getGPUUnitStream("r");
        GPUReader outReader = GPUReader("../test/data/bt_last.out");
        GPUIntStream CORRECT_STREAM = outReader.getGPUIntStream("y");

        // Prepare empty output stream to fill
        int size = CORRECT_STREAM.size;
        GPUIntStream outputStream;

        // Run kernel
        inputStreamV.copy_to_device();
        inputStreamR.copy_to_device();
       // inputStreamV.print();
      //  inputStreamR.print();
        last(&inputStreamV, &inputStreamR, &outputStream, 0);
        //inputStreamR.print();
        outputStream.copy_to_host();

        //outputStream.print();
        // Compare kernel result to correct data
        std::vector<int> kernelTimestamps(outputStream.host_timestamp+*(outputStream.host_offset), outputStream.host_timestamp+outputStream.size);
        std::vector<int> kernelValues(outputStream.host_values+*(outputStream.host_offset), outputStream.host_values+outputStream.size);
        std::vector<int> correctTimestamps(CORRECT_STREAM.host_timestamp, CORRECT_STREAM.host_timestamp + CORRECT_STREAM.size);
        std::vector<int> correctValues(CORRECT_STREAM.host_values, CORRECT_STREAM.host_values + CORRECT_STREAM.size);

        REQUIRE(kernelTimestamps == correctTimestamps);
        REQUIRE(kernelValues == correctValues);
        //outputStream.print();
        // Cleanup
        inputStreamV.free_device();
        inputStreamR.free_device();
        outputStream.free_device();
    }

    SECTION("last() small random example") {
        // Read input and correct output data
        GPUReader inReader = GPUReader("../test/data/last_test1.in");
        GPUIntStream inputStreamV = inReader.getGPUIntStream("z");
        GPUUnitStream inputStreamR = inReader.getGPUUnitStream("a");
        GPUReader outReader = GPUReader("../test/data/last_test1.out");
        GPUIntStream CORRECT_STREAM = outReader.getGPUIntStream("y");

        // Prepare empty output stream to fill
        int size = CORRECT_STREAM.size;
        //printf("%d \n\n",  CORRECT_STREAM.size);

        GPUIntStream outputStream;

        // Run kernel
        inputStreamV.copy_to_device();
        inputStreamR.copy_to_device();
        //inputStreamV.print();
        //inputStreamR.print();
        last(&inputStreamV, &inputStreamR, &outputStream, 0);
        outputStream.copy_to_host();
        //printf("xx");
       // outputStream.print();
        //outputStream.print();
        // Compare kernel result to correct data
        std::vector<int> kernelTimestamps(outputStream.host_timestamp+*(outputStream.host_offset), outputStream.host_timestamp+outputStream.size);
        std::vector<int> kernelValues(outputStream.host_values+*(outputStream.host_offset), outputStream.host_values+outputStream.size);
        std::vector<int> correctTimestamps(CORRECT_STREAM.host_timestamp, CORRECT_STREAM.host_timestamp + CORRECT_STREAM.size);
        std::vector<int> correctValues(CORRECT_STREAM.host_values, CORRECT_STREAM.host_values + CORRECT_STREAM.size);

        REQUIRE(kernelTimestamps == correctTimestamps);
        REQUIRE(kernelValues == correctValues);

        // Cleanup
        inputStreamV.free_device();
        inputStreamR.free_device();
        outputStream.free_device();
    }

    SECTION("last() large random example") {
        // Read input and correct output data
        GPUReader inReader = GPUReader("../test/data/last_test2.in");
        GPUIntStream inputStreamV = inReader.getGPUIntStream("z");
        GPUUnitStream inputStreamR = inReader.getGPUUnitStream("a");
        GPUReader outReader = GPUReader("../test/data/last_test2.out");
        GPUIntStream CORRECT_STREAM = outReader.getGPUIntStream("y");
        // Prepare empty output stream to fill
        GPUIntStream outputStream;

        // Run kernel
        inputStreamV.copy_to_device();
        inputStreamR.copy_to_device();
        //inputStreamV.print();
        //inputStreamR.print();

        last(&inputStreamV, &inputStreamR, &outputStream, 0);
        outputStream.copy_to_host();

        //outputStream.print();

        // Compare kernel result to correct data
        std::vector<int> kernelTimestamps(outputStream.host_timestamp+*(outputStream.host_offset), outputStream.host_timestamp+outputStream.size);
        std::vector<int> kernelValues(outputStream.host_values+*(outputStream.host_offset), outputStream.host_values+outputStream.size);
        std::vector<int> correctTimestamps(CORRECT_STREAM.host_timestamp, CORRECT_STREAM.host_timestamp + CORRECT_STREAM.size);
        std::vector<int> correctValues(CORRECT_STREAM.host_values, CORRECT_STREAM.host_values + CORRECT_STREAM.size);
        for (int i = 0; i< CORRECT_STREAM.size; i++){
            REQUIRE(kernelTimestamps[i] == correctTimestamps[i]);
        }

        REQUIRE(kernelValues == correctValues);
        //outputStream.print();
        // Cleanup
        inputStreamV.free_device();
        inputStreamR.free_device();
        outputStream.free_device();
    }

    SECTION("last() twice test with no invalids") {
        // Read input and correct output data
        GPUReader inReader = GPUReader("../test/data/last_test3.in");
        GPUIntStream inputStreamV = inReader.getGPUIntStream("z");
        GPUUnitStream inputStream2 = inReader.getGPUUnitStream("a");
        GPUUnitStream inputStreamR = inReader.getGPUUnitStream("x");
        GPUReader outReader = GPUReader("../test/data/last_test3.out");
        GPUIntStream CORRECT_STREAM = outReader.getGPUIntStream("o");

        // Prepare empty output stream to fill
        int size = CORRECT_STREAM.size;

        GPUIntStream intermediateStream;
        GPUIntStream outputStream;
          
        // Run kernel
        inputStreamV.copy_to_device();
        inputStreamR.copy_to_device();
        inputStream2.copy_to_device();
        last(&inputStreamV, &inputStreamR, &intermediateStream, 0);
        intermediateStream.copy_to_host();
        last(&intermediateStream, &inputStream2, &outputStream, 0);

        outputStream.copy_to_host();
//        outputStream.print();

        // Compare kernel result to correct data
        std::vector<int> kernelTimestamps(outputStream.host_timestamp+*(outputStream.host_offset), outputStream.host_timestamp+outputStream.size);
        std::vector<int> kernelValues(outputStream.host_values+*(outputStream.host_offset), outputStream.host_values+outputStream.size);
        std::vector<int> correctTimestamps(CORRECT_STREAM.host_timestamp, CORRECT_STREAM.host_timestamp + CORRECT_STREAM.size);
        std::vector<int> correctValues(CORRECT_STREAM.host_values, CORRECT_STREAM.host_values + CORRECT_STREAM.size);
        REQUIRE(kernelTimestamps == correctTimestamps);


        REQUIRE(kernelValues == correctValues);
        // Cleanup
        inputStreamV.free_device();
        inputStreamR.free_device();
        intermediateStream.free_device();
        outputStream.free_device();
        inputStream2.free_device();
    }

    SECTION("last() twice test with invalids in Unit Stream") {
        //printf("-------------------------\n");
        // Read input and correct output data
        GPUReader inReader = GPUReader("../test/data/last_test4.in");
        GPUIntStream inputStreamV = inReader.getGPUIntStream("z");
        GPUUnitStream inputStream2 = inReader.getGPUUnitStream("x");
        GPUUnitStream inputStreamR = inReader.getGPUUnitStream("a");
        GPUReader outReader = GPUReader("../test/data/last_test4.out");
        GPUIntStream CORRECT_STREAM = outReader.getGPUIntStream("o");

        // Prepare empty output stream to fill
        int size = CORRECT_STREAM.size;
        int sizeAllocated = (size_t)  inputStreamR.size * sizeof(int);
        int *host_timestampOut = (int *) malloc( inputStreamR.size * sizeof(int));
        int *host_valueOut = (int*) malloc( inputStreamR.size * sizeof(int));

        memset(host_timestampOut, 0, sizeAllocated);
        memset(host_valueOut, 0, sizeAllocated);

        int *host_timestampOut2 = (int *) malloc( inputStream2.size * sizeof(int));
        int *host_valueOut2 = (int*) malloc( inputStream2.size * sizeof(int));

        memset(host_timestampOut2, 0, inputStream2.size * sizeof(int));
        memset(host_valueOut2, 0, inputStream2.size * sizeof(int));
        GPUIntStream intermediateStream(host_timestampOut, host_valueOut, inputStreamR.size);
        GPUIntStream outputStream(host_timestampOut2, host_valueOut2,inputStream2.size);
        // Run kernel
        intermediateStream.copy_to_device();
        inputStreamV.copy_to_device();
        inputStreamR.copy_to_device();
        outputStream.copy_to_device();
        inputStream2.copy_to_device();

        last(&inputStreamV, &inputStreamR, &intermediateStream, 0);
        last(&intermediateStream, &inputStream2, &outputStream, 0);

        outputStream.copy_to_host();
        //outputStream.print();

        // Compare kernel result to correct data
        std::vector<int> kernelTimestamps(outputStream.host_timestamp+*(outputStream.host_offset), outputStream.host_timestamp+outputStream.size);
        std::vector<int> kernelValues(outputStream.host_values+*(outputStream.host_offset), outputStream.host_values+outputStream.size);
        std::vector<int> correctTimestamps(CORRECT_STREAM.host_timestamp, CORRECT_STREAM.host_timestamp + CORRECT_STREAM.size);
        std::vector<int> correctValues(CORRECT_STREAM.host_values, CORRECT_STREAM.host_values + CORRECT_STREAM.size);

        REQUIRE(kernelTimestamps == correctTimestamps);

        REQUIRE(kernelValues == correctValues);
        //outputStream.print();
        // Cleanup
        inputStreamV.free_device();
        inputStreamR.free_device();
        intermediateStream.free_device();
        outputStream.free_device();
        inputStream2.free_device();
        free(host_valueOut);
        free(host_timestampOut);
        free(host_valueOut2);
        free(host_timestampOut2);
    }
    
    

}
#define BENCHMARKING_CASES 5
#define BENCHMARKING_LOOPS 1

TEST_CASE("BENCHMARKING"){
    SECTION("last() benchmarking example"){
        //int BENCHMARKING_CASES = 6;
        std::ofstream output_last;
        //delete previous
        output_last.open("benchmarking_last.data");
        output_last << "";
        output_last.close();
        int dev = 0;
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        //might wanna derive MAX_THREADS and so on from here! TODO!
        printf("Using Device %d: %s\n", dev, deviceProp.name);
        for (int j=1;j <=BENCHMARKING_LOOPS;j++){
            
            cudaDeviceSynchronize();
            
            for (int i = 1;i<=BENCHMARKING_CASES; i++){
                auto start2 = std::chrono::high_resolution_clock::now();
                std::string path = "../test/data/benchmarking";
                GPUReader inReader = GPUReader(path + std::to_string(i) + ".in");
                GPUIntStream inputStreamV = inReader.getGPUIntStream("z");
                GPUUnitStream inputStreamR = inReader.getGPUUnitStream("a");
                GPUReader outReader = GPUReader(path + std::to_string(i) + "_last.out");
                GPUIntStream CORRECT_STREAM = outReader.getGPUIntStream("y");
                /*GPUReader inReader = GPUReader("../test/data/bt_last.in");
                GPUIntStream inputStreamV = inReader.getGPUIntStream("v");
                GPUUnitStream inputStreamR = inReader.getGPUUnitStream("r");
                GPUReader outReader = GPUReader("../test/data/bt_last.out");
                GPUIntStream CORRECT_STREAM = outReader.getGPUIntStream("y");*/
                //now start timer
                
                // Prepare empty output stream to fill
                int size = CORRECT_STREAM.size;
                auto start = std::chrono::high_resolution_clock::now();
                GPUIntStream outputStream;

                // Run kernel
                inputStreamV.copy_to_device();
                inputStreamR.copy_to_device();
                
                // inputStreamV.print();
                //  inputStreamR.print();
                last(&inputStreamV, &inputStreamR, &outputStream, 0);
                //inputStreamR.print();
                outputStream.copy_to_host();
                cudaDeviceSynchronize();

                auto stop = std::chrono::high_resolution_clock::now();
                //outputStream.print();
                // Compare kernel result to correct data
                std::vector<int> kernelTimestamps(outputStream.host_timestamp+*(outputStream.host_offset), outputStream.host_timestamp+outputStream.size);

                std::vector<int> kernelValues(outputStream.host_values+*(outputStream.host_offset), outputStream.host_values+outputStream.size);
                std::vector<int> correctTimestamps(CORRECT_STREAM.host_timestamp, CORRECT_STREAM.host_timestamp + CORRECT_STREAM.size);
                std::vector<int> correctValues(CORRECT_STREAM.host_values, CORRECT_STREAM.host_values + CORRECT_STREAM.size);



                REQUIRE(kernelTimestamps == correctTimestamps);
                REQUIRE(kernelValues == correctValues);
                //outputStream.print();
                // Cleanup
                inputStreamV.free_device();
                inputStreamR.free_device();
                outputStream.free_device();
                inputStreamV.free_host();
                inputStreamR.free_host();
                outputStream.free_host();
                
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                printf("%li us\n",duration.count());
                auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(stop - start2);
                output_last.open ("benchmarking_last.data",std::ios::app);
                output_last <<"Benchmark "<< i <<": "<<duration.count() <<  " us" << " with reader: " <<duration2.count() <<" us size: "<<size <<"\n";
                output_last.close();
            }
        }
    }



    SECTION("time() benchmarking example"){
        std::ofstream output_time;
        //delete previous
        printf("---time---- benchmark\n");
        output_time.open("benchmarking_time.data");
        output_time << "";
        output_time.close();
        int dev = 0;
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        //might wanna derive MAX_THREADS and so on from here! TODO!

        for (int j=1;j <=BENCHMARKING_LOOPS;j++){
            //cudaDeviceSynchronize();
            cudaDeviceSynchronize();
            for (int i = 1;i<=BENCHMARKING_CASES; i++){
                auto start2 = std::chrono::high_resolution_clock::now();
                std::string path = "../test/data/benchmarking";

                // Prepare empty output stream to fill
                GPUReader inReader = GPUReader(path + std::to_string(i) + ".in");
                GPUIntStream inputStream = inReader.getGPUIntStream("z");
                GPUReader outReader = GPUReader(path + std::to_string(i) + "_time.out");
                GPUIntStream CORRECT_STREAM = outReader.getGPUIntStream("y");

                // Prepare empty output stream to fill
                int size = CORRECT_STREAM.size;

                GPUIntStream outputStream; //(host_timestampOut, host_valueOut, size);

                // Run kernel
                 auto start = std::chrono::high_resolution_clock::now();
                inputStream.copy_to_device();
                //outputStream.copy_to_device();
                time(&inputStream, &outputStream, 0);
                outputStream.copy_to_host();
                cudaDeviceSynchronize();
                auto stop = std::chrono::high_resolution_clock::now();

                // Compare kernel result to correct data
                std::vector<int> kernelTimestamps(outputStream.host_timestamp, outputStream.host_timestamp + sizeof(outputStream.host_timestamp) / sizeof(int));
                std::vector<int> kernelValues(outputStream.host_values, outputStream.host_values + sizeof(outputStream.host_values) / sizeof(int));
                std::vector<int> correctTimestamps(CORRECT_STREAM.host_timestamp, CORRECT_STREAM.host_timestamp + sizeof(CORRECT_STREAM.host_timestamp) / sizeof(int));
                std::vector<int> correctValues(CORRECT_STREAM.host_values, CORRECT_STREAM.host_values + sizeof(CORRECT_STREAM.host_values) / sizeof(int));
                REQUIRE(kernelTimestamps == correctTimestamps);
                REQUIRE(kernelValues == correctValues);

                // Cleanup
                inputStream.free_device();
                outputStream.free_device();
                inputStream.free_host();
                outputStream.free_host();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(stop - start2);
                
                output_time.open ("benchmarking_time.data",std::ios::app);
                output_time <<"Benchmark "<< i <<": "<<duration.count() <<  " us" << " with reader: " <<duration2.count() <<" us size: "<<size <<"\n";
                output_time.close();
            }
    }
    }


    SECTION("delay() benchmarking example"){
        std::ofstream output_delay;
        //delete previous
        output_delay.open("benchmarking_delay.data");
        output_delay << "";
        output_delay.close();
        int dev = 0;
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        //might wanna derive MAX_THREADS and so on from here! TODO!

        for (int j=1;j <=BENCHMARKING_LOOPS;j++){
            //cudaDeviceSynchronize();
            cudaDeviceSynchronize();
            for (int i = 3;i<=BENCHMARKING_CASES; i++){
                auto start2 = std::chrono::high_resolution_clock::now();
                std::string path = "../test/data/benchmarking";

                // Prepare empty output stream to fill

                GPUReader inReader = GPUReader(path + std::to_string(i) + ".in");
                GPUIntStream inputStreamD = inReader.getGPUIntStream("z");
                GPUUnitStream inputStreamR = inReader.getGPUUnitStream("a");
                GPUReader outReader = GPUReader(path + std::to_string(i) + "_delay.out");
                GPUUnitStream CORRECT_STREAM = outReader.getGPUUnitStream("y");

                // Prepare empty output stream to fill
                int size = inputStreamR.size;
                int sizeAllocated = (size_t) size * sizeof(int);
                int *host_timestampOut = (int *) malloc(size * sizeof(int));
                memset(host_timestampOut, -1, sizeAllocated);
                GPUUnitStream outputStream(host_timestampOut, size);
                auto start = std::chrono::high_resolution_clock::now();
                // Run kernel
                inputStreamD.copy_to_device();
                inputStreamR.copy_to_device();
                outputStream.copy_to_device();
                delay(&inputStreamD, &inputStreamR, &outputStream, 0);
                outputStream.copy_to_host();
                cudaDeviceSynchronize();

                auto stop = std::chrono::high_resolution_clock::now();

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

                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(stop - start2);
                
                output_delay.open ("benchmarking_delay.data",std::ios::app);
                output_delay <<"Benchmark "<< i <<": "<<duration.count() <<  " us" << " with reader: " <<duration2.count() <<" us size: "<<size <<"\n";
                output_delay.close();
            }
    }
    }


/*
    SECTION("lift() benchmarking"){
        std::ofstream output_delay;
        //delete previous
        output_delay.open("benchmarking_lift.data");
        output_delay << "";
        output_delay.close();
        int dev = 0;
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        //might wanna derive MAX_THREADS and so on from here! TODO!

        for (int j=1;j <=BENCHMARKING_LOOPS;j++){
            //cudaDeviceSynchronize();
            cudaDeviceSynchronize();
            for (int i = 3;i<=BENCHMARKING_CASES; i++){
                auto start2 = std::chrono::high_resolution_clock::now();
                std::string path = "../test/data/benchmarking";

                // Prepare empty output stream to fill

                GPUReader inReader = GPUReader(path+std::to_string(i)+".in");
                GPUIntStream inputStreamD = inReader.getGPUIntStream("z");
                GPUUnitStream inputStreamR = inReader.getGPUUnitStream("a");
                GPUReader outReader = GPUReader(path+std::to_string(i)+"_lift.out");
                GPUUnitStream CORRECT_STREAM = outReader.getGPUUnitStream("y");

                // Prepare empty output stream to fill
                int size = inputStreamR.size;
                int sizeAllocated = (size_t) size * sizeof(int);
                int *host_timestampOut = (int *) malloc(size * sizeof(int));
                memset(host_timestampOut, -1, sizeAllocated);
                GPUUnitStream outputStream(host_timestampOut, size);
                auto start = std::chrono::high_resolution_clock::now();
                // Run kernel
                inputStreamD.copy_to_device();
                inputStreamR.copy_to_device();
                outputStream.copy_to_device();
                delay(&inputStreamD, &inputStreamR, &outputStream, 0);
                outputStream.copy_to_host();
                cudaDeviceSynchronize();

                auto stop = std::chrono::high_resolution_clock::now();

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

                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(stop - start2);
                
                output_delay.open ("benchmarking_delay.data",std::ios::app);
                output_delay <<"Benchmark "<< i <<": "<<duration.count() <<  " us" << " with reader: " <<duration2.count() <<" us size: "<<size <<"\n";
                output_delay.close();
            }
    }
    }*/
}

TEST_CASE("extensive stream ops"){
    SECTION("last|time"){
        // Read input and correct output data
        GPUReader inReader = GPUReader("../test/data/extensive_benchmark.in");
        GPUIntStream inputStreamZ = inReader.getGPUIntStream("z");
        GPUUnitStream inputStreamA = inReader.getGPUUnitStream("a");
        GPUUnitStream inputStreamB = inReader.getGPUUnitStream("b");
        GPUReader outReader = GPUReader("../test/data/extensive_benchmark.out");
        GPUIntStream CORRECT_STREAM = outReader.getGPUIntStream("y5");

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
        GPUIntStream inputStreamZ = inReader.getGPUIntStream("z");
        GPUUnitStream inputStreamA = inReader.getGPUUnitStream("a");
        GPUUnitStream inputStreamB = inReader.getGPUUnitStream("b");
        GPUReader outReader = GPUReader("../test/data/extensive_benchmark2.out");
        GPUIntStream CORRECT_STREAM1 = outReader.getGPUIntStream("y1");
        GPUIntStream CORRECT_STREAM2 = outReader.getGPUIntStream("y2");
        //TODO! crate nil streams in output file!
        //otherwise comment out the below
        GPUUnitStream CORRECT_STREAM3 = outReader.getGPUUnitStream("y3");
        GPUUnitStream CORRECT_STREAM4 = outReader.getGPUUnitStream("y4");
        GPUIntStream CORRECT_STREAM5 = outReader.getGPUIntStream("y5");
        GPUUnitStream CORRECT_STREAM6 = outReader.getGPUUnitStream("y6");
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
        */
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
            GPUIntStream inputStreamD = inReader.getGPUIntStream("d");
            GPUUnitStream inputStreamR = inReader.getGPUUnitStream("r");
            GPUReader outReader = GPUReader("../test/data/bt_delay.out");
            GPUUnitStream CORRECT_STREAM = outReader.getGPUUnitStream("y");

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
            GPUIntStream inputStreamD = inReader.getGPUIntStream("z");
            GPUUnitStream inputStreamR = inReader.getGPUUnitStream("a");
            GPUReader outReader = GPUReader("../test/data/bt_delay.middle.out");
            GPUUnitStream CORRECT_STREAM = outReader.getGPUUnitStream("y");

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
            GPUIntStream inputStreamD = inReader.getGPUIntStream("z");
            GPUUnitStream inputStreamR = inReader.getGPUUnitStream("a");
            GPUReader outReader = GPUReader("../test/data/bt_delay.bigger.out");
            GPUUnitStream CORRECT_STREAM = outReader.getGPUUnitStream("y");

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
        GPUIntStream inputStreamX = inReader.getGPUIntStream("x");
        GPUIntStream inputStreamY = inReader.getGPUIntStream("y");
        GPUReader outReader = GPUReader("../test/data/bt_merge.out");
        GPUIntStream CORRECT_STREAM = outReader.getGPUIntStream("z");

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

    SECTION("time()") {
        SECTION("time() with small dataset") {
            // Read input and correct output data
            GPUReader inReader = GPUReader("../test/data/bt_time.in");
            std::shared_ptr<GPUIntStream> inputStream = inReader.getGPUIntStream("x");
            GPUReader outReader = GPUReader("../test/data/bt_time.out");
            std::shared_ptr<GPUIntStream> CORRECT_STREAM = outReader.getGPUIntStream("x");

            // Prepare empty output stream to fill
            int size = CORRECT_STREAM.size;

            GPUIntStream outputStream; //(host_timestampOut, host_valueOut, size);

            // Run kernel
            inputStream.copy_to_device();
            //outputStream.copy_to_device();
            time(&inputStream, &outputStream, 0);
            outputStream.copy_to_host();

            // Compare kernel result to correct data
            std::vector<int> kernelTimestamps(outputStream.host_timestamp, outputStream.host_timestamp + sizeof(outputStream.host_timestamp) / sizeof(int));
            std::vector<int> kernelValues(outputStream.host_values, outputStream.host_values + sizeof(outputStream.host_values) / sizeof(int));
            std::vector<int> correctTimestamps(CORRECT_STREAM.host_timestamp, CORRECT_STREAM.host_timestamp + sizeof(CORRECT_STREAM.host_timestamp) / sizeof(int));
            std::vector<int> correctValues(CORRECT_STREAM.host_values, CORRECT_STREAM.host_values + sizeof(CORRECT_STREAM.host_values) / sizeof(int));
            REQUIRE(kernelTimestamps == correctTimestamps);
            REQUIRE(kernelValues == correctValues);

            // Cleanup
            inputStream.free_device();
            outputStream.free_device();
        }
        
        SECTION("time() with bigger dataset (~109k/250k events)") {
            // Read input and correct output data
            GPUReader inReader = GPUReader("../test/data/bt_time.bigger.in");
            GPUIntStream inputStream = inReader.getGPUIntStream("z");
            GPUReader outReader = GPUReader("../test/data/bt_time.bigger.out");
            GPUIntStream CORRECT_STREAM = outReader.getGPUIntStream("y");

            // Prepare empty output stream to fill
            int size = CORRECT_STREAM.size;

            GPUIntStream outputStream;

            // Run kernel
            inputStream.copy_to_device();
            outputStream.copy_to_device();
            time(&inputStream, &outputStream, 0);
            outputStream.copy_to_host();

            // Compare kernel result to correct data
            std::vector<int> kernelTimestamps(outputStream.host_timestamp, outputStream.host_timestamp + sizeof(outputStream.host_timestamp) / sizeof(int));
            std::vector<int> kernelValues(outputStream.host_values, outputStream.host_values + sizeof(outputStream.host_values) / sizeof(int));
            std::vector<int> correctTimestamps(CORRECT_STREAM.host_timestamp, CORRECT_STREAM.host_timestamp + sizeof(CORRECT_STREAM.host_timestamp) / sizeof(int));
            std::vector<int> correctValues(CORRECT_STREAM.host_values, CORRECT_STREAM.host_values + sizeof(CORRECT_STREAM.host_values) / sizeof(int));
            REQUIRE(kernelTimestamps == correctTimestamps);
            REQUIRE(kernelValues == correctValues);

            // Cleanup
            inputStream.free_device();
            outputStream.free_device();
        }
    }
    
}

TEST_CASE("GPUReader (bt_delay.in)") {
    GPUReader reader = GPUReader("../test/data/bt_delay.in");

    SECTION("Read existing GPUIntStream correctly") {
        int CORRECT_STREAM_SIZE = 6;
        int CORRECT_STREAM_TIMESTAMPS[CORRECT_STREAM_SIZE] = {1, 2, 3, 4, 6, 7};
        int CORRECT_STREAM_INTS[CORRECT_STREAM_SIZE] = {1, 2, 4, 1, 2, 3};

        GPUIntStream dStream = reader.getGPUIntStream("d");
        REQUIRE(dStream.size == CORRECT_STREAM_SIZE);
        for (int i = 0; i < CORRECT_STREAM_SIZE; i++) {
            REQUIRE(dStream.host_timestamp[i] == CORRECT_STREAM_TIMESTAMPS[i]);
            REQUIRE(dStream.host_values[i] == CORRECT_STREAM_INTS[i]);
        }
    }

    SECTION("Read existing GPUUnitStream correctly") {
        int CORRECT_STREAM_SIZE = 3;
        int CORRECT_STREAM_TIMESTAMPS[CORRECT_STREAM_SIZE] = {2, 7, 8};

        GPUUnitStream dStream = reader.getGPUUnitStream("r");
        REQUIRE(dStream.size == CORRECT_STREAM_SIZE);
        for (int i = 0; i < CORRECT_STREAM_SIZE; i++) {
            REQUIRE(dStream.host_timestamp[i] == CORRECT_STREAM_TIMESTAMPS[i]);
        }
    }

}
