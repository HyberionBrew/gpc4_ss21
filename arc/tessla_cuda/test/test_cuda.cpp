
#include <iostream>
#include <string>
#include <string.h>
#include <vector>
#include "../../test/catch2/catch.hpp"
#include "../src/Reader.cuh"
#include "../src/Stream.cuh"
#include "../src/StreamFunctions.cuh"

#include <chrono>
#include <fstream>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <cuda_profiler_api.h>

#include "../src/StreamFunctionsThrust.cuh"

TEST_CASE("last_thrust()"){

    SECTION("last() tuwel example") {
        // Read input and correct output data
        
        Reader inReader = Reader("../test/data/bt_last.in");
        IntStream inputStreamV = inReader.getIntStream("v");
        UnitStream inputStreamR = inReader.getUnitStream("r");
        Reader outReader = Reader("../test/data/bt_last.out");
        IntStream CORRECT_STREAM = outReader.getIntStream("y");

        // Prepare empty output stream to fill
        int size = CORRECT_STREAM.size;
        IntStream outputStream;
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
        Reader inReader = Reader("../test/data/last_test1.in");
        IntStream inputStreamV = inReader.getIntStream("z");
        UnitStream inputStreamR = inReader.getUnitStream("a");
        Reader outReader = Reader("../test/data/last_test1.out");
        IntStream CORRECT_STREAM = outReader.getIntStream("y");

        // Prepare empty output stream to fill
        int size = CORRECT_STREAM.size;
        //printf("%d \n\n",  CORRECT_STREAM.size);

        IntStream outputStream;

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
        Reader inReader = Reader("../test/data/last_test2.in");
        IntStream inputStreamV = inReader.getIntStream("z");
        UnitStream inputStreamR = inReader.getUnitStream("a");
        Reader outReader = Reader("../test/data/last_test2.out");
        IntStream CORRECT_STREAM = outReader.getIntStream("y");
        // Prepare empty output stream to fill
        IntStream outputStream;

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
        Reader inReader = Reader("../test/data/last_test3.in");
        IntStream inputStreamV = inReader.getIntStream("z");
        UnitStream inputStream2 = inReader.getUnitStream("a");
        UnitStream inputStreamR = inReader.getUnitStream("x");
        Reader outReader = Reader("../test/data/last_test3.out");
        IntStream CORRECT_STREAM = outReader.getIntStream("o");

        // Prepare empty output stream to fill
        int size = CORRECT_STREAM.size;

        IntStream intermediateStream;
        IntStream outputStream;
          
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
        Reader inReader = Reader("../test/data/last_test4.in");
        IntStream inputStreamV = inReader.getIntStream("z");
        UnitStream inputStream2 = inReader.getUnitStream("x");
        UnitStream inputStreamR = inReader.getUnitStream("a");
        Reader outReader = Reader("../test/data/last_test4.out");
        IntStream CORRECT_STREAM = outReader.getIntStream("o");

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
        IntStream intermediateStream(host_timestampOut, host_valueOut, inputStreamR.size);
        IntStream outputStream(host_timestampOut2, host_valueOut2,inputStream2.size);
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




}



TEST_CASE("last()"){

    SECTION("last() tuwel example") {
        // Read input and correct output data
        
        Reader inReader = Reader("../test/data/bt_last.in");
        IntStream inputStreamV = inReader.getIntStream("v");
        UnitStream inputStreamR = inReader.getUnitStream("r");
        Reader outReader = Reader("../test/data/bt_last.out");
        IntStream CORRECT_STREAM = outReader.getIntStream("y");

        // Prepare empty output stream to fill
        int size = CORRECT_STREAM.size;
        IntStream outputStream;

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
        Reader inReader = Reader("../test/data/last_test1.in");
        IntStream inputStreamV = inReader.getIntStream("z");
        UnitStream inputStreamR = inReader.getUnitStream("a");
        Reader outReader = Reader("../test/data/last_test1.out");
        IntStream CORRECT_STREAM = outReader.getIntStream("y");

        // Prepare empty output stream to fill
        int size = CORRECT_STREAM.size;
        //printf("%d \n\n",  CORRECT_STREAM.size);

        IntStream outputStream;

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
        Reader inReader = Reader("../test/data/last_test2.in");
        IntStream inputStreamV = inReader.getIntStream("z");
        UnitStream inputStreamR = inReader.getUnitStream("a");
        Reader outReader = Reader("../test/data/last_test2.out");
        IntStream CORRECT_STREAM = outReader.getIntStream("y");
        // Prepare empty output stream to fill
        IntStream outputStream;

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
        Reader inReader = Reader("../test/data/last_test3.in");
        IntStream inputStreamV = inReader.getIntStream("z");
        UnitStream inputStream2 = inReader.getUnitStream("a");
        UnitStream inputStreamR = inReader.getUnitStream("x");
        Reader outReader = Reader("../test/data/last_test3.out");
        IntStream CORRECT_STREAM = outReader.getIntStream("o");

        // Prepare empty output stream to fill
        int size = CORRECT_STREAM.size;

        IntStream intermediateStream;
        IntStream outputStream;
          
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
        Reader inReader = Reader("../test/data/last_test4.in");
        IntStream inputStreamV = inReader.getIntStream("z");
        UnitStream inputStream2 = inReader.getUnitStream("x");
        UnitStream inputStreamR = inReader.getUnitStream("a");
        Reader outReader = Reader("../test/data/last_test4.out");
        IntStream CORRECT_STREAM = outReader.getIntStream("o");

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
        IntStream intermediateStream(host_timestampOut, host_valueOut, inputStreamR.size);
        IntStream outputStream(host_timestampOut2, host_valueOut2,inputStream2.size);
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
                Reader inReader = Reader(path+std::to_string(i)+".in");
                IntStream inputStreamV = inReader.getIntStream("z");
                UnitStream inputStreamR = inReader.getUnitStream("a");
                Reader outReader = Reader(path+std::to_string(i)+"_last.out");
                IntStream CORRECT_STREAM = outReader.getIntStream("y");
                /*Reader inReader = Reader("../test/data/bt_last.in");
                IntStream inputStreamV = inReader.getIntStream("v");
                UnitStream inputStreamR = inReader.getUnitStream("r");
                Reader outReader = Reader("../test/data/bt_last.out");
                IntStream CORRECT_STREAM = outReader.getIntStream("y");*/
                //now start timer
                
                // Prepare empty output stream to fill
                int size = CORRECT_STREAM.size;
                auto start = std::chrono::high_resolution_clock::now();
                IntStream outputStream;

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
                Reader inReader = Reader(path+std::to_string(i)+".in");
                IntStream inputStream = inReader.getIntStream("z");
                Reader outReader = Reader(path+std::to_string(i)+"_time.out");
                IntStream CORRECT_STREAM = outReader.getIntStream("y");

                // Prepare empty output stream to fill
                int size = CORRECT_STREAM.size;

                IntStream outputStream; //(host_timestampOut, host_valueOut, size);

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

                Reader inReader = Reader(path+std::to_string(i)+".in");
                IntStream inputStreamD = inReader.getIntStream("z");
                UnitStream inputStreamR = inReader.getUnitStream("a");
                Reader outReader = Reader(path+std::to_string(i)+"_delay.out");
                UnitStream CORRECT_STREAM = outReader.getUnitStream("y");

                // Prepare empty output stream to fill
                int size = inputStreamR.size;
                int sizeAllocated = (size_t) size * sizeof(int);
                int *host_timestampOut = (int *) malloc(size * sizeof(int));
                memset(host_timestampOut, -1, sizeAllocated);
                UnitStream outputStream(host_timestampOut, size);
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

                Reader inReader = Reader(path+std::to_string(i)+".in");
                IntStream inputStreamD = inReader.getIntStream("z");
                UnitStream inputStreamR = inReader.getUnitStream("a");
                Reader outReader = Reader(path+std::to_string(i)+"_lift.out");
                UnitStream CORRECT_STREAM = outReader.getUnitStream("y");

                // Prepare empty output stream to fill
                int size = inputStreamR.size;
                int sizeAllocated = (size_t) size * sizeof(int);
                int *host_timestampOut = (int *) malloc(size * sizeof(int));
                memset(host_timestampOut, -1, sizeAllocated);
                UnitStream outputStream(host_timestampOut, size);
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
        Reader inReader = Reader("../test/data/extensive_benchmark.in");
        IntStream inputStreamZ = inReader.getIntStream("z");
        UnitStream inputStreamA = inReader.getUnitStream("a");
        UnitStream inputStreamB = inReader.getUnitStream("b");
        Reader outReader = Reader("../test/data/extensive_benchmark.out");
        IntStream CORRECT_STREAM = outReader.getIntStream("y5");

        // Prepare empty output stream to fill
        int size = CORRECT_STREAM.size;
        IntStream y1,y2,y3,y4,y5;

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
        printf("-------------");
        Reader inReader = Reader("../test/data/extensive_benchmark.in");
        IntStream inputStreamZ = inReader.getIntStream("z");
        UnitStream inputStreamA = inReader.getUnitStream("a");
        UnitStream inputStreamB = inReader.getUnitStream("b");
        Reader outReader = Reader("../test/data/extensive_benchmark2.out");
        IntStream CORRECT_STREAM1 = outReader.getIntStream("y1");
        IntStream CORRECT_STREAM2 = outReader.getIntStream("y2");
        //TODO! crate nil streams in output file!
        //otherwise comment out the below
        UnitStream CORRECT_STREAM3 = outReader.getUnitStream("y3");
        UnitStream CORRECT_STREAM4 = outReader.getUnitStream("y4");
        IntStream CORRECT_STREAM5 = outReader.getIntStream("y5");
        UnitStream CORRECT_STREAM6 = outReader.getUnitStream("y6");
        // Prepare empty output stream to fill
        //int size = CORRECT_STREAM.size;
        IntStream y1,y2,y5;
        UnitStream y3,y4,y6;
        // Run kernel
        inputStreamZ.copy_to_device();
        inputStreamA.copy_to_device();
        inputStreamB.copy_to_device();
       // inputStreamV.print();
      //  inputStreamR.print();
        //printf("before delay \n");
        last(&inputStreamZ, &inputStreamA, &y1, 0);
        time(&y1,&y2,0);
        //printf("before delay \n");
        delay(&y2,&inputStreamA,&y3,0);
        delay(&inputStreamZ,&y3,&y4,0);
        last(&y2, &y4, &y5, 0);
        delay(&y5,&y4,&y6,0);
        //inputStreamR.print();
        y1.copy_to_host();
        y2.copy_to_host();
        y6.copy_to_host();
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
            Reader inReader = Reader("../test/data/bt_delay.in");
            IntStream inputStreamD = inReader.getIntStream("d");
            UnitStream inputStreamR = inReader.getUnitStream("r");
            Reader outReader = Reader("../test/data/bt_delay.out");
            UnitStream CORRECT_STREAM = outReader.getUnitStream("y");

            // Prepare empty output stream to fill
            int size = inputStreamR.size;
            int sizeAllocated = (size_t) size * sizeof(int);
            int *host_timestampOut = (int *) malloc(size * sizeof(int));
            memset(host_timestampOut, -1, sizeAllocated);
            UnitStream outputStream(host_timestampOut, size);

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
            Reader inReader = Reader("../test/data/bt_delay.middle.in");
            IntStream inputStreamD = inReader.getIntStream("z");
            UnitStream inputStreamR = inReader.getUnitStream("a");
            Reader outReader = Reader("../test/data/bt_delay.middle.out");
            UnitStream CORRECT_STREAM = outReader.getUnitStream("y");

            // Prepare empty output stream to fill
            int size = CORRECT_STREAM.size;
            int sizeAllocated = (size_t) size * sizeof(int);
            int *host_timestampOut = (int *) malloc(size * sizeof(int));
            memset(host_timestampOut, 0, sizeAllocated);
            UnitStream outputStream(host_timestampOut, size);

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
            Reader inReader = Reader("../test/data/bt_delay.bigger.in");
            IntStream inputStreamD = inReader.getIntStream("z");
            UnitStream inputStreamR = inReader.getUnitStream("a");
            Reader outReader = Reader("../test/data/bt_delay.bigger.out");
            UnitStream CORRECT_STREAM = outReader.getUnitStream("y");

            // Prepare empty output stream to fill
            int size = CORRECT_STREAM.size;
            int sizeAllocated = (size_t) size * sizeof(int);
            int *host_timestampOut = (int *) malloc(size * sizeof(int));
            memset(host_timestampOut, 0, sizeAllocated);
            UnitStream outputStream(host_timestampOut, size);

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
        Reader inReader = Reader("../test/data/bt_merge.in");
        IntStream inputStreamX = inReader.getIntStream("x");
        IntStream inputStreamY = inReader.getIntStream("y");
        Reader outReader = Reader("../test/data/bt_merge.out");
        IntStream CORRECT_STREAM = outReader.getIntStream("z");

        // Prepare empty output stream to fill
        int size = CORRECT_STREAM.size;
        int sizeAllocated = (size_t) size * sizeof(int);
        int *host_timestampOut = (int *) malloc(size * sizeof(int));
        int *host_valueOut = (int*) malloc(size * sizeof(int));
        memset(host_timestampOut, 0, sizeAllocated);
        memset(host_valueOut, 0, sizeAllocated);
        IntStream outputStream(host_timestampOut, host_valueOut, size);

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
            Reader inReader = Reader("../test/data/bt_time.in");
            IntStream inputStream = inReader.getIntStream("x");
            Reader outReader = Reader("../test/data/bt_time.out");
            IntStream CORRECT_STREAM = outReader.getIntStream("x");

            // Prepare empty output stream to fill
            int size = CORRECT_STREAM.size;

            IntStream outputStream; //(host_timestampOut, host_valueOut, size);

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
            Reader inReader = Reader("../test/data/bt_time.bigger.in");
            IntStream inputStream = inReader.getIntStream("z");
            Reader outReader = Reader("../test/data/bt_time.bigger.out");
            IntStream CORRECT_STREAM = outReader.getIntStream("y");

            // Prepare empty output stream to fill
            int size = CORRECT_STREAM.size;

            IntStream outputStream;

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
