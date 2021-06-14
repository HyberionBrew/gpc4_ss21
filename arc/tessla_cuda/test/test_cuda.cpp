
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

TEST_CASE("last()"){
    /**/
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
#define BENCHMARKING_CASES 6

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

        cudaDeviceSynchronize();
        for (int i = 1;i<BENCHMARKING_CASES; i++){
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
            printf("offset values %d\n",*(outputStream.host_offset));
            printf("outstream vals: %d\n",outputStream.host_values[0]);
            printf("%d \n",outputStream.host_values[1]);
            std::vector<int> kernelValues(outputStream.host_values+*(outputStream.host_offset), outputStream.host_values+outputStream.size);
            std::vector<int> correctTimestamps(CORRECT_STREAM.host_timestamp, CORRECT_STREAM.host_timestamp + CORRECT_STREAM.size);
            std::vector<int> correctValues(CORRECT_STREAM.host_values, CORRECT_STREAM.host_values + CORRECT_STREAM.size);
            printf("%d\n",kernelTimestamps[0]);
            printf("%d \n",kernelTimestamps[1]);
            printf("kernel vals: %d\n",kernelValues[0]);
            printf("%d \n",kernelValues[1]);
            printf("outputStream offset: %d  \n",*outputStream.host_offset);
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
            output_last << duration.count() <<  " us" << " with reader: " <<duration2.count() <<"us size: "<<size <<"\n";
            output_last.close();
        }
    }

    SECTION("delay() benchmarking example"){
        int BENCHMARKING_CASES = 6;
        std::ofstream output_last;
        //delete previous
        output_last.open("benchmarking_delay.data");
        output_last << "";
        output_last.close();
        int dev = 0;
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        //might wanna derive MAX_THREADS and so on from here! TODO!
        printf("Using Device %d: %s\n", dev, deviceProp.name);

        cudaDeviceSynchronize();
        for (int i = 1;i<BENCHMARKING_CASES; i++){
            auto start2 = std::chrono::high_resolution_clock::now();
            std::string path = "../test/data/benchmarking";
            Reader inReader = Reader(path+std::to_string(i)+".in");
            IntStream inputStreamV = inReader.getIntStream("z");
            UnitStream inputStreamR = inReader.getUnitStream("a");
            Reader outReader = Reader(path+std::to_string(i)+"_delay.out");
            IntStream CORRECT_STREAM = outReader.getIntStream("y");
            
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
            printf("offset values %d\n",*(outputStream.host_offset));
            printf("outstream vals: %d\n",outputStream.host_values[0]);
            printf("%d \n",outputStream.host_values[1]);
            std::vector<int> kernelValues(outputStream.host_values+*(outputStream.host_offset), outputStream.host_values+outputStream.size);
            std::vector<int> correctTimestamps(CORRECT_STREAM.host_timestamp, CORRECT_STREAM.host_timestamp + CORRECT_STREAM.size);
            std::vector<int> correctValues(CORRECT_STREAM.host_values, CORRECT_STREAM.host_values + CORRECT_STREAM.size);
            printf("%d\n",kernelTimestamps[0]);
            printf("%d \n",kernelTimestamps[1]);
            printf("kernel vals: %d\n",kernelValues[0]);
            printf("%d \n",kernelValues[1]);
            printf("outputStream offset: %d  \n",*outputStream.host_offset);
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
            output_last << duration.count() <<  " us" << " with reader: " <<duration2.count() <<"us size: "<<size <<"\n";
            output_last.close();
        }
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
