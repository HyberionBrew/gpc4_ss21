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


#define BENCHMARKING_CASES 6
#define BENCHMARKING_LOOPS 1

TEST_CASE("BENCHMARKING") {
    SECTION("last() benchmarking example") {
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
        for (int j = 1; j <= BENCHMARKING_LOOPS; j++) {

            cudaDeviceSynchronize();

            for (int i = 1; i <= BENCHMARKING_CASES; i++) {
                auto start2 = std::chrono::high_resolution_clock::now();
                std::string path = "test/data/benchmarking";
                GPUReader inReader = GPUReader(path + std::to_string(i) + ".in");
                std::shared_ptr<GPUIntStream> inputStreamV = inReader.getIntStreamDebug("z");
                std::shared_ptr<GPUUnitStream> inputStreamR = inReader.getUnitStreamDebug("a");
                GPUReader outReader = GPUReader(path + std::to_string(i) + "_last.out");
                std::shared_ptr<GPUIntStream> CORRECT_STREAM = outReader.getIntStreamDebug("y");
                /*GPUReader inReader = GPUReader("test/data/bt_last.in");
                std::shared_ptr<GPUIntStream> inputStreamV = inReader.getIntStreamDebug("v");
                std::shared_ptr<GPUUnitStream> inputStreamR = inReader.getUnitStreamDebug("r");
                GPUReader outReader = GPUReader("../test/data/bt_last.out");
                std::shared_ptr<GPUIntStream> CORRECT_STREAM = outReader.getIntStreamDebug("y");*/
                //now start timer
                // Prepare empty output stream to fill
                int size = CORRECT_STREAM->size;
                auto start = std::chrono::high_resolution_clock::now();
                std::shared_ptr<GPUIntStream> outputStream;
                // Run kernel
                inputStreamV->copy_to_device();

                inputStreamR->copy_to_device();
                // inputStreamV->print();
                //  inputStreamR->print();
                outputStream = last(inputStreamV, inputStreamR, 0);
                //inputStreamR->print();
                outputStream->copy_to_host();

                cudaDeviceSynchronize();

                auto stop = std::chrono::high_resolution_clock::now();
                //outputStream->print();
                // Compare kernel result to correct data
                std::vector<int> kernelTimestamps(outputStream->host_timestamp + *(outputStream->host_offset),
                                                  outputStream->host_timestamp + outputStream->size);

                std::vector<int> kernelValues(outputStream->host_values + *(outputStream->host_offset),
                                              outputStream->host_values + outputStream->size);
                std::vector<int> correctTimestamps(CORRECT_STREAM->host_timestamp,
                                                   CORRECT_STREAM->host_timestamp + CORRECT_STREAM->size);
                std::vector<int> correctValues(CORRECT_STREAM->host_values,
                                               CORRECT_STREAM->host_values + CORRECT_STREAM->size);


                REQUIRE(kernelTimestamps== correctTimestamps);
                REQUIRE(kernelValues== correctValues);
                //outputStream.print();
                // Cleanup
                inputStreamV->free_device();

                inputStreamR->free_device();

                outputStream->free_device();

                inputStreamV->free_host();

                inputStreamR->free_host();

                outputStream->free_host();

                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                printf("%li us\n", duration.count());
                auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(stop - start2);
                output_last.open("benchmarking_last.data", std::ios::app);
                output_last << "Benchmark " << i << ": " << duration.count()
                 << " us" << " with reader: " << duration2.count()<< " us size: " << size << "\n";
                output_last.close();

            }
        }
    }


    SECTION("time() benchmarking example") {
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

        for (int j = 1;j <= BENCHMARKING_LOOPS; j++) {
            //cudaDeviceSynchronize();
            cudaDeviceSynchronize();

            for (int i = 1;i <= BENCHMARKING_CASES; i++) {
                auto start2 = std::chrono::high_resolution_clock::now();
                std::string path = "test/data/benchmarking";

                // Prepare empty output stream to fill
                GPUReader inReader = GPUReader(path + std::to_string(i) + ".in");
                std::shared_ptr<GPUIntStream> inputStream = inReader.getIntStreamDebug("z");
                GPUReader outReader = GPUReader(path + std::to_string(i) + "_time.out");
                std::shared_ptr<GPUIntStream> CORRECT_STREAM = outReader.getIntStreamDebug("y");

            // Prepare empty output stream to fill
                int size = CORRECT_STREAM->size;

                std::shared_ptr<GPUIntStream> outputStream; //(host_timestampOut, host_valueOut, size);

                // Run kernel
                auto start = std::chrono::high_resolution_clock::now();
                inputStream->copy_to_device();
                //inputStream->print();
                //outputStream.copy_to_device();
                outputStream = time(inputStream, 0);
                outputStream->copy_to_host();
                //outputStream->print();

                auto stop = std::chrono::high_resolution_clock::now();
                //CORRECT_STREAM->print();
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
                inputStream->free_device();

                outputStream->free_device();

                inputStream->free_host();

                outputStream->free_host();

                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(stop - start2);

                output_time.open("benchmarking_time.data", std::ios::app);
                output_time << "Benchmark " << i << ": " << duration.count()<< " us" << " with reader: " << duration2.count()<< " us size: " << size << "\n";
                output_time.close();

            }
        }
    }


    SECTION("delay() benchmarking example") {
        std::ofstream output_delay;
//delete previous
        output_delay.open("benchmarking_delay.data");
        output_delay << "";
        output_delay.close();

        int dev = 0;
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev
        );
//might wanna derive MAX_THREADS and so on from here! TODO!

        for (int j = 1;j <= BENCHMARKING_LOOPS; j++) {
//cudaDeviceSynchronize();
            cudaDeviceSynchronize();

            for (int i = 3;i <= BENCHMARKING_CASES; i++) {
                auto start2 = std::chrono::high_resolution_clock::now();
                std::string path = "test/data/benchmarking";

// Prepare empty output stream to fill

                GPUReader inReader = GPUReader(path + std::to_string(i) + ".in");
                std::shared_ptr<GPUIntStream> inputStreamD = inReader.getIntStreamDebug("z");
                std::shared_ptr<GPUUnitStream> inputStreamR = inReader.getUnitStreamDebug("a");
                GPUReader outReader = GPUReader(path + std::to_string(i) + "_delay.out");
                std::shared_ptr<GPUUnitStream> CORRECT_STREAM = outReader.getUnitStreamDebug("y");

// Prepare empty output stream to fill
                int size = inputStreamR->size;
                int sizeAllocated = (size_t) size * sizeof(int);
                int *host_timestampOut = (int *) malloc(size * sizeof(int));
                memset(host_timestampOut,
                       -1, sizeAllocated);


                std::shared_ptr<GPUUnitStream> outputStream = std::make_shared<GPUUnitStream>(host_timestampOut, size);
                auto start = std::chrono::high_resolution_clock::now();
// Run kernel
                inputStreamD->copy_to_device();

                inputStreamR->copy_to_device();

                outputStream->copy_to_device();

                outputStream = delay(inputStreamD, inputStreamR,0);
                outputStream->copy_to_host();

                cudaDeviceSynchronize();

                auto stop = std::chrono::high_resolution_clock::now();

// Compare kernel result to correct data
                int *resultStart = outputStream->host_timestamp + *outputStream->host_offset;
                std::vector<int> kernelTimestamps(resultStart, resultStart + sizeof(resultStart) / sizeof(int));
                std::vector<int> correctTimestamps(CORRECT_STREAM->host_timestamp,
                                                   CORRECT_STREAM->host_timestamp +
                                                   sizeof(CORRECT_STREAM->host_timestamp) / sizeof(int));

                REQUIRE(kernelTimestamps== correctTimestamps);

// Cleanup
                inputStreamD->free_device();

                inputStreamR->free_device();

                outputStream->free_device();
                free(host_timestampOut);

                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(stop - start2);

                output_delay.open("benchmarking_delay.data", std::ios::app);
                output_delay << "Benchmark " << i << ": " << duration.count() << " us" << " with reader: " << duration2.count() << " us size: " << size << "\n";
                output_delay.close();
            }
        }
    }

/*
    SECTION("lift() benchmarking") {
        std::ofstream output_delay;
//delete previous
        output_delay.open("benchmarking_lift.data");
        output_delay << "";
        output_delay.close();

        int dev = 0;
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
//might wanna derive MAX_THREADS and so on from here! TODO!

        for (int j = 1; j <= BENCHMARKING_LOOPS; j++) {
//cudaDeviceSynchronize();
            cudaDeviceSynchronize();

            for (int i = 3; i <= BENCHMARKING_CASES; i++) {
                auto start2 = std::chrono::high_resolution_clock::now();
                std::string path = "test/data/benchmarking";

// Prepare empty output stream to fill

                GPUReader inReader = GPUReader(path + std::to_string(i) + ".in");
                std::shared_ptr<GPUIntStream> inputStreamD = inReader.getIntStreamDebug("z");
                std::shared_ptr<GPUUnitStream> inputStreamR = inReader.getUnitStreamDebug("a");
                GPUReader outReader = GPUReader(path + std::to_string(i) + "_lift.out");
                std::shared_ptr<GPUUnitStream> CORRECT_STREAM = outReader.getUnitStreamDebug("y");

// Prepare empty output stream to fill
                int size = inputStreamR->size;
                int sizeAllocated = (size_t) size * sizeof(int);
                int *host_timestampOut = (int *) malloc(size * sizeof(int));
                memset(host_timestampOut,
                       -1, sizeAllocated);
                std::shared_ptr<GPUUnitStream> outputStream = std::make_shared<GPUUnitStream>(host_timestampOut, size);
                auto start = std::chrono::high_resolution_clock::now();
// Run kernel
                inputStreamD->copy_to_device();

                inputStreamR->copy_to_device();

                outputStream->copy_to_device();

                outputStream = delay(inputStreamD, inputStreamR, 0);
                outputStream->copy_to_host();

                cudaDeviceSynchronize();

                auto stop = std::chrono::high_resolution_clock::now();

// Compare kernel result to correct data
                int *resultStart = outputStream->host_timestamp + *outputStream->host_offset;
                std::vector<int> kernelTimestamps(resultStart, resultStart + sizeof(resultStart) / sizeof(int));
                std::vector<int> correctTimestamps(CORRECT_STREAM->host_timestamp,
                                                   CORRECT_STREAM->host_timestamp +
                                                   sizeof(CORRECT_STREAM->host_timestamp) / sizeof(int));

                REQUIRE(kernelTimestamps== correctTimestamps);

// Cleanup
                inputStreamD->free_device();

                inputStreamR->free_device();

                outputStream->free_device();

                free(host_timestampOut);

                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(stop - start2);

                output_delay.open("benchmarking_delay.data", std::ios::app);
                output_delay << "Benchmark " << i << ": " << duration.count()<< " us" << " with reader: " << duration2.count() << " us size: " << size << "\n";
                output_delay.close();

            }
        }
    }
*/

    SECTION("count() benchmarking example") {
        std::ofstream output_delay;
//delete previous
        output_delay.open("benchmarking_count.data");
        output_delay << "";
        output_delay.close();

        int dev = 0;
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev
        );
//might wanna derive MAX_THREADS and so on from here! TODO!

        for (int j = 1;j <= BENCHMARKING_LOOPS; j++) {
//cudaDeviceSynchronize();
            cudaDeviceSynchronize();

            for (int i = 1;i <= BENCHMARKING_CASES; i++) {
                auto start2 = std::chrono::high_resolution_clock::now();
                std::string path = "test/data/benchmarking";

// Prepare empty output stream to fill

                GPUReader inReader = GPUReader(path + std::to_string(i) + ".in");
                std::shared_ptr<GPUUnitStream> inputStreamD = inReader.getUnitStreamDebug("a");
                GPUReader outReader = GPUReader(path + std::to_string(i) + "_count.out");
                std::shared_ptr<GPUIntStream> CORRECT_STREAM = outReader.getIntStreamDebug("y");

// Prepare empty output stream to fill
                int size = inputStreamD->size;

                std::shared_ptr<GPUIntStream> outputStream;// = std::make_shared<GPUIntStream>(host_timestampOut, size);
                auto start = std::chrono::high_resolution_clock::now();
// Run kernel
                inputStreamD->copy_to_device();


                //outputStream->copy_to_device();
                outputStream = count(inputStreamD);
                outputStream->copy_to_host();

                cudaDeviceSynchronize();

                auto stop = std::chrono::high_resolution_clock::now();

// Compare kernel result to correct data
                int *resultStart = outputStream->host_timestamp + *outputStream->host_offset;
                std::vector<int> kernelTimestamps(outputStream->host_timestamp + *(outputStream->host_offset), outputStream->host_timestamp + outputStream->size);
                std::vector<int> kernelValues(outputStream->host_values + *(outputStream->host_offset), outputStream->host_values + outputStream->size);
                std::vector<int> correctTimestamps(CORRECT_STREAM->host_timestamp, CORRECT_STREAM->host_timestamp + CORRECT_STREAM->size);
                std::vector<int> correctValues(CORRECT_STREAM->host_values, CORRECT_STREAM->host_values + CORRECT_STREAM->size);

                REQUIRE(kernelTimestamps == correctTimestamps);
                REQUIRE(kernelValues == correctValues);

// Cleanup
                inputStreamD->free_device();
                outputStream->free_device();
                inputStreamD->free_host();
                outputStream->free_host();

                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(stop - start2);

                output_delay.open("benchmarking_count.data", std::ios::app);
                output_delay << "Benchmark " << i << ": " << duration.count() << " us" << " with reader: " << duration2.count() << " us size: " << size << "\n";
                output_delay.close();
            }
        }
    }
    
}