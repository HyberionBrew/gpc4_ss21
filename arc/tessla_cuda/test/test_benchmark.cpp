//
// Created by fabian on 24/06/2021.
//

#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/time.h>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include <GPUReader.cuh>
#include <GPUStream.cuh>
#include <StreamFunctions.cuh>
#include <StreamFunctionsThrust.cuh>

#include "../../test/catch2/catch.hpp"

#define BENCHMARKING_CASES 5
#define BENCHMARKING_LOOPS 1

static std::string path = "test/data/benchmarking";

TEST_CASE("BENCHMARKING CUDA") {
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("Using Device %d: %s\n", dev, deviceProp.name);

    SECTION("last() benchmarking example") {
        // Clear previous benchmarking results
        std::ofstream output_last;
        output_last.open("benchmarking_last.data");
        output_last << "";
        output_last.close();

        for (int j = 1; j <= BENCHMARKING_LOOPS; j++) {
            cudaDeviceSynchronize();
            for (int i = 1; i <= BENCHMARKING_CASES; i++) {
                auto start2 = std::chrono::high_resolution_clock::now();

                GPUReader inReader = GPUReader(path + std::to_string(i) + ".in");
                GPUReader outReader = GPUReader(path + std::to_string(i) + "_last.out");
                std::shared_ptr<GPUIntStream> inputStreamV = inReader.getIntStream("z");
                std::shared_ptr<GPUUnitStream> inputStreamR = inReader.getUnitStream("a");
                std::shared_ptr<GPUIntStream> CORRECT_STREAM = outReader.getIntStream("y");

                auto start = std::chrono::high_resolution_clock::now();

                // Run kernel
                inputStreamV->copy_to_device();
                inputStreamR->copy_to_device();
                std::shared_ptr<GPUIntStream> outputStream = last(inputStreamV, inputStreamR, 0);
                outputStream->copy_to_host();

                cudaDeviceSynchronize();

                auto stop = std::chrono::high_resolution_clock::now();

                // Compare kernel result to correct data
                std::vector<int> kernelTimestamps(outputStream->host_timestamp + *(outputStream->host_offset),
                                                  outputStream->host_timestamp + outputStream->size);
                std::vector<int> kernelValues(outputStream->host_values + *(outputStream->host_offset),
                                              outputStream->host_values + outputStream->size);
                std::vector<int> correctTimestamps(CORRECT_STREAM->host_timestamp,
                                                   CORRECT_STREAM->host_timestamp + CORRECT_STREAM->size);
                std::vector<int> correctValues(CORRECT_STREAM->host_values,
                                               CORRECT_STREAM->host_values + CORRECT_STREAM->size);
                REQUIRE(kernelTimestamps == correctTimestamps);
                REQUIRE(kernelValues == correctValues);

                // Cleanup
                inputStreamV->free_device();
                inputStreamR->free_device();
                outputStream->free_device();
                inputStreamV->free_host();
                inputStreamR->free_host();
                outputStream->free_host();

                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(stop - start2);
                output_last.open("benchmarking_last.data", std::ios::app);
                output_last << "Benchmark " << i << ": " << duration.count() << " us" << " with reader: " << duration2.count()<< " us size: " << CORRECT_STREAM->size << "\n";
                output_last.close();
            }
        }
    }

    SECTION("time() benchmarking example") {
        // Clear previous benchmarking results
        std::ofstream output_time;
        output_time.open("benchmarking_time.data");
        output_time << "";
        output_time.close();

        for (int j = 1; j <= BENCHMARKING_LOOPS; j++) {
            cudaDeviceSynchronize();
            for (int i = 1; i <= BENCHMARKING_CASES; i++) {
                auto start2 = std::chrono::high_resolution_clock::now();

                GPUReader inReader = GPUReader(path + std::to_string(i) + ".in");
                GPUReader outReader = GPUReader(path + std::to_string(i) + "_time.out");
                std::shared_ptr<GPUIntStream> inputStream = inReader.getIntStream("z");
                std::shared_ptr<GPUIntStream> CORRECT_STREAM = outReader.getIntStream("y");

                auto start = std::chrono::high_resolution_clock::now();
                // Run kernel
                inputStream->copy_to_device();
                std::shared_ptr<GPUIntStream> outputStream = time(inputStream, 0);
                outputStream->copy_to_host();
                
                cudaDeviceSynchronize();

                auto stop = std::chrono::high_resolution_clock::now();
                std::vector<int> kernelTimestamps(outputStream->host_timestamp + *(outputStream->host_offset), outputStream->host_timestamp + outputStream->size);
                std::vector<int> kernelValues(outputStream->host_values + *(outputStream->host_offset), outputStream->host_values + outputStream->size);
                std::vector<int> correctTimestamps(CORRECT_STREAM->host_timestamp, CORRECT_STREAM->host_timestamp + CORRECT_STREAM->size);
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
                output_time << "Benchmark " << i << ": " << duration.count()<< " us" << " with reader: " << duration2.count()<< " us size: " << CORRECT_STREAM->size << "\n";
                output_time.close();
            }
        }
    }

    SECTION("delay() benchmarking example") {
        // Clear previous benchmarking results
        std::ofstream output_delay;
        output_delay.open("benchmarking_delay.data");
        output_delay << "";
        output_delay.close();

        for (int j = 1; j <= BENCHMARKING_LOOPS; j++) {
            cudaDeviceSynchronize();
            for (int i = 3; i <= BENCHMARKING_CASES; i++) {
                //if (i == 4)
                //    continue;
                auto start2 = std::chrono::high_resolution_clock::now();

                GPUReader inReader = GPUReader(path + std::to_string(i) + ".in");
                GPUReader outReader = GPUReader(path + std::to_string(i) + "_delay.out");
                std::shared_ptr<GPUIntStream> inputStreamD = inReader.getIntStream("z");
                std::shared_ptr<GPUUnitStream> inputStreamR = inReader.getUnitStream("a");
                std::shared_ptr<GPUUnitStream> CORRECT_STREAM = outReader.getUnitStream("y");

                auto start = std::chrono::high_resolution_clock::now();
                // Run kernel
                inputStreamD->copy_to_device();
                inputStreamR->copy_to_device();
                std::shared_ptr<GPUUnitStream> outputStream = delay(inputStreamD, inputStreamR,0);
                outputStream->copy_to_host();

                cudaDeviceSynchronize();

                auto stop = std::chrono::high_resolution_clock::now();

                // Compare kernel result to correct data
                int *resultStart = outputStream->host_timestamp + *outputStream->host_offset;
                std::vector<int> kernelTimestamps(resultStart, resultStart + sizeof(resultStart) / sizeof(int));
                std::vector<int> correctTimestamps(CORRECT_STREAM->host_timestamp,
                                                   CORRECT_STREAM->host_timestamp +
                                                   sizeof(CORRECT_STREAM->host_timestamp) / sizeof(int));
                REQUIRE(kernelTimestamps == correctTimestamps);

                // Cleanup
                inputStreamD->free_device();
                inputStreamR->free_device();
                outputStream->free_device();

                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(stop - start2);
                output_delay.open("benchmarking_delay.data", std::ios::app);
                output_delay << "Benchmark " << i << ": " << duration.count() << " us" << " with reader: " << duration2.count() << " us size: " << CORRECT_STREAM->size << "\n";
                output_delay.close();
            }
        }
    }

    SECTION("count() benchmarking example") {
        // Clear previous benchmarking results
        std::ofstream output_count;
        output_count.open("benchmarking_count.data");
        output_count << "";
        output_count.close();

        for (int j = 1; j <= BENCHMARKING_LOOPS; j++) {
            cudaDeviceSynchronize();
            for (int i = 1; i <= BENCHMARKING_CASES; i++) {
                auto start2 = std::chrono::high_resolution_clock::now();

                GPUReader inReader = GPUReader(path + std::to_string(i) + ".in");
                GPUReader outReader = GPUReader(path + std::to_string(i) + "_count.out");
                std::shared_ptr<GPUUnitStream> inputStreamD = inReader.getUnitStreamDebug("a");
                std::shared_ptr<GPUIntStream> CORRECT_STREAM = outReader.getIntStreamDebug("y");

                auto start = std::chrono::high_resolution_clock::now();
                
                // Run kernel
                inputStreamD->copy_to_device();
                std::shared_ptr<GPUIntStream> outputStream = count(inputStreamD);
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
                output_count.open("benchmarking_count.data", std::ios::app);
                output_count << "Benchmark " << i << ": " << duration.count() << " us" << " with reader: " << duration2.count() << " us size: " << CORRECT_STREAM->size << "\n";
                output_count.close();
            }
        }
    }
    
    SECTION("lift() merge benchmarking") {
        std::ofstream output_last;
        output_last.open("benchmarking_lift_merge.data");
        output_last << "";
        output_last.close();

        int dev = 0;
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        printf("Using Device %d: %s\n", dev, deviceProp.name);
        for (int j = 1; j <= BENCHMARKING_LOOPS; j++) {
            cudaDeviceSynchronize();
            for (int i = 1; i <= 5; i++) {
                auto start2 = std::chrono::high_resolution_clock::now();

                std::string path = "test/data/benchmarking";
                GPUReader inReader = GPUReader(path + std::to_string(i) + "_lift.in");
                GPUReader outReader = GPUReader(path + std::to_string(i) + "_slift_merge.out");

                std::shared_ptr<GPUIntStream> inputStreamZ = inReader.getIntStream("z");
                std::shared_ptr<GPUIntStream> inputStreamY = inReader.getIntStream("y");
                std::shared_ptr<GPUIntStream> CORRECT_STREAM = outReader.getIntStream("x");

                int size = CORRECT_STREAM->size;
                auto start = std::chrono::high_resolution_clock::now();

                std::shared_ptr<GPUIntStream> outputStream;
                inputStreamZ->copy_to_device();
                inputStreamY->copy_to_device();

                outputStream = slift(inputStreamZ, inputStreamY, TH_OP_merge);
                outputStream->copy_to_host();

                cudaDeviceSynchronize();

                auto stop = std::chrono::high_resolution_clock::now();

                std::vector<int> kernelTimestamps(outputStream->host_timestamp + *(outputStream->host_offset), outputStream->host_timestamp + outputStream->size);
                std::vector<int> kernelValues(outputStream->host_values + *(outputStream->host_offset), outputStream->host_values + outputStream->size);
                std::vector<int> correctTimestamps(CORRECT_STREAM->host_timestamp, CORRECT_STREAM->host_timestamp + CORRECT_STREAM->size);
                std::vector<int> correctValues(CORRECT_STREAM->host_values, CORRECT_STREAM->host_values + CORRECT_STREAM->size);

                REQUIRE(kernelTimestamps == correctTimestamps);
                REQUIRE(kernelValues == correctValues);

                // Cleanup
                inputStreamZ->free_device();
                inputStreamY->free_device();
                outputStream->free_device();
                inputStreamZ->free_host();
                inputStreamY->free_host();
                outputStream->free_host();

                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(stop - start2);
                output_last.open("benchmarking_lift_merge.data", std::ios::app);
                output_last << "Benchmark " << i << ": " << duration.count()
                            << " us" << " with reader: " << duration2.count()<< " us size: " << size << "\n";
                output_last.close();
            }
        }
    }
}