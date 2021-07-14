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

TEST_CASE("BENCHMARKING THRUST") {
    SECTION("last() benchmarking, thrust version") {
        std::ofstream output_last;
        output_last.open("benchmarking_last_thrust.data");
        output_last << "";
        output_last.close();

        int dev = 0;
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        printf("Using Device %d: %s\n", dev, deviceProp.name);
        for (int j = 1; j <= BENCHMARKING_LOOPS; j++) {
            cudaDeviceSynchronize();
            for (int i = 1; i <= BENCHMARKING_CASES; i++) {
                auto start2 = std::chrono::high_resolution_clock::now();

                std::string path = "test/data/benchmarking";
                GPUReader inReader = GPUReader(path + std::to_string(i) + ".in");
                GPUReader outReader = GPUReader(path + std::to_string(i) + "_last.out");

                std::shared_ptr<GPUIntStream> inputStreamZ = inReader.getIntStream("z");
                std::shared_ptr<GPUUnitStream> inputStreamA = inReader.getUnitStream("a");
                std::shared_ptr<GPUIntStream> CORRECT_STREAM = outReader.getIntStream("y");

                int size = CORRECT_STREAM->size;
                auto start = std::chrono::high_resolution_clock::now();

                std::shared_ptr<GPUIntStream> outputStream;

                inputStreamZ->copy_to_device();
                inputStreamA->copy_to_device();

                outputStream = last_thrust(inputStreamZ, inputStreamA, 0);
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
                inputStreamA->free_device();
                inputStreamZ->free_device();
                outputStream->free_device();
                inputStreamZ->free_host();
                inputStreamA->free_host();
                outputStream->free_host();

                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(stop - start2);
                output_last.open("benchmarking_last_thrust.data", std::ios::app);
                output_last << "Benchmark " << i << ": " << duration.count()
                            << " us" << " with reader: " << duration2.count()<< " us size: " << size << "\n";
                output_last.close();
            }
        }
    }

    SECTION("time() benchmarking, thrust version") {
        // OMITTED
    }

    SECTION("delay() benchmarking, thrust version") {
        std::ofstream output_last;
        output_last.open("benchmarking_delay_thrust.data");
        output_last << "";
        output_last.close();

        int dev = 0;
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        printf("Using Device %d: %s\n", dev, deviceProp.name);
        for (int j = 1; j <= BENCHMARKING_LOOPS; j++) {
            cudaDeviceSynchronize();
            for (int i = 3; i <= BENCHMARKING_CASES; i++) {
                auto start2 = std::chrono::high_resolution_clock::now();

                std::string path = "test/data/benchmarking";
                GPUReader inReader = GPUReader(path + std::to_string(i) + ".in");
                GPUReader outReader = GPUReader(path + std::to_string(i) + "_delay.out");

                std::shared_ptr<GPUIntStream> inputStreamZ = inReader.getIntStream("z");
                std::shared_ptr<GPUUnitStream> inputStreamA = inReader.getUnitStream("a");
                std::shared_ptr<GPUUnitStream> CORRECT_STREAM = outReader.getUnitStream("y");

                int size = CORRECT_STREAM->size;
                auto start = std::chrono::high_resolution_clock::now();

                std::shared_ptr<GPUUnitStream> outputStream;

                inputStreamZ->copy_to_device();
                inputStreamA->copy_to_device();

                outputStream = delay_thrust(inputStreamZ, inputStreamA, 0);
                outputStream->copy_to_host();

                cudaDeviceSynchronize();

                auto stop = std::chrono::high_resolution_clock::now();

                std::vector<int> kernelTimestamps(outputStream->host_timestamp + *(outputStream->host_offset), outputStream->host_timestamp + outputStream->size);
                std::vector<int> correctTimestamps(CORRECT_STREAM->host_timestamp, CORRECT_STREAM->host_timestamp + CORRECT_STREAM->size);

                REQUIRE(kernelTimestamps == correctTimestamps);

                // Cleanup
                inputStreamA->free_device();
                inputStreamZ->free_device();
                outputStream->free_device();
                inputStreamZ->free_host();
                inputStreamA->free_host();
                outputStream->free_host();

                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(stop - start2);
                output_last.open("benchmarking_delay_thrust.data", std::ios::app);
                output_last << "Benchmark " << i << ": " << duration.count()
                            << " us" << " with reader: " << duration2.count()<< " us size: " << size << "\n";
                output_last.close();
            }
        }
    }

    SECTION("count() benchmarking, thrust version") {
        std::ofstream output_last;
        output_last.open("benchmarking_count_thrust.data");
        output_last << "";
        output_last.close();

        int dev = 0;
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        printf("Using Device %d: %s\n", dev, deviceProp.name);
        for (int j = 1; j <= BENCHMARKING_LOOPS; j++) {
            cudaDeviceSynchronize();
            for (int i = 1; i <= BENCHMARKING_CASES - 1; i++) {
                printf("Starting testcase %i\n", i);
                auto start2 = std::chrono::high_resolution_clock::now();

                std::string path = "test/data/benchmarking";
                GPUReader inReader = GPUReader(path + std::to_string(i) + ".in");
                GPUReader outReader = GPUReader(path + std::to_string(i) + "_count.out");

                std::shared_ptr<GPUUnitStream> inputStreamA = inReader.getUnitStream("a");
                std::shared_ptr<GPUIntStream> CORRECT_STREAM = outReader.getIntStream("y");

                int size = CORRECT_STREAM->size;
                auto start = std::chrono::high_resolution_clock::now();

                std::shared_ptr<GPUIntStream> outputStream;
                inputStreamA->copy_to_device();

                outputStream = count_thrust(inputStreamA);
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
                inputStreamA->free_device();
                outputStream->free_device();
                inputStreamA->free_host();
                outputStream->free_host();

                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(stop - start2);
                output_last.open("benchmarking_count_thrust.data", std::ios::app);
                output_last << "Benchmark " << i << ": " << duration.count()
                            << " us" << " with reader: " << duration2.count()<< " us size: " << size << "\n";
                output_last.close();
            }
        }
    }

    SECTION("time() benchmarking, thrust version") {
        std::ofstream output_last;
        output_last.open("benchmarking_time_thrust.data");
        output_last << "";
        output_last.close();

        int dev = 0;
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        printf("Using Device %d: %s\n", dev, deviceProp.name);
        for (int j = 1; j <= BENCHMARKING_LOOPS; j++) {
            cudaDeviceSynchronize();
            for (int i = 1; i <= BENCHMARKING_CASES; i++) {
                auto start2 = std::chrono::high_resolution_clock::now();

                std::string path = "test/data/benchmarking";
                GPUReader inReader = GPUReader(path + std::to_string(i) + ".in");
                GPUReader outReader = GPUReader(path + std::to_string(i) + "_time.out");

                std::shared_ptr<GPUIntStream> inputStreamZ = inReader.getIntStream("z");
                std::shared_ptr<GPUIntStream> CORRECT_STREAM = outReader.getIntStream("y");

                int size = CORRECT_STREAM->size;
                auto start = std::chrono::high_resolution_clock::now();

                std::shared_ptr<GPUIntStream> outputStream;

                inputStreamZ->copy_to_device();

                outputStream = time_thrust(inputStreamZ);
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
                outputStream->free_device();
                inputStreamZ->free_host();
                outputStream->free_host();

                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(stop - start2);
                output_last.open("benchmarking_time_thrust.data", std::ios::app);
                output_last << "Benchmark " << i << ": " << duration.count()
                            << " us" << " with reader: " << duration2.count()<< " us size: " << size << "\n";
                output_last.close();
            }
        }
    }

    SECTION("lift() merge benchmarking, thrust version") {
        std::ofstream output_last;
        output_last.open("benchmarking_lift_merge_thrust.data");
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

                outputStream = slift_thrust(inputStreamZ, inputStreamY, TH_OP_merge, 0);
                outputStream->copy_to_host();

                cudaDeviceSynchronize();

                auto stop = std::chrono::high_resolution_clock::now();

                std::vector<int> kernelTimestamps(outputStream->host_timestamp + *(outputStream->host_offset), outputStream->host_timestamp + outputStream->size);
                std::vector<int> kernelValues(outputStream->host_values + *(outputStream->host_offset), outputStream->host_values + outputStream->size);
                std::vector<int> correctTimestamps(CORRECT_STREAM->host_timestamp, CORRECT_STREAM->host_timestamp + CORRECT_STREAM->size);
                std::vector<int> correctValues(CORRECT_STREAM->host_values, CORRECT_STREAM->host_values + CORRECT_STREAM->size);

                //REQUIRE(kernelTimestamps[2] == correctTimestamps[2]);
                /*
                for (int i = 0; i < 50; i++ ) {
                    printf("%i, ", kernelTimestamps[i]);
                }
                printf("\n");
                for (int i = 0; i < 50; i++ ) {
                    printf("%i, ", kernelValues[i]);
                }
                printf("\n");
                */
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
                output_last.open("benchmarking_lift_merge_thrust.data", std::ios::app);
                output_last << "Benchmark " << i << ": " << duration.count()
                            << " us" << " with reader: " << duration2.count()<< " us size: " << size << "\n";
                output_last.close();
            }
        }
    }

}