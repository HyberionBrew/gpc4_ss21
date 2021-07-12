
#include <iostream>
#include <string>
#include <string.h>
#include <vector>
#include "../../test/catch2/catch.hpp"

#include <chrono>
#include <fstream>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <cuda_profiler_api.h>
#include <StreamFunctions.h>

#include <Event.h>
#include <Stream.h>
#include <Reader.h>
#include <Writer.h>
#include <StreamFunctions.h>
#include <Debug.h>
#include <chrono>
#include <fstream>

#define BENCHMARKING_CASES 6
#define BENCHMARKING_LOOPS 1
#define CHECK_RESULTS

TEST_CASE("BENCHMARKING SEQUENTIAL") {
    SECTION("last() benchmarking, sequential version") {
        std::ofstream output_last;
        //delete previous
        output_last.open("benchmarking_last_seq.data");
        output_last << "";
        output_last.close();

        for (int j = 1; j <= BENCHMARKING_LOOPS; j++){
            for (int i = 1; i <= BENCHMARKING_CASES; i++) {
                auto start2 = std::chrono::high_resolution_clock::now();

                std::string path = "test/data/benchmarking";
                Reader inReader = Reader(path+std::to_string(i)+".in");
                Reader outReader = Reader(path+std::to_string(i)+"_last.out");

                std::shared_ptr<IntStream> inputStreamV = inReader.getIntStream("z");
                std::shared_ptr<UnitStream> inputStreamR = inReader.getUnitStream("a");
                std::shared_ptr<IntStream> CORRECT_STREAM = outReader.getIntStream("y");

                int size = sizeof(*CORRECT_STREAM) / sizeof(int);

                auto start = std::chrono::high_resolution_clock::now();

                std::shared_ptr<IntStream> result = last(*inputStreamV, *inputStreamR);

                auto stop = std::chrono::high_resolution_clock::now();

                #ifdef CHECK_RESULTS
                REQUIRE(*result == *CORRECT_STREAM);
                #endif

                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(stop - start2);
                output_last.open ("benchmarking_last_seq.data",std::ios::app);
                output_last << "Benchmark " << i << ": " << duration.count()
                            << " us" << " with reader: " << duration2.count()<< " us size: " << size << "\n";
                output_last.close();
            }
        }
    }

    SECTION("time() benchmarking, sequential version") {
        std::ofstream output_last;
        //delete previous
        output_last.open("benchmarking_last_seq.data");
        output_last << "";
        output_last.close();

        for (int j = 1; j <= BENCHMARKING_LOOPS; j++){
            for (int i = 1; i <= BENCHMARKING_CASES; i++) {
                auto start2 = std::chrono::high_resolution_clock::now();

                std::string path = "test/data/benchmarking";
                Reader inReader = Reader(path+std::to_string(i)+".in");
                Reader outReader = Reader(path+std::to_string(i)+"_time.out");

                std::shared_ptr<IntStream> inputStreamV = inReader.getIntStream("z");
                std::shared_ptr<IntStream> CORRECT_STREAM = outReader.getIntStream("y");

                int size = sizeof(*CORRECT_STREAM) / sizeof(int);

                auto start = std::chrono::high_resolution_clock::now();

                std::shared_ptr<IntStream> result = time(*inputStreamV);

                auto stop = std::chrono::high_resolution_clock::now();

                #ifdef CHECK_RESULTS
                REQUIRE(*result == *CORRECT_STREAM);
                #endif

                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(stop - start2);
                output_last.open ("benchmarking_time_seq.data",std::ios::app);
                output_last << "Benchmark " << i << ": " << duration.count()
                            << " us" << " with reader: " << duration2.count()<< " us size: " << size << "\n";
                output_last.close();
            }
        }
    }

    SECTION("delay() benchmarking, sequential version") {
        std::ofstream output_last;
        //delete previous
        output_last.open("benchmarking_delay_seq.data");
        output_last << "";
        output_last.close();

        for (int j = 1; j <= BENCHMARKING_LOOPS; j++){
            for (int i = 3; i <= BENCHMARKING_CASES; i++) {
                printf("dleay test case %i\n", i);
                auto start2 = std::chrono::high_resolution_clock::now();

                std::string path = "test/data/benchmarking";
                Reader inReader = Reader(path+std::to_string(i)+".in");
                Reader outReader = Reader(path+std::to_string(i)+"_delay.out");

                std::shared_ptr<IntStream> inputStreamV = inReader.getIntStream("z");
                std::shared_ptr<UnitStream> inputStreamR = inReader.getUnitStream("a");
                std::shared_ptr<UnitStream> CORRECT_STREAM = outReader.getUnitStream("y");

                int size = sizeof(*CORRECT_STREAM) / sizeof(int);

                auto start = std::chrono::high_resolution_clock::now();

                std::shared_ptr<UnitStream> result = delay(*inputStreamV, *inputStreamR);

                auto stop = std::chrono::high_resolution_clock::now();

                #ifdef CHECK_RESULTS
                REQUIRE(*result == *CORRECT_STREAM);
                #endif

                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(stop - start2);
                output_last.open ("benchmarking_delay_seq.data",std::ios::app);
                output_last << "Benchmark " << i << ": " << duration.count()
                            << " us" << " with reader: " << duration2.count()<< " us size: " << size << "\n";
                output_last.close();
            }
        }
    }

    SECTION("count() benchmarking, sequential version") {
        std::ofstream output_last;
        //delete previous
        output_last.open("benchmarking_last_seq.data");
        output_last << "";
        output_last.close();

        for (int j = 1; j <= BENCHMARKING_LOOPS; j++){
            for (int i = 1; i <= BENCHMARKING_CASES; i++) {
                auto start2 = std::chrono::high_resolution_clock::now();

                std::string path = "test/data/benchmarking";
                Reader inReader = Reader(path+std::to_string(i)+".in");
                Reader outReader = Reader(path+std::to_string(i)+"_count.out");

                std::shared_ptr<UnitStream> inputStreamV = inReader.getUnitStream("a");
                std::shared_ptr<IntStream> CORRECT_STREAM = outReader.getIntStream("y");

                int size = sizeof(*CORRECT_STREAM) / sizeof(int);

                auto start = std::chrono::high_resolution_clock::now();

                std::shared_ptr<IntStream> result = count(*inputStreamV);

                auto stop = std::chrono::high_resolution_clock::now();

                #ifdef CHECK_RESULTS
                REQUIRE(*result == *CORRECT_STREAM);
                #endif

                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(stop - start2);
                output_last.open ("benchmarking_count_seq.data",std::ios::app);
                output_last << "Benchmark " << i << ": " << duration.count()
                            << " us" << " with reader: " << duration2.count()<< " us size: " << size << "\n";
                output_last.close();
            }
        }
    }

}