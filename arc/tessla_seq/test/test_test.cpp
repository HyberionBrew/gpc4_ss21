#include "catch2/catch.hpp"
#include <Event.h>
#include <Stream.h>
#include <Reader.h>
#include <Writer.h>
#include <StreamFunctions.h>
#include <Debug.h>
#include <chrono>
#include <fstream>

TEST_CASE("Basic Stream Operations") {

    SECTION("bt_delay") {
        Reader inReader = Reader("test/data/bt_delay.in");
        std::shared_ptr<IntStream> delayStreamIn = inReader.getIntStream("d");
        std::shared_ptr<UnitStream> resetStreamIn = inReader.getUnitStream("r");

        Reader outReader = Reader("test/data/bt_delay.out");
        std::shared_ptr<UnitStream> intendedResult = outReader.getUnitStream("y");

        std::shared_ptr<UnitStream> result = delay(*delayStreamIn, *resetStreamIn);

        REQUIRE(*result == *intendedResult);
    }

    SECTION("bt_last") {
        Reader inReader = Reader("test/data/bt_last.in");
        std::shared_ptr<IntStream> vStreamIn = inReader.getIntStream("v");
        std::shared_ptr<UnitStream> rStreamIn = inReader.getUnitStream("r");

        Reader outReader = Reader("test/data/bt_last.out");
        std::shared_ptr<IntStream> intendedResult = outReader.getIntStream("y");

        std::shared_ptr<IntStream> result = last(*vStreamIn, *rStreamIn);

        REQUIRE(*result == *intendedResult);
    }

    SECTION("bt_merge") {
        Reader inReader = Reader("test/data/bt_merge.in");
        std::shared_ptr<IntStream> xStreamIn = inReader.getIntStream("x");
        std::shared_ptr<IntStream> yStreamIn = inReader.getIntStream("y");

        Reader outReader = Reader("test/data/bt_merge.out");
        std::shared_ptr<IntStream> intendedResult = outReader.getIntStream("z");

        std::shared_ptr<IntStream> result = merge(*xStreamIn, *yStreamIn);

        REQUIRE(*result == *intendedResult);
    }

    SECTION("bt_time") {
        SECTION("bt_time small dataset") {
            Reader inReader = Reader("test/data/bt_time.in");
            std::shared_ptr<IntStream> xStreamIn = inReader.getIntStream("x");

            Reader outReader = Reader("test/data/bt_time.out");
            std::shared_ptr<IntStream> intendedResult = outReader.getIntStream("x");

            std::shared_ptr<IntStream> result = time(*xStreamIn);

            REQUIRE(*result == *intendedResult);
        }

        SECTION("bt_time bigger dataset") {
            Reader inReader = Reader("test/data/bt_time.bigger.in");
            std::shared_ptr<IntStream> xStreamIn = inReader.getIntStream("z");

            Reader outReader = Reader("test/data/bt_time.bigger.out");
            std::shared_ptr<IntStream> intendedResult = outReader.getIntStream("y");

            std::shared_ptr<IntStream> result = time(*xStreamIn);

            REQUIRE(*result == *intendedResult);
        }
    }

}

TEST_CASE("Constant Test Cases") {

    SECTION("bt_addc") {
        Reader inReader = Reader("test/data/bt_addc.in");
        std::shared_ptr<IntStream> inStream = inReader.getIntStream("x");
        Reader outReader = Reader("test/data/bt_addc.out");
        std::shared_ptr<IntStream> intendedResult = outReader.getIntStream("y");

        std::shared_ptr<IntStream> result = add(*inStream, 1);
        REQUIRE(*result == *intendedResult);
    }

    SECTION("bt_adds") {
        Reader inReader = Reader("test/data/bt_adds.in");
        std::shared_ptr<IntStream> inStream1 = inReader.getIntStream("x");
        std::shared_ptr<IntStream> inStream2 = inReader.getIntStream("y");
        Reader outReader = Reader("test/data/bt_adds.out");
        std::shared_ptr<IntStream> intendedResult = outReader.getIntStream("z");

        std::shared_ptr<IntStream> result = add(*inStream1, *inStream2);
        REQUIRE(*result == *intendedResult);
    }

    SECTION("bt_subc") {
        Reader inReader = Reader("test/data/bt_subc.in");
        std::shared_ptr<IntStream> inStream = inReader.getIntStream("x");
        Reader outReader = Reader("test/data/bt_subc.out");
        std::shared_ptr<IntStream> intendedResult = outReader.getIntStream("y");

        std::shared_ptr<IntStream> result = sub1(*inStream, 3);
        REQUIRE(*result == *intendedResult);
    }

    SECTION("bt_mulc") {
        Reader inReader = Reader("test/data/bt_mulc.in");
        std::shared_ptr<IntStream> inStream = inReader.getIntStream("x");
        Reader outReader = Reader("test/data/bt_mulc.out");
        std::shared_ptr<IntStream> intendedResult = outReader.getIntStream("y");

        std::shared_ptr<IntStream> result = mul(*inStream, 4);
        REQUIRE(*result == *intendedResult);
    }

    SECTION("bt_divc") {
        Reader inReader = Reader("test/data/bt_divc.in");
        std::shared_ptr<IntStream> inStream = inReader.getIntStream("x");
        Reader outReader = Reader("test/data/bt_divc.out");
        std::shared_ptr<IntStream> intendedResult = outReader.getIntStream("y");

        std::shared_ptr<IntStream> result = div1(*inStream, 3);
        REQUIRE(*result == *intendedResult);
    }

    SECTION("bt_modc") {
        Reader inReader = Reader("test/data/bt_modc.in");
        std::shared_ptr<IntStream> inStream = inReader.getIntStream("x");
        Reader outReader = Reader("test/data/bt_modc.out");
        std::shared_ptr<IntStream> intendedResult = outReader.getIntStream("y");

        std::shared_ptr<IntStream> result = mod1(*inStream, 2);
        REQUIRE(*result == *intendedResult);
    }
}

TEST_CASE("Stream Arithmetic Test Cases (slift)") {

    SECTION("bt_adds") {
        Reader inReader = Reader("test/data/bt_adds.in");
        std::shared_ptr<IntStream> inStream1 = inReader.getIntStream("x");
        std::shared_ptr<IntStream> inStream2 = inReader.getIntStream("y");
        Reader outReader = Reader("test/data/bt_adds.out");
        std::shared_ptr<IntStream> intendedResult = outReader.getIntStream("z");

        std::shared_ptr<IntStream> result = add(*inStream1, *inStream2);
        REQUIRE(*result == *intendedResult);
    }

    SECTION("bt_subs") {
        Reader inReader = Reader("test/data/bt_subs.in");
        std::shared_ptr<IntStream> inStream1 = inReader.getIntStream("x");
        std::shared_ptr<IntStream> inStream2 = inReader.getIntStream("y");
        Reader outReader = Reader("test/data/bt_subs.out");
        std::shared_ptr<IntStream> intendedResult = outReader.getIntStream("z");

        std::shared_ptr<IntStream> result = sub(*inStream1, *inStream2);
        REQUIRE(*result == *intendedResult);
    }

    SECTION("bt_muls") {
        Reader inReader = Reader("test/data/bt_muls.in");
        std::shared_ptr<IntStream> inStream1 = inReader.getIntStream("x");
        std::shared_ptr<IntStream> inStream2 = inReader.getIntStream("y");
        Reader outReader = Reader("test/data/bt_muls.out");
        std::shared_ptr<IntStream> intendedResult = outReader.getIntStream("z");

        std::shared_ptr<IntStream> result = mul(*inStream1, *inStream2);
        REQUIRE(*result == *intendedResult);
    }

    SECTION("bt_divs") {
        Reader inReader = Reader("test/data/bt_divs.in");
        std::shared_ptr<IntStream> inStream1 = inReader.getIntStream("x");
        std::shared_ptr<IntStream> inStream2 = inReader.getIntStream("y");
        Reader outReader = Reader("test/data/bt_divs.out");
        std::shared_ptr<IntStream> intendedResult = outReader.getIntStream("z");

        std::shared_ptr<IntStream> result = div(*inStream1, *inStream2);
        REQUIRE(*result == *intendedResult);
    }

    SECTION("bt_mods") {
        Reader inReader = Reader("test/data/bt_mods.in");
        std::shared_ptr<IntStream> inStream1 = inReader.getIntStream("x");
        std::shared_ptr<IntStream> inStream2 = inReader.getIntStream("y");
        Reader outReader = Reader("test/data/bt_mods.out");
        std::shared_ptr<IntStream> intendedResult = outReader.getIntStream("z");

        std::shared_ptr<IntStream> result = mod(*inStream1, *inStream2);
        REQUIRE(*result == *intendedResult);
    }
}

#define BENCHMARKING_CASES 5
#define BENCHMARKING_LOOPS 1
//#define CHECK_RESULTS

TEST_CASE("Benchmarks") {
    SECTION("last() benchmarking example"){
        std::ofstream output_last;
        //delete previous
        printf("---last---- benchmark\n");
        output_last.open("benchmarking_last.data");
        output_last << "";
        output_last.close();

        for (int j=1;j <=BENCHMARKING_LOOPS;j++){
            for (int i = 1; i<BENCHMARKING_CASES; i++) {
                auto start2 = std::chrono::high_resolution_clock::now();
                std::string path = "test/data/benchmarking";
                Reader inReader = Reader(path+std::to_string(i)+".in");
                std::shared_ptr<IntStream> inputStreamV = inReader.getIntStream("z");
                std::shared_ptr<UnitStream> inputStreamR = inReader.getUnitStream("a");
                Reader outReader = Reader(path+std::to_string(i)+"_last.out");
                std::shared_ptr<IntStream> CORRECT_STREAM = outReader.getIntStream("y");
                
                auto start = std::chrono::high_resolution_clock::now();
                std::shared_ptr<IntStream> result = last(*inputStreamV, *inputStreamR);
                auto stop = std::chrono::high_resolution_clock::now();

                #ifdef CHECK_RESULTS
                REQUIRE(*result == *CORRECT_STREAM);
                #endif

                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(stop - start2);
                output_last.open ("benchmarking_last.data",std::ios::app);
                output_last << duration.count() <<  " us" << " with reader: " <<duration2.count() <<"us \n";
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
        for (int j=1;j <=BENCHMARKING_LOOPS;j++){
            for (int i = 1;i<=BENCHMARKING_CASES; i++){
                auto start2 = std::chrono::high_resolution_clock::now();
                std::string path = "test/data/benchmarking";

                // Prepare empty output stream to fill
                Reader inReader = Reader(path+std::to_string(i)+".in");
                std::shared_ptr<IntStream> inputStream = inReader.getIntStream("z");
                Reader outReader = Reader(path+std::to_string(i)+"_time.out");
                std::shared_ptr<IntStream> CORRECT_STREAM = outReader.getIntStream("y");

                auto start = std::chrono::high_resolution_clock::now();
                std::shared_ptr<IntStream> result = time(*inputStream);
                auto stop = std::chrono::high_resolution_clock::now();

                #ifdef CHECK_RESULTS
                REQUIRE(*result == *CORRECT_STREAM);
                #endif

                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(stop - start2);
                
                output_time.open ("benchmarking_time.data",std::ios::app);
                output_time <<"Benchmark "<< i <<": "<<duration.count() <<  " us" << " with reader: " <<duration2.count() <<" us\n";
                output_time.close();
            }
        }
    }

    SECTION("delay() benchmarking example"){
        std::ofstream output_delay;
        //delete previous
        printf("---delay---- benchmark\n");
        output_delay.open("benchmarking_delay.data");
        output_delay << "";
        output_delay.close();
        for (int j=1;j <=BENCHMARKING_LOOPS;j++){
            // TODO: Check benchmarking case 3
            for (int i = 4;i<=BENCHMARKING_CASES; i++){
                auto start2 = std::chrono::high_resolution_clock::now();
                std::string path = "test/data/benchmarking";

                Reader inReader = Reader(path+std::to_string(i)+".in");
                std::shared_ptr<IntStream> inputStreamD = inReader.getIntStream("z");
                std::shared_ptr<UnitStream> inputStreamR = inReader.getUnitStream("a");
                Reader outReader = Reader(path+std::to_string(i)+"_delay.out");
                std::shared_ptr<UnitStream> CORRECT_STREAM = outReader.getUnitStream("y");
                auto start = std::chrono::high_resolution_clock::now();
                std::shared_ptr<UnitStream> result = delay(*inputStreamD, *inputStreamR);
                auto stop = std::chrono::high_resolution_clock::now();

                #ifdef CHECK_RESULTS
                REQUIRE(*result == *CORRECT_STREAM);
                #endif

                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(stop - start2);
                
                output_delay.open ("benchmarking_delay.data",std::ios::app);
                output_delay <<"Benchmark "<< i <<": "<<duration.count() <<  " us" << " with reader: " <<duration2.count() <<" us\n";
                output_delay.close();
            }
        }
    }
}