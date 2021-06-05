#include <string.h>
#include <vector>
#include "../../test/catch2/catch.hpp"
#include "../src/Reader.cuh"
#include "../src/Stream.cuh"
#include "../src/StreamFunctions.cuh"
TEST_CASE("delay()"){
    SECTION("delay() tuwel example"){
        
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
        int sizeAllocated = (size_t)  inputStreamR.size * sizeof(int);
        int *host_timestampOut = (int *) malloc( inputStreamR.size * sizeof(int));
        int *host_valueOut = (int*) malloc( inputStreamR.size * sizeof(int));
        memset(host_timestampOut, 0, sizeAllocated);
        memset(host_valueOut, 0, sizeAllocated);
        IntStream outputStream(host_timestampOut, host_valueOut, inputStreamR.size);

        // Run kernel
        inputStreamV.copy_to_device();
        inputStreamR.copy_to_device();
        outputStream.copy_to_device();
       // inputStreamV.print();
      //  inputStreamR.print();
        last(&inputStreamV, &inputStreamR, &outputStream, 0);
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
        free(host_valueOut);
        free(host_timestampOut);
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
        int sizeAllocated = (size_t)  inputStreamR.size * sizeof(int);
        int *host_timestampOut = (int *) malloc( inputStreamR.size * sizeof(int));
        int *host_valueOut = (int*) malloc( inputStreamR.size * sizeof(int));
        memset(host_timestampOut, 0, sizeAllocated);
        memset(host_valueOut, 0, sizeAllocated);
        IntStream outputStream(host_timestampOut, host_valueOut, inputStreamR.size);

        // Run kernel
        inputStreamV.copy_to_device();
        inputStreamR.copy_to_device();
        outputStream.copy_to_device();
        //inputStreamV.print();
        //inputStreamR.print();
        last(&inputStreamV, &inputStreamR, &outputStream, 0);
        outputStream.copy_to_host();
        //printf("xx");
        //outputStream.print();
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
        free(host_valueOut);
        free(host_timestampOut);
    }

    SECTION("last() large random example") {
        // Read input and correct output data
        Reader inReader = Reader("../test/data/last_test2.in");
        IntStream inputStreamV = inReader.getIntStream("z");
        UnitStream inputStreamR = inReader.getUnitStream("a");
        Reader outReader = Reader("../test/data/last_test2.out");
        IntStream CORRECT_STREAM = outReader.getIntStream("y");

        // Prepare empty output stream to fill
        int size = CORRECT_STREAM.size;
        int sizeAllocated = (size_t)  inputStreamR.size * sizeof(int);
        int *host_timestampOut = (int *) malloc( inputStreamR.size * sizeof(int));
        int *host_valueOut = (int*) malloc( inputStreamR.size * sizeof(int));
        memset(host_timestampOut, 0, sizeAllocated);
        memset(host_valueOut, 0, sizeAllocated);
        IntStream outputStream(host_timestampOut, host_valueOut, inputStreamR.size);

        // Run kernel
        inputStreamV.copy_to_device();
        inputStreamR.copy_to_device();
        outputStream.copy_to_device();
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
        free(host_valueOut);
        free(host_timestampOut);
    }

    SECTION("last() twice test with no invalids") {
        printf("-------------------------\n");
        // Read input and correct output data
        Reader inReader = Reader("../test/data/last_test3.in");
        IntStream inputStreamV = inReader.getIntStream("z");
        UnitStream inputStream2 = inReader.getUnitStream("a");
        UnitStream inputStreamR = inReader.getUnitStream("x");
        Reader outReader = Reader("../test/data/last_test3.out");
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

    SECTION("last() twice test with invalids in Unit Stream") {
        printf("-------------------------\n");
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

TEST_CASE("Basic Stream Operations") {
    SECTION("delay() (not implemented yet)") {
        // Read input and correct output data
        Reader inReader = Reader("../test/data/bt_delay.in");
        IntStream inputStreamD = inReader.getIntStream("d");
        UnitStream inputStreamR = inReader.getUnitStream("r");
        Reader outReader = Reader("../test/data/bt_delay.out");
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
        // merge(...) // TODO: Implement
        outputStream.copy_to_host();

        // Compare kernel result to correct data
        std::vector<int> kernelTimestamps(outputStream.host_timestamp, outputStream.host_timestamp + sizeof(outputStream.host_timestamp) / sizeof(int));
        std::vector<int> correctTimestamps(CORRECT_STREAM.host_timestamp, CORRECT_STREAM.host_timestamp + sizeof(CORRECT_STREAM.host_timestamp) / sizeof(int));
        REQUIRE(kernelTimestamps == correctTimestamps);

        // Cleanup
        inputStreamD.free_device();
        inputStreamR.free_device();
        outputStream.free_device();
        free(host_timestampOut);
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
        REQUIRE(kernelTimestamps == correctTimestamps);
        REQUIRE(kernelValues == correctValues);

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
            int sizeAllocated = (size_t) size * sizeof(int);
            int *host_timestampOut = (int *) malloc(size * sizeof(int));
            int *host_valueOut = (int*) malloc(size * sizeof(int));
            memset(host_timestampOut, 0, sizeAllocated);
            memset(host_valueOut, 0, sizeAllocated);
            IntStream outputStream(host_timestampOut, host_valueOut, size);

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
            free(host_valueOut);
            free(host_timestampOut);
        }
        
        SECTION("time() with bigger dataset (~109k/250k events)") {
            // Read input and correct output data
            Reader inReader = Reader("../test/data/bt_time.bigger.in");
            IntStream inputStream = inReader.getIntStream("z");
            Reader outReader = Reader("../test/data/bt_time.bigger.out");
            IntStream CORRECT_STREAM = outReader.getIntStream("y");

            // Prepare empty output stream to fill
            int size = CORRECT_STREAM.size;
            int sizeAllocated = (size_t) size * sizeof(int);
            int *host_timestampOut = (int *) malloc(size * sizeof(int));
            int *host_valueOut = (int*) malloc(size * sizeof(int));
            memset(host_timestampOut, 0, sizeAllocated);
            memset(host_valueOut, 0, sizeAllocated);
            IntStream outputStream(host_timestampOut, host_valueOut, size);

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
            free(host_valueOut);
            free(host_timestampOut);
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
