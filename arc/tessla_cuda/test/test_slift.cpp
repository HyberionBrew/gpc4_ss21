//
// Created by fabian on 24/06/2021.
//

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
#include <stdio.h>

TEST_CASE("slift_regular"){
    SECTION("slift(merge) simple example") {
        // Read input and correct output data
        std::cout << typeid(thrust::plus<int>()).name() << std::endl;
        GPUReader inReader = GPUReader("test/data/slift1.in");
        std::shared_ptr<GPUIntStream> inputStreamZ = inReader.getIntStreamDebug("z");
        std::shared_ptr<GPUIntStream> inputStreamY = inReader.getIntStreamDebug("y");
        GPUReader outReader = GPUReader("test/data/slift1_merge.out");
        std::shared_ptr<GPUIntStream> CORRECT_STREAM = outReader.getIntStreamDebug("f");

        // Prepare empty output stream to fill
        int size = CORRECT_STREAM->size;
        std::shared_ptr<GPUIntStream> outputStream;
        // Run kernel
        inputStreamZ->copy_to_device();
        inputStreamY->copy_to_device();
       // inputStreamV.print();
      //  inputStreamR.print();
        outputStream = slift(inputStreamZ, inputStreamY, MRG);
        //inputStreamR.print();
        outputStream->copy_to_host();
        outputStream->print();
        // Compare kernel result to correct data
        std::vector<int> kernelTimestamps(outputStream->host_timestamp+*(outputStream->host_offset), outputStream->host_timestamp+outputStream->size);
        std::vector<int> kernelValues(outputStream->host_values+*(outputStream->host_offset), outputStream->host_values+outputStream->size);
        std::vector<int> correctTimestamps(CORRECT_STREAM->host_timestamp, CORRECT_STREAM->host_timestamp + CORRECT_STREAM->size);
        std::vector<int> correctValues(CORRECT_STREAM->host_values, CORRECT_STREAM->host_values + CORRECT_STREAM->size);

        REQUIRE(kernelTimestamps == correctTimestamps);
        REQUIRE(kernelValues == correctValues);
        //outputStream->print();
        // Cleanup
        inputStreamZ->free_device();

        inputStreamY->free_device();
        outputStream->free_device();
        
        inputStreamZ->free_host();
        outputStream->free_host();
        inputStreamY->free_host();
    }


    SECTION("slift(+) simple example") {
        // Read input and correct output data
        GPUReader inReader = GPUReader("test/data/slift1.in");
        std::shared_ptr<GPUIntStream> inputStreamZ = inReader.getIntStreamDebug("z");
        std::shared_ptr<GPUIntStream> inputStreamY = inReader.getIntStreamDebug("y");
        GPUReader outReader = GPUReader("test/data/slift1.out");
        std::shared_ptr<GPUIntStream> CORRECT_STREAM = outReader.getIntStreamDebug("f");

        // Prepare empty output stream to fill
        int size = CORRECT_STREAM->size;
        std::shared_ptr<GPUIntStream> outputStream;
        // Run kernel
        inputStreamZ->copy_to_device();
        inputStreamY->copy_to_device();
       // inputStreamV.print();
      //  inputStreamR.print();
        outputStream = slift(inputStreamZ, inputStreamY, ADD);
        //inputStreamR.print();
        outputStream->copy_to_host();
        //outputStream.print();
        // Compare kernel result to correct data
        std::vector<int> kernelTimestamps(outputStream->host_timestamp+*(outputStream->host_offset), outputStream->host_timestamp+outputStream->size);
        std::vector<int> kernelValues(outputStream->host_values+*(outputStream->host_offset), outputStream->host_values+outputStream->size);
        std::vector<int> correctTimestamps(CORRECT_STREAM->host_timestamp, CORRECT_STREAM->host_timestamp + CORRECT_STREAM->size);
        std::vector<int> correctValues(CORRECT_STREAM->host_values, CORRECT_STREAM->host_values + CORRECT_STREAM->size);

        REQUIRE(kernelTimestamps == correctTimestamps);
        REQUIRE(kernelValues == correctValues);
        //outputStream->print();
        // Cleanup
        inputStreamZ->free_device();

        inputStreamY->free_device();
        outputStream->free_device();
        
        inputStreamZ->free_host();
        outputStream->free_host();
        inputStreamY->free_host();
    }

    SECTION("slift(-) simple example") {
        // Read input and correct output data
        GPUReader inReader = GPUReader("test/data/slift1.in");
        std::shared_ptr<GPUIntStream> inputStreamZ = inReader.getIntStreamDebug("z");
        std::shared_ptr<GPUIntStream> inputStreamY = inReader.getIntStreamDebug("y");
        GPUReader outReader = GPUReader("test/data/slift1_minus.out");
        std::shared_ptr<GPUIntStream> CORRECT_STREAM = outReader.getIntStreamDebug("f");

        // Prepare empty output stream to fill
        int size = CORRECT_STREAM->size;
        std::shared_ptr<GPUIntStream> outputStream;
        // Run kernel
        inputStreamZ->copy_to_device();
        inputStreamY->copy_to_device();
       // inputStreamV.print();
      //  inputStreamR.print();
        outputStream = slift(inputStreamZ, inputStreamY, SUB);
        //inputStreamR.print();
        outputStream->copy_to_host();
        //outputStream->print();
        //outputStream.print();
        // Compare kernel result to correct data
        std::vector<int> kernelTimestamps(outputStream->host_timestamp+*(outputStream->host_offset), outputStream->host_timestamp+outputStream->size);
        std::vector<int> kernelValues(outputStream->host_values+*(outputStream->host_offset), outputStream->host_values+outputStream->size);
        std::vector<int> correctTimestamps(CORRECT_STREAM->host_timestamp, CORRECT_STREAM->host_timestamp + CORRECT_STREAM->size);
        std::vector<int> correctValues(CORRECT_STREAM->host_values, CORRECT_STREAM->host_values + CORRECT_STREAM->size);

        REQUIRE(kernelTimestamps == correctTimestamps);
        REQUIRE(kernelValues == correctValues);
        //outputStream->print();
        // Cleanup
        inputStreamZ->free_device();

        inputStreamY->free_device();
        outputStream->free_device();
        
        inputStreamZ->free_host();
        outputStream->free_host();
        inputStreamY->free_host();
    }


        SECTION("slift(/) simple example") {
        // Read input and correct output data
        GPUReader inReader = GPUReader("test/data/slift1.in");
        std::shared_ptr<GPUIntStream> inputStreamZ = inReader.getIntStreamDebug("z");
        std::shared_ptr<GPUIntStream> inputStreamY = inReader.getIntStreamDebug("y");
        GPUReader outReader = GPUReader("test/data/slift1_div.out");

        std::shared_ptr<GPUIntStream> CORRECT_STREAM = outReader.getIntStreamDebug("f");

        // Prepare empty output stream to fill
        int size = CORRECT_STREAM->size;
        std::shared_ptr<GPUIntStream> outputStream;
        // Run kernel
        inputStreamZ->copy_to_device();
        inputStreamY->copy_to_device();
       // inputStreamV.print();
      //  inputStreamR.print();
        outputStream = slift(inputStreamZ, inputStreamY, DIV);
        //inputStreamR.print();
        outputStream->copy_to_host();
        outputStream->print();
        //outputStream.print();
        // Compare kernel result to correct data
        std::vector<int> kernelTimestamps(outputStream->host_timestamp+*(outputStream->host_offset), outputStream->host_timestamp+outputStream->size);
        std::vector<int> kernelValues(outputStream->host_values+*(outputStream->host_offset), outputStream->host_values+outputStream->size);
        std::vector<int> correctTimestamps(CORRECT_STREAM->host_timestamp, CORRECT_STREAM->host_timestamp + CORRECT_STREAM->size);
        std::vector<int> correctValues(CORRECT_STREAM->host_values, CORRECT_STREAM->host_values + CORRECT_STREAM->size);

        REQUIRE(kernelTimestamps == correctTimestamps);
        REQUIRE(kernelValues == correctValues);
        //outputStream->print();
        // Cleanup
        inputStreamZ->free_device();

        inputStreamY->free_device();
        outputStream->free_device();
        
        inputStreamZ->free_host();
        outputStream->free_host();
        inputStreamY->free_host();
    }


        SECTION("slift(/) simple example with invalids") {
        // Read input and correct output data
        
        GPUReader inReader = GPUReader("test/data/slift1_div.in");
        std::shared_ptr<GPUIntStream> inputStreamZ = inReader.getIntStreamDebug("z");
        std::shared_ptr<GPUIntStream> inputStreamY = inReader.getIntStreamDebug("y");
        GPUReader outReader = GPUReader("test/data/slift1_div_inv.out");
        std::shared_ptr<GPUIntStream> CORRECT_STREAM = outReader.getIntStreamDebug("f");

        // Prepare empty output stream to fill
        int size = CORRECT_STREAM->size;
        std::shared_ptr<GPUIntStream> outputStream;
        // Run kernel
        inputStreamZ->copy_to_device();
        inputStreamY->copy_to_device();
       // inputStreamV.print();
      //  inputStreamR.print();
      try
        {
        outputStream = slift(inputStreamZ, inputStreamY, DIV);
        //inputStreamR.print();
        outputStream->copy_to_host();
        //outputStream->print();
        //outputStream.print();
        // Compare kernel result to correct data
        std::vector<int> kernelTimestamps(outputStream->host_timestamp+*(outputStream->host_offset), outputStream->host_timestamp+outputStream->size);
        std::vector<int> kernelValues(outputStream->host_values+*(outputStream->host_offset), outputStream->host_values+outputStream->size);
        std::vector<int> correctTimestamps(CORRECT_STREAM->host_timestamp, CORRECT_STREAM->host_timestamp + CORRECT_STREAM->size);
        std::vector<int> correctValues(CORRECT_STREAM->host_values, CORRECT_STREAM->host_values + CORRECT_STREAM->size);

        REQUIRE(kernelTimestamps == correctTimestamps);
        REQUIRE(kernelValues == correctValues);
        //outputStream->print();
        // Cleanup
        outputStream->free_device();
        outputStream->free_host();
        REQUIRE(false);
        }
       catch(std::runtime_error& e){
           printf("Caught Runtime error \n");
       }
        inputStreamZ->free_device();

        inputStreamY->free_device();
        
        
        inputStreamZ->free_host();
        inputStreamY->free_host();
    }

        SECTION("slift(+) simple example 2") {
        // Read input and correct output data
        //printf("----slift 2-----");
        GPUReader inReader = GPUReader("test/data/slift2.in");
        std::shared_ptr<GPUIntStream> inputStreamZ = inReader.getIntStreamDebug("z");
        std::shared_ptr<GPUIntStream> inputStreamY = inReader.getIntStreamDebug("y");
        GPUReader outReader = GPUReader("test/data/slift2.out");
        std::shared_ptr<GPUIntStream> CORRECT_STREAM = outReader.getIntStreamDebug("f");

        // Prepare empty output stream to fill
        int size = CORRECT_STREAM->size;
        std::shared_ptr<GPUIntStream> outputStream;
        // Run kernel
        inputStreamZ->copy_to_device();
        inputStreamY->copy_to_device();
       // inputStreamV.print();
      //  inputStreamR.print();
        outputStream = slift(inputStreamZ, inputStreamY, ADD);
        //inputStreamR.print();
        outputStream->copy_to_host();
        //outputStream.print();
        // Compare kernel result to correct data
        std::vector<int> kernelTimestamps(outputStream->host_timestamp+*(outputStream->host_offset), outputStream->host_timestamp+outputStream->size);
        std::vector<int> kernelValues(outputStream->host_values+*(outputStream->host_offset), outputStream->host_values+outputStream->size);
        std::vector<int> correctTimestamps(CORRECT_STREAM->host_timestamp, CORRECT_STREAM->host_timestamp + CORRECT_STREAM->size);
        std::vector<int> correctValues(CORRECT_STREAM->host_values, CORRECT_STREAM->host_values + CORRECT_STREAM->size);

        REQUIRE(kernelTimestamps == correctTimestamps);
        REQUIRE(kernelValues == correctValues);
        //outputStream->print();
        // Cleanup
        inputStreamZ->free_device();

        inputStreamY->free_device();
        outputStream->free_device();
        
        inputStreamZ->free_host();
        outputStream->free_host();
        inputStreamY->free_host();
    }

SECTION("slift(+) simple example 2 flipped") {
        // Read input and correct output data
        GPUReader inReader = GPUReader("test/data/slift2.in");
        std::shared_ptr<GPUIntStream> inputStreamZ = inReader.getIntStreamDebug("z");
        std::shared_ptr<GPUIntStream> inputStreamY = inReader.getIntStreamDebug("y");
        GPUReader outReader = GPUReader("test/data/slift2.out");
        std::shared_ptr<GPUIntStream> CORRECT_STREAM = outReader.getIntStreamDebug("f");

        // Prepare empty output stream to fill
        int size = CORRECT_STREAM->size;
        std::shared_ptr<GPUIntStream> outputStream;
        // Run kernel
        inputStreamZ->copy_to_device();
        inputStreamY->copy_to_device();
       // inputStreamV.print();
      //  inputStreamR.print();
        outputStream = slift(inputStreamY, inputStreamZ, ADD);
        //inputStreamR.print();
        outputStream->copy_to_host();
        //outputStream->print();
        // Compare kernel result to correct data
        std::vector<int> kernelTimestamps(outputStream->host_timestamp+*(outputStream->host_offset), outputStream->host_timestamp+outputStream->size);
        std::vector<int> kernelValues(outputStream->host_values+*(outputStream->host_offset), outputStream->host_values+outputStream->size);
        std::vector<int> correctTimestamps(CORRECT_STREAM->host_timestamp, CORRECT_STREAM->host_timestamp + CORRECT_STREAM->size);
        std::vector<int> correctValues(CORRECT_STREAM->host_values, CORRECT_STREAM->host_values + CORRECT_STREAM->size);

        REQUIRE(kernelTimestamps == correctTimestamps);
        REQUIRE(kernelValues == correctValues);
        //outputStream->print();
        // Cleanup
        inputStreamZ->free_device();

        inputStreamY->free_device();
        outputStream->free_device();
        
        inputStreamZ->free_host();
        outputStream->free_host();
        inputStreamY->free_host();
    }
    SECTION("slift(+) empty example") {
        // Read input and correct output data
        GPUReader inReader = GPUReader("test/data/slift3.in");
        std::shared_ptr<GPUIntStream> inputStreamZ = inReader.getIntStreamDebug("z");
        std::shared_ptr<GPUIntStream> inputStreamY = inReader.getIntStreamDebug("y");
        GPUReader outReader = GPUReader("test/data/slift3.out");
        std::shared_ptr<GPUIntStream> CORRECT_STREAM = outReader.getIntStreamDebug("f");

        // Prepare empty output stream to fill
        int size = CORRECT_STREAM->size;
        std::shared_ptr<GPUIntStream> outputStream;
        // Run kernel
        inputStreamZ->copy_to_device();
        inputStreamY->copy_to_device();
       // inputStreamV.print();
      //  inputStreamR.print();
        outputStream = slift(inputStreamY, inputStreamZ, ADD);
        //inputStreamR.print();
        outputStream->copy_to_host();
        //outputStream.print();
        // Compare kernel result to correct data
        std::vector<int> kernelTimestamps(outputStream->host_timestamp+*(outputStream->host_offset), outputStream->host_timestamp+outputStream->size);
        std::vector<int> kernelValues(outputStream->host_values+*(outputStream->host_offset), outputStream->host_values+outputStream->size);
        std::vector<int> correctTimestamps(CORRECT_STREAM->host_timestamp, CORRECT_STREAM->host_timestamp + CORRECT_STREAM->size);
        std::vector<int> correctValues(CORRECT_STREAM->host_values, CORRECT_STREAM->host_values + CORRECT_STREAM->size);

        REQUIRE(kernelTimestamps == correctTimestamps);
        REQUIRE(kernelValues == correctValues);
        //outputStream->print();
        // Cleanup
        inputStreamZ->free_device();

        inputStreamY->free_device();
        outputStream->free_device();
        
        inputStreamZ->free_host();
        outputStream->free_host();
        inputStreamY->free_host();
    }

       SECTION("slift(+) large example") {
        // Read input and correct output data
        GPUReader inReader = GPUReader("test/data/slift4.in");
        std::shared_ptr<GPUIntStream> inputStreamZ = inReader.getIntStreamDebug("z");
        std::shared_ptr<GPUIntStream> inputStreamY = inReader.getIntStreamDebug("y");
        GPUReader outReader = GPUReader("test/data/slift4.out");
        std::shared_ptr<GPUIntStream> CORRECT_STREAM = outReader.getIntStreamDebug("f");

        // Prepare empty output stream to fill
        int size = CORRECT_STREAM->size;
        std::shared_ptr<GPUIntStream> outputStream;
        // Run kernel
        inputStreamZ->copy_to_device();
        inputStreamY->copy_to_device();
       // inputStreamV.print();
      //  inputStreamR.print();
        outputStream = slift(inputStreamY, inputStreamZ, ADD);
        //inputStreamR.print();
        outputStream->copy_to_host();
        //outputStream->print();
        
        // Compare kernel result to correct data
        std::vector<int> kernelTimestamps(outputStream->host_timestamp+*(outputStream->host_offset), outputStream->host_timestamp+outputStream->size);
        std::vector<int> kernelValues(outputStream->host_values+*(outputStream->host_offset), outputStream->host_values+outputStream->size);
        std::vector<int> correctTimestamps(CORRECT_STREAM->host_timestamp, CORRECT_STREAM->host_timestamp + CORRECT_STREAM->size);
        std::vector<int> correctValues(CORRECT_STREAM->host_values, CORRECT_STREAM->host_values + CORRECT_STREAM->size);

        REQUIRE(kernelTimestamps == correctTimestamps);
        REQUIRE(kernelValues == correctValues);
        //outputStream->print();
        // Cleanup
        inputStreamZ->free_device();

        inputStreamY->free_device();
        outputStream->free_device();
        
        inputStreamZ->free_host();
        outputStream->free_host();
        inputStreamY->free_host();
    }
}

//////////////////
// Thrust slift //
//////////////////

TEST_CASE("slift_thrust()"){
    SECTION("slift(merge) simple example") {
        // Read input and correct output data
        std::cout << typeid(thrust::plus<int>()).name() << std::endl;
        GPUReader inReader = GPUReader("test/data/slift1.in");
        std::shared_ptr<GPUIntStream> inputStreamZ = inReader.getIntStreamDebug("z");
        std::shared_ptr<GPUIntStream> inputStreamY = inReader.getIntStreamDebug("y");
        GPUReader outReader = GPUReader("test/data/slift1_merge.out");
        std::shared_ptr<GPUIntStream> CORRECT_STREAM = outReader.getIntStreamDebug("f");

        // Prepare empty output stream to fill
        int size = CORRECT_STREAM->size;
        std::shared_ptr<GPUIntStream> outputStream;
        // Run kernel
        inputStreamZ->copy_to_device();
        inputStreamY->copy_to_device();
       // inputStreamV.print();
      //  inputStreamR.print();
        outputStream = slift_thrust(inputStreamZ, inputStreamY,TH_OP_merge, 0);
        //inputStreamR.print();
        outputStream->copy_to_host();
        outputStream->print();
        // Compare kernel result to correct data
        std::vector<int> kernelTimestamps(outputStream->host_timestamp+*(outputStream->host_offset), outputStream->host_timestamp+outputStream->size);
        std::vector<int> kernelValues(outputStream->host_values+*(outputStream->host_offset), outputStream->host_values+outputStream->size);
        std::vector<int> correctTimestamps(CORRECT_STREAM->host_timestamp, CORRECT_STREAM->host_timestamp + CORRECT_STREAM->size);
        std::vector<int> correctValues(CORRECT_STREAM->host_values, CORRECT_STREAM->host_values + CORRECT_STREAM->size);

        REQUIRE(kernelTimestamps == correctTimestamps);
        REQUIRE(kernelValues == correctValues);
        //outputStream->print();
        // Cleanup
        inputStreamZ->free_device();

        inputStreamY->free_device();
        outputStream->free_device();
        
        inputStreamZ->free_host();
        outputStream->free_host();
        inputStreamY->free_host();
    }


    SECTION("slift(+) simple example") {
        // Read input and correct output data
        GPUReader inReader = GPUReader("test/data/slift1.in");
        std::shared_ptr<GPUIntStream> inputStreamZ = inReader.getIntStreamDebug("z");
        std::shared_ptr<GPUIntStream> inputStreamY = inReader.getIntStreamDebug("y");
        GPUReader outReader = GPUReader("test/data/slift1.out");
        std::shared_ptr<GPUIntStream> CORRECT_STREAM = outReader.getIntStreamDebug("f");

        // Prepare empty output stream to fill
        int size = CORRECT_STREAM->size;
        std::shared_ptr<GPUIntStream> outputStream;
        // Run kernel
        inputStreamZ->copy_to_device();
        inputStreamY->copy_to_device();
       // inputStreamV.print();
      //  inputStreamR.print();
        outputStream = slift_thrust(inputStreamZ, inputStreamY,TH_OP_add, 0);
        //inputStreamR.print();
        outputStream->copy_to_host();
        //outputStream.print();
        // Compare kernel result to correct data
        std::vector<int> kernelTimestamps(outputStream->host_timestamp+*(outputStream->host_offset), outputStream->host_timestamp+outputStream->size);
        std::vector<int> kernelValues(outputStream->host_values+*(outputStream->host_offset), outputStream->host_values+outputStream->size);
        std::vector<int> correctTimestamps(CORRECT_STREAM->host_timestamp, CORRECT_STREAM->host_timestamp + CORRECT_STREAM->size);
        std::vector<int> correctValues(CORRECT_STREAM->host_values, CORRECT_STREAM->host_values + CORRECT_STREAM->size);

        REQUIRE(kernelTimestamps == correctTimestamps);
        REQUIRE(kernelValues == correctValues);
        //outputStream->print();
        // Cleanup
        inputStreamZ->free_device();

        inputStreamY->free_device();
        outputStream->free_device();
        
        inputStreamZ->free_host();
        outputStream->free_host();
        inputStreamY->free_host();
    }

    SECTION("slift(-) simple example") {
        // Read input and correct output data
        GPUReader inReader = GPUReader("test/data/slift1.in");
        std::shared_ptr<GPUIntStream> inputStreamZ = inReader.getIntStreamDebug("z");
        std::shared_ptr<GPUIntStream> inputStreamY = inReader.getIntStreamDebug("y");
        GPUReader outReader = GPUReader("test/data/slift1_minus.out");
        std::shared_ptr<GPUIntStream> CORRECT_STREAM = outReader.getIntStreamDebug("f");

        // Prepare empty output stream to fill
        int size = CORRECT_STREAM->size;
        std::shared_ptr<GPUIntStream> outputStream;
        // Run kernel
        inputStreamZ->copy_to_device();
        inputStreamY->copy_to_device();
       // inputStreamV.print();
      //  inputStreamR.print();
        outputStream = slift_thrust(inputStreamZ, inputStreamY, TH_OP_subtract, 0);
        //inputStreamR.print();
        outputStream->copy_to_host();
        //outputStream->print();
        //outputStream.print();
        // Compare kernel result to correct data
        std::vector<int> kernelTimestamps(outputStream->host_timestamp+*(outputStream->host_offset), outputStream->host_timestamp+outputStream->size);
        std::vector<int> kernelValues(outputStream->host_values+*(outputStream->host_offset), outputStream->host_values+outputStream->size);
        std::vector<int> correctTimestamps(CORRECT_STREAM->host_timestamp, CORRECT_STREAM->host_timestamp + CORRECT_STREAM->size);
        std::vector<int> correctValues(CORRECT_STREAM->host_values, CORRECT_STREAM->host_values + CORRECT_STREAM->size);

        REQUIRE(kernelTimestamps == correctTimestamps);
        REQUIRE(kernelValues == correctValues);
        //outputStream->print();
        // Cleanup
        inputStreamZ->free_device();

        inputStreamY->free_device();
        outputStream->free_device();
        
        inputStreamZ->free_host();
        outputStream->free_host();
        inputStreamY->free_host();
    }


        SECTION("slift(/) simple example") {
        // Read input and correct output data
        GPUReader inReader = GPUReader("test/data/slift1.in");
        std::shared_ptr<GPUIntStream> inputStreamZ = inReader.getIntStreamDebug("z");
        std::shared_ptr<GPUIntStream> inputStreamY = inReader.getIntStreamDebug("y");
        GPUReader outReader = GPUReader("test/data/slift1_div.out");

        std::shared_ptr<GPUIntStream> CORRECT_STREAM = outReader.getIntStreamDebug("f");

        // Prepare empty output stream to fill
        int size = CORRECT_STREAM->size;
        std::shared_ptr<GPUIntStream> outputStream;
        // Run kernel
        inputStreamZ->copy_to_device();
        inputStreamY->copy_to_device();
       // inputStreamV.print();
      //  inputStreamR.print();
        outputStream = slift_thrust(inputStreamZ, inputStreamY,TH_OP_divide, 0);
        //inputStreamR.print();
        outputStream->copy_to_host();
        outputStream->print();
        //outputStream.print();
        // Compare kernel result to correct data
        std::vector<int> kernelTimestamps(outputStream->host_timestamp+*(outputStream->host_offset), outputStream->host_timestamp+outputStream->size);
        std::vector<int> kernelValues(outputStream->host_values+*(outputStream->host_offset), outputStream->host_values+outputStream->size);
        std::vector<int> correctTimestamps(CORRECT_STREAM->host_timestamp, CORRECT_STREAM->host_timestamp + CORRECT_STREAM->size);
        std::vector<int> correctValues(CORRECT_STREAM->host_values, CORRECT_STREAM->host_values + CORRECT_STREAM->size);

        REQUIRE(kernelTimestamps == correctTimestamps);
        REQUIRE(kernelValues == correctValues);
        //outputStream->print();
        // Cleanup
        inputStreamZ->free_device();

        inputStreamY->free_device();
        outputStream->free_device();
        
        inputStreamZ->free_host();
        outputStream->free_host();
        inputStreamY->free_host();
    }


        SECTION("slift(/) simple example with invalids") {
        // Read input and correct output data
        
        GPUReader inReader = GPUReader("test/data/slift1_div.in");
        std::shared_ptr<GPUIntStream> inputStreamZ = inReader.getIntStreamDebug("z");
        std::shared_ptr<GPUIntStream> inputStreamY = inReader.getIntStreamDebug("y");
        GPUReader outReader = GPUReader("test/data/slift1_div_inv.out");
        std::shared_ptr<GPUIntStream> CORRECT_STREAM = outReader.getIntStreamDebug("f");

        // Prepare empty output stream to fill
        int size = CORRECT_STREAM->size;
        std::shared_ptr<GPUIntStream> outputStream;
        // Run kernel
        inputStreamZ->copy_to_device();
        inputStreamY->copy_to_device();
       // inputStreamV.print();
      //  inputStreamR.print();
      try
        {
        outputStream = slift_thrust(inputStreamZ, inputStreamY,TH_OP_divide, 0);
        //inputStreamR.print();
        outputStream->copy_to_host();
        //outputStream->print();
        //outputStream.print();
        // Compare kernel result to correct data
        std::vector<int> kernelTimestamps(outputStream->host_timestamp+*(outputStream->host_offset), outputStream->host_timestamp+outputStream->size);
        std::vector<int> kernelValues(outputStream->host_values+*(outputStream->host_offset), outputStream->host_values+outputStream->size);
        std::vector<int> correctTimestamps(CORRECT_STREAM->host_timestamp, CORRECT_STREAM->host_timestamp + CORRECT_STREAM->size);
        std::vector<int> correctValues(CORRECT_STREAM->host_values, CORRECT_STREAM->host_values + CORRECT_STREAM->size);

        REQUIRE(kernelTimestamps == correctTimestamps);
        REQUIRE(kernelValues == correctValues);
        //outputStream->print();
        // Cleanup
        outputStream->free_device();
        outputStream->free_host();
        REQUIRE(false);
        }
       catch(std::runtime_error& e){
           printf("Caught Runtime error \n");
       }
        inputStreamZ->free_device();

        inputStreamY->free_device();
        
        
        inputStreamZ->free_host();
        inputStreamY->free_host();
    }

        SECTION("slift(+) simple example 2") {
        // Read input and correct output data
        //printf("----slift 2-----");
        GPUReader inReader = GPUReader("test/data/slift2.in");
        std::shared_ptr<GPUIntStream> inputStreamZ = inReader.getIntStreamDebug("z");
        std::shared_ptr<GPUIntStream> inputStreamY = inReader.getIntStreamDebug("y");
        GPUReader outReader = GPUReader("test/data/slift2.out");
        std::shared_ptr<GPUIntStream> CORRECT_STREAM = outReader.getIntStreamDebug("f");

        // Prepare empty output stream to fill
        int size = CORRECT_STREAM->size;
        std::shared_ptr<GPUIntStream> outputStream;
        // Run kernel
        inputStreamZ->copy_to_device();
        inputStreamY->copy_to_device();
       // inputStreamV.print();
      //  inputStreamR.print();
        outputStream = slift_thrust(inputStreamZ, inputStreamY,TH_OP_add, 0);
        //inputStreamR.print();
        outputStream->copy_to_host();
        //outputStream.print();
        // Compare kernel result to correct data
        std::vector<int> kernelTimestamps(outputStream->host_timestamp+*(outputStream->host_offset), outputStream->host_timestamp+outputStream->size);
        std::vector<int> kernelValues(outputStream->host_values+*(outputStream->host_offset), outputStream->host_values+outputStream->size);
        std::vector<int> correctTimestamps(CORRECT_STREAM->host_timestamp, CORRECT_STREAM->host_timestamp + CORRECT_STREAM->size);
        std::vector<int> correctValues(CORRECT_STREAM->host_values, CORRECT_STREAM->host_values + CORRECT_STREAM->size);

        REQUIRE(kernelTimestamps == correctTimestamps);
        REQUIRE(kernelValues == correctValues);
        //outputStream->print();
        // Cleanup
        inputStreamZ->free_device();

        inputStreamY->free_device();
        outputStream->free_device();
        
        inputStreamZ->free_host();
        outputStream->free_host();
        inputStreamY->free_host();
    }

SECTION("slift(+) simple example 2 flipped") {
        // Read input and correct output data
        GPUReader inReader = GPUReader("test/data/slift2.in");
        std::shared_ptr<GPUIntStream> inputStreamZ = inReader.getIntStreamDebug("z");
        std::shared_ptr<GPUIntStream> inputStreamY = inReader.getIntStreamDebug("y");
        GPUReader outReader = GPUReader("test/data/slift2.out");
        std::shared_ptr<GPUIntStream> CORRECT_STREAM = outReader.getIntStreamDebug("f");

        // Prepare empty output stream to fill
        int size = CORRECT_STREAM->size;
        std::shared_ptr<GPUIntStream> outputStream;
        // Run kernel
        inputStreamZ->copy_to_device();
        inputStreamY->copy_to_device();
       // inputStreamV.print();
      //  inputStreamR.print();
        outputStream = slift_thrust(inputStreamY, inputStreamZ,TH_OP_add, 0);
        //inputStreamR.print();
        outputStream->copy_to_host();
        //outputStream->print();
        // Compare kernel result to correct data
        std::vector<int> kernelTimestamps(outputStream->host_timestamp+*(outputStream->host_offset), outputStream->host_timestamp+outputStream->size);
        std::vector<int> kernelValues(outputStream->host_values+*(outputStream->host_offset), outputStream->host_values+outputStream->size);
        std::vector<int> correctTimestamps(CORRECT_STREAM->host_timestamp, CORRECT_STREAM->host_timestamp + CORRECT_STREAM->size);
        std::vector<int> correctValues(CORRECT_STREAM->host_values, CORRECT_STREAM->host_values + CORRECT_STREAM->size);

        REQUIRE(kernelTimestamps == correctTimestamps);
        REQUIRE(kernelValues == correctValues);
        //outputStream->print();
        // Cleanup
        inputStreamZ->free_device();

        inputStreamY->free_device();
        outputStream->free_device();
        
        inputStreamZ->free_host();
        outputStream->free_host();
        inputStreamY->free_host();
    }
    SECTION("slift(+) empty example") {
        // Read input and correct output data
        GPUReader inReader = GPUReader("test/data/slift3.in");
        std::shared_ptr<GPUIntStream> inputStreamZ = inReader.getIntStreamDebug("z");
        std::shared_ptr<GPUIntStream> inputStreamY = inReader.getIntStreamDebug("y");
        GPUReader outReader = GPUReader("test/data/slift3.out");
        std::shared_ptr<GPUIntStream> CORRECT_STREAM = outReader.getIntStreamDebug("f");

        // Prepare empty output stream to fill
        int size = CORRECT_STREAM->size;
        std::shared_ptr<GPUIntStream> outputStream;
        // Run kernel
        inputStreamZ->copy_to_device();
        inputStreamY->copy_to_device();
       // inputStreamV.print();
      //  inputStreamR.print();
        outputStream = slift_thrust(inputStreamY, inputStreamZ,TH_OP_add, 0);
        //inputStreamR.print();
        outputStream->copy_to_host();
        //outputStream.print();
        // Compare kernel result to correct data
        std::vector<int> kernelTimestamps(outputStream->host_timestamp+*(outputStream->host_offset), outputStream->host_timestamp+outputStream->size);
        std::vector<int> kernelValues(outputStream->host_values+*(outputStream->host_offset), outputStream->host_values+outputStream->size);
        std::vector<int> correctTimestamps(CORRECT_STREAM->host_timestamp, CORRECT_STREAM->host_timestamp + CORRECT_STREAM->size);
        std::vector<int> correctValues(CORRECT_STREAM->host_values, CORRECT_STREAM->host_values + CORRECT_STREAM->size);

        REQUIRE(kernelTimestamps == correctTimestamps);
        REQUIRE(kernelValues == correctValues);
        //outputStream->print();
        // Cleanup
        inputStreamZ->free_device();

        inputStreamY->free_device();
        outputStream->free_device();
        
        inputStreamZ->free_host();
        outputStream->free_host();
        inputStreamY->free_host();
    }

       SECTION("slift(+) large example") {
        // Read input and correct output data
        GPUReader inReader = GPUReader("test/data/slift4.in");
        std::shared_ptr<GPUIntStream> inputStreamZ = inReader.getIntStreamDebug("z");
        std::shared_ptr<GPUIntStream> inputStreamY = inReader.getIntStreamDebug("y");
        GPUReader outReader = GPUReader("test/data/slift4.out");
        std::shared_ptr<GPUIntStream> CORRECT_STREAM = outReader.getIntStreamDebug("f");

        // Prepare empty output stream to fill
        int size = CORRECT_STREAM->size;
        std::shared_ptr<GPUIntStream> outputStream;
        // Run kernel
        inputStreamZ->copy_to_device();
        inputStreamY->copy_to_device();
       // inputStreamV.print();
      //  inputStreamR.print();
        outputStream = slift_thrust(inputStreamY, inputStreamZ,TH_OP_add, 0);
        //inputStreamR.print();
        outputStream->copy_to_host();
        //outputStream->print();
        
        // Compare kernel result to correct data
        std::vector<int> kernelTimestamps(outputStream->host_timestamp+*(outputStream->host_offset), outputStream->host_timestamp+outputStream->size);
        std::vector<int> kernelValues(outputStream->host_values+*(outputStream->host_offset), outputStream->host_values+outputStream->size);
        std::vector<int> correctTimestamps(CORRECT_STREAM->host_timestamp, CORRECT_STREAM->host_timestamp + CORRECT_STREAM->size);
        std::vector<int> correctValues(CORRECT_STREAM->host_values, CORRECT_STREAM->host_values + CORRECT_STREAM->size);

        REQUIRE(kernelTimestamps == correctTimestamps);
        REQUIRE(kernelValues == correctValues);
        //outputStream->print();
        // Cleanup
        inputStreamZ->free_device();

        inputStreamY->free_device();
        outputStream->free_device();
        
        inputStreamZ->free_host();
        outputStream->free_host();
        inputStreamY->free_host();
    }
}