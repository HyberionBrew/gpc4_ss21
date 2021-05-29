//
// Created by fabian on 28.05.21.
//
#include <cuda_runtime.h>
#include <sys/time.h>
#include "main.cuh"
#include "helper.cuh"
#include "Stream.cuh"
#include "StreamFunctions.cuh"
#include "device_information.cuh"
// https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#numa-best-practices
// ADD stream argument to enable multiple kernels in parallel (10.5. Concurrent Kernel Execution)
// Note:Low Medium Priority: Use signed integers rather than unsigned integers as loop counters.
void time(IntStream *input, IntStream *result,cudaStream_t stream){
    //already malloced on host at this time
    //are both streams allocated on the device?

    //TODO! asynchronous copying to the device could be done here!
    // check if already on device! if not copy it to device asynchronously and
    // launch kernels piecewise as in
    // https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#numa-best-practices 10.5

    //choose block sizes
    // spawn threads in increments of 32
    int threads = input->size;
    //shift block size until 1024 than shift block size until maximal? Do we have to schedule twice
    int block_size = 1;
    int blocks = 1;

    //cannot schedule all at once
    // 10.3. Thread and Block Heuristics
    // The number of threads per block should be a multiple of 32 threads
    if (MAX_BLOCKS*MAX_THREADS_PER_BLOCK<threads){
        printf("Cannot schedule the whole stream! TODO! implement iterative scheduling \n");
        //return;
    }

    for (int bs = 32; bs <= MAX_THREADS_PER_BLOCK;bs +=32){
        if (block_size > threads){
            break;
        }
        block_size = bs;
    }
    //TODO! MAX_BLOCKS?
    // the number of blocks per kernel launch should be in the thousands.
    for (int bl=1; bl <= MAX_BLOCKS*1000; bl++){
        blocks = bl;
        if (bl*block_size > threads){
            break;
        }
    }

    //the pointers are now surely on device
    time_cuda<<<blocks,block_size,0,stream>>>(input->device_timestamp, result->device_timestamp, result->device_values, threads);
    printf("Scheduled with <<<%d,%d>>> \n",blocks,block_size);

};

__global__ void time_cuda(int* input_timestamp, int* output_timestamps, int* output_values,int size){
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i<size){
        output_timestamps[i] = input_timestamp[i];
        output_values[i] = input_timestamp[i];
    }
}