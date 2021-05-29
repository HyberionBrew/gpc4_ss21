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

void time(IntStream *input, IntStream *result){
    //already malloced on host at this time
    //are both streams allocated on the device?

    //choose block sizes
    // spawn threads in increments of 32
    int threads = input->size;
    //shift block size until 1024 than shift block size until maximal? Do we have to schedule twice
    int block_size = 1;
    int blocks = 1;

    //cannot schedule all at once
    if (MAX_BLOCKS*MAX_THREADS_PER_BLOCK<threads){
        printf("Cannot schedule the whole stream! TODO! implement iterative scheduling \n");
        return;
    }
    for (int bs = 1; bs <= MAX_THREADS_PER_BLOCK;bs<<=1){
        if (block_size > threads){
            break;
        }
        block_size = bs;
    }

    for (int bl=1; bl <= MAX_BLOCKS; bl++){
        blocks = bl;
        if (bl*block_size > threads){
            break;
        }
    }

    //the pointers are now surely on device
    time_cuda<<<blocks,block_size>>>(input->device_timestamp, result->device_timestamp, result->device_values, threads);
    printf("Scheduled with <<<%d,%d>>> \n",blocks,block_size);

};

__global__ void time_cuda(int* input_timestamp, int* output_timestamps, int* output_values,int size){
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i<size){
        output_timestamps[i] = input_timestamp[i];
        output_values[i] = input_timestamp[i];
    }
}