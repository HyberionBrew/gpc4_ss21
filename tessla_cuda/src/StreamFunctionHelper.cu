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
#include "StreamFunctionHelper.cuh"

void calcThreadsBlocks(int threads, int *block_size, int*blocks){
    *block_size = 1;
    *blocks = 1;
    if (MAX_BLOCKS*MAX_THREADS_PER_BLOCK<threads){
        printf("Cannot schedule the whole stream! TODO! implement iterative scheduling \n");
        //return;
    }
    //schedule in warp size
    for (int bs = 32; bs <= MAX_THREADS_PER_BLOCK;bs +=32){
        if (*block_size > threads){
            break;
        }
        *block_size = bs;
    }
    //TODO! MAX_BLOCKS?
    // the number of blocks per kernel launch should be in the thousands.
    for (int bl=1; bl <= MAX_BLOCKS*1000; bl++){
        *blocks = bl;
        if (bl* (*block_size) > threads){
            break;
        }
    }

    //TODO! make iterative! see last for hints (code already there)
    if (*blocks > 1024){
        printf("Many Blocks");
        return;
    }
}

__global__ void final_reduce(int* block_red,int size,int* offset) {
    __shared__ int sdata[1024];
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int tid = threadIdx.x;
    if (i < size) {
        sdata[tid] = block_red[i];
        for (unsigned int s = (int)1024 / 2; s > 0; s >>= 1) {
            if (s < size){
                if (tid < s) {
                    if ((i+s+1) > size){
                        sdata[tid] += 0;
                    }
                    else {
                        sdata[tid] += sdata[tid + s];
                    }
                }
            }
            __syncthreads();
        }

        if (i == 0){
            printf("The offset: %d \n",*offset);
        }
    }
}



