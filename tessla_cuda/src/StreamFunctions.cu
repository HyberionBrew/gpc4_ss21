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

#define MAX_STREAMS 10
//Memory pointer for the streams
//TODO! not used
// implement pointer passed memory (i.e. not as it is done currently!)
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#dynamic-global-memory-allocation-and-operations
// DISCUSS HOW TO BEST DO THIS
// example https://forums.developer.nvidia.com/t/how-to-allocate-global-dynamic-memory-on-device-from-host/71011/2

__device__ int** streamTable[MAX_STREAMS]; // Per-stream pointer

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
    //TODO! check how many MAX_BLOCKS and
    for (int bl=1; bl <= MAX_BLOCKS*1000; bl++){
        blocks = bl;
        if (bl*block_size > threads){
            break;
        }
    }

    //the pointers are now surely on device
    time_cuda<<<blocks,block_size,0,stream>>>(input->device_timestamp, result->device_timestamp, result->device_values, threads);
    printf("Scheduled time() with <<<%d,%d>>> \n",blocks,block_size);

};


void last(IntStream *inputInt, UnitStream *inputUnit, IntStream *result, cudaStream_t stream){
    int threads = (int) inputUnit->size;
    int block_size = 1;
    int blocks = 1;
    if (MAX_BLOCKS*MAX_THREADS_PER_BLOCK<threads){
        printf("Cannot schedule the whole stream! TODO! implement iterative scheduling \n");
        //return;
    }
    //schedule in warp size
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
    //TODO! check that no expection is thrown at launch!
    last_cuda<<<blocks,block_size,0,stream>>>(inputInt->device_timestamp, inputInt->device_values, inputUnit->device_timestamp,result->device_timestamp,result->device_values, inputInt->size, threads);
    printf("Scheduled last() with <<<%d,%d>>> \n",blocks,block_size);
}

//reduction example followed from: https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
__device__ void count_valid(int * sdata,int * output_timestamp,int* valid, int size, unsigned int tid, const int i){
    //each thread loads one Element from global to shared memory
    if (i<size) {
        sdata[tid] = 0;
        printf("data %d \n",output_timestamp[i] );
        if (output_timestamp[i] < 0) {
            sdata[tid] = 1;
            printf("call\n");
        }
        for (unsigned int s = size / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }
        //write result of block atomically to global memory
        if (tid == 0) atomicAdd(valid,sdata[0]);
    }
}

//we should also hand this function the number of invalid input values! -> we have invalid values!
//TODO! check what happens for == and adjust >= or > accordingly (/remove else)
// wikipedia binary search: https://en.wikipedia.org/wiki/Binary_search_algorithm
__global__ void last_cuda(int* input_timestamp, int* input_values,int*unit_stream_timestamps,  int* output_timestamps, int* output_values, int intStreamSize, int size){
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int tid = threadIdx.x;
    int local_unit_timestamp = unit_stream_timestamps[i];
    __shared__ int sdata[1024];

   //printf("data %d \n",*(sdata));
    int L = 0;
    int R = intStreamSize-1;
    int m = 0;
    output_timestamps[i] = INT_MIN;
    int out =  INT_MIN;
    int* valid =(int*) malloc(sizeof(int));
    //TODO! implement more efficient version with local shared memory?
    if (i<size) {
        while (L<=R) {
            // is this needed? TODO! check and discuss
            //maybe it helps? CHECK!
           //__syncthreads();
            m = (int) (L+R)/2;
            if (input_timestamp[m]<local_unit_timestamp){
                L = m + 1;
                out = input_values[m];
                output_timestamps[i] = unit_stream_timestamps[i];
                //output_values[i] = input_values[m];
            }
            else if (input_timestamp[m]>=local_unit_timestamp){
                R = m -1;
            }
            else{
                // how to handle == ? look up!
                out = input_values[m];
                output_timestamps[i] = unit_stream_timestamps[i];
                break;
            }
        }
        output_values[i] = out;
        __syncthreads();
        count_valid(sdata,output_timestamps,valid, size,tid,i);

    }
    if (i == 0){
       printf(" valid %d \n", *valid);
    }
    free(valid);
}

// working
__global__ void time_cuda(int* input_timestamp, int* output_timestamps, int* output_values,int size){
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i<size){
        output_timestamps[i] = input_timestamp[i];
        output_values[i] = input_timestamp[i];
    }
}

__global__ void delay_cuda(int* input_timestamp, int* input_values,int*unit_stream_timestamps,  int* output_timestamps, int* output_values, int intStreamSize, int size){
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    //search for timestamps[i] value in unitEvents. If found:
    // counterTarget = input_values[i]
    // counter = 0
    // while (counter != counterTarget)
    //  counter++;
    //  if unitEvent at counter+
    //      reset //i.e. return
    //  if
    //
}