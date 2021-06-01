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

    //create kernel memory

    //the pointers are now surely on device
    time_cuda<<<blocks,block_size,0,stream>>>(input->device_timestamp, result->device_timestamp, result->device_values, threads,input->device_offset,result->device_offset);


    //kernel free
    printf("Scheduled time() with <<<%d,%d>>> \n",blocks,block_size);

};


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

    //TODO! make iterative! see last for hints
    if (*blocks > 1024){
        printf("Many Blocks");
        return;
    }
}

void last(IntStream *inputInt, UnitStream *inputUnit, IntStream *result, cudaStream_t stream){
    int threads = (int) inputUnit->size;
    int block_size =1;
    int blocks = 1;
    calcThreadsBlocks(threads,&block_size,&blocks);
    int* block_red;
    cudaMalloc((void**)&block_red, sizeof(int)*blocks);
    //TODO! check that no expection is thrown at launch!
    last_cuda<<<blocks,block_size,0,stream>>>(block_red, inputInt->device_timestamp, inputInt->device_values, inputUnit->device_timestamp,result->device_timestamp,result->device_values,inputInt->size, threads);
    int leftBlocks = blocks;
    //TODO! implement and check below functions! for schedulings > 1024 blocks
    /* while(leftBlocks>1024)
        calcThreadsBlocks(leftBlocks,&block_size,&blocks);
        reduce_blocks<<<blocks, block_size, 0, stream>>>(block_red, leftBlocks);
        leftBlocks = blocks;
    };*/
    final_reduce<<<1, block_size, 0, stream>>>(block_red, leftBlocks, result->device_offset);

    cudaFree(block_red);
    printf("Scheduled last() with <<<%d,%d>>> \n",blocks,block_size);
    printf("RESULT pointer: %d", result->device_offset);
}

__global__ void final_reduce(int* block_red,int size,int* offset) {
    __shared__ int sdata[1024];
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int tid = threadIdx.x;
    if (i < size) {
        sdata[tid] = block_red[i];
        __syncthreads();
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
            *offset = sdata[0];
            printf("The offset: %d \n",*offset);
        }
    }
}
//reduction example followed from: https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
__device__ void count_valid(int * sdata,int * output_timestamp,int* valid, int size, int MaxSize, unsigned int tid, const int i){
    //each thread loads one Element from global to shared memory
    sdata[tid] = 0;

    if (output_timestamp[i] < 0) {
        sdata[tid] = 1;
        //printf("%d ? %d\n",i,output_timestamp[i]);
    }
    __syncthreads();
    for (unsigned int s = (int)size / 2; s > 0; s >>= 1) {
        if (s < size){
            if (tid < s) {
                if ((i+s+1) > MaxSize){
                    sdata[tid] += 0;
                }
                else {
                    sdata[tid] += sdata[tid + s];
                }
            }
        }
        __syncthreads();
    }
    //result to array
    if (tid == 0) *valid=sdata[0];
}

//we should also hand this function the number of invalid input values! -> we have invalid values!
//TODO! check what happens for == and adjust >= or > accordingly (/remove else)
// wikipedia binary search: https://en.wikipedia.org/wiki/Binary_search_algorithm
__global__ void last_cuda(int* block_red, int* input_timestamp, int* input_values,int*unit_stream_timestamps,  int* output_timestamps, int* output_values, int intStreamSize, int size){
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
        //__syncthreads(); //should be unneeded
        block_red[blockIdx.x] = 0; //not really needed
        count_valid(sdata,output_timestamps,&block_red[blockIdx.x], 1024,size,tid,i);
    }

}

// working
__global__ void time_cuda(int* input_timestamp, int* output_timestamps, int* output_values,int size, int*offset, int* resultOffset){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= *offset && i<size){
        output_timestamps[i] = input_timestamp[i];
        output_values[i] = input_timestamp[i];
    }
    if (i == 0) *resultOffset = *offset;
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