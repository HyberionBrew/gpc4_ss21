//
// Created by fabian on 28.05.21.
//
#include <cuda_runtime.h>
#include <sys/time.h>
#include "main.cuh"
#include "helper.cuh"
#include "GPUStream.cuh"
#include "StreamFunctions.cuh"
#include "device_information.cuh"
#include "StreamFunctionHelper.cuh"

void calcThreadsBlocks(int threads, int *block_size, int*blocks){
    *block_size = 0;
    *blocks = 0;
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


// CUDA sorting algorithm implementations
// https://github.com/icaroharry/sort
// NVIDIA's
// Selection sort used when depth gets too big or the number of elements drops
// below a threshold.
__device__ void selection_sort( int *data, int left, int right )
{
  for( int i = left ; i <= right ; ++i ){
    int min_val = data[i];
    int min_idx = i;

    // Find the smallest value in the range [left, right].
    for( int j = i+1 ; j <= right ; ++j ){
      int val_j = data[j];
      if( val_j < min_val ){
        min_idx = j;
        min_val = val_j;
      }
    }

    // Swap the values.
    if( i != min_idx ){
      data[min_idx] = data[i];
      data[i] = min_val;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// Very basic quicksort algorithm, recursively launching the next level.
////////////////////////////////////////////////////////////////////////////////
__global__ void cdp_simple_quicksort(int *data, int left, int right, int depth ){
    //If we're too deep or there are few elements left, we use an insertion sort...
    if( depth >= 16 || right-left <= 32 ){
        selection_sort( data, left, right );
        return;
    }

    cudaStream_t s,s1;
    int *lptr = data+left;
    int *rptr = data+right;
    int  pivot = data[(left+right)/2];

    int lval;
    int rval;

    int nright, nleft;

    // Do the partitioning.
    while (lptr <= rptr){
        // Find the next left- and right-hand values to swap
        lval = *lptr;
        rval = *rptr;

        // Move the left pointer as long as the pointed element is smaller than the pivot.
        while (lval < pivot && lptr < data+right){
            lptr++;
            lval = *lptr;
        }

        // Move the right pointer as long as the pointed element is larger than the pivot.
        while (rval > pivot && rptr > data+left){
            rptr--;
            rval = *rptr;
        }

        // If the swap points are valid, do the swap!
        if (lptr <= rptr){
            *lptr = rval;
            *rptr = lval;
            lptr++;
            rptr--;
        }
    }

    // Now the recursive part
    nright = rptr - data;
    nleft  = lptr - data;

    // Launch a new block to sort the left part.
    if (left < (rptr-data)){
        cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
        cdp_simple_quicksort<<< 1, 1, 0, s >>>(data, left, nright, depth+1);
        cudaStreamDestroy(s);
    }

    // Launch a new block to sort the right part.
    if ((lptr-data) < right){
        cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
        cdp_simple_quicksort<<< 1, 1, 0, s1 >>>(data, nleft, right, depth+1);
        cudaStreamDestroy(s1);
    }
}


