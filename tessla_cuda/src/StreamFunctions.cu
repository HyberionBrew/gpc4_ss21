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
    int threads = input->size;
    int block_size = 1;
    int blocks = 1;
    calcThreadsBlocks(threads,&block_size,&blocks);
    //set output stream to input stream size
    if (!result->onDevice) {
        int sizeAllocated = input->size * sizeof(int);
        result->size = input->size;
        result->host_timestamp = (int *) malloc(input->size * sizeof(int));
        result->host_values = (int *) malloc(input->size * sizeof(int));
        memset(result->host_timestamp, 0, sizeAllocated);
        memset(result->host_values, 0, sizeAllocated);
        result->copy_to_device(false);
    }
    time_cuda<<<blocks,block_size,0,stream>>>(input->device_timestamp, result->device_timestamp, result->device_values, threads,input->device_offset,result->device_offset);
    printf("Scheduled time() with <<<%d,%d>>> \n",blocks,block_size);
};




void last(IntStream *inputInt, UnitStream *inputUnit, IntStream *result, cudaStream_t stream){
    int threads = (int) inputUnit->size;
    int block_size =1;
    int blocks = 1;
    calcThreadsBlocks(threads,&block_size,&blocks);

    //copy result vector to device
    if (!result->onDevice) {
        //TODO! where do we free this?
        int sizeAllocated = inputUnit->size * sizeof(int);
        result->size = inputUnit->size;
        result->host_timestamp = (int *) malloc(inputUnit->size * sizeof(int));
        result->host_values = (int *) malloc(inputUnit->size * sizeof(int));
        memset(result->host_timestamp, 0, sizeAllocated);
        memset(result->host_values, 0, sizeAllocated);
        result->copy_to_device(false);
    }

    int* block_red;
    cudaMalloc((void**)&block_red, sizeof(int)*blocks);
    //TODO! check that no expection is thrown at launch!
    last_cuda<<<blocks,block_size,0,stream>>>(block_red, inputInt->device_timestamp, inputInt->device_values, inputUnit->device_timestamp,result->device_timestamp,result->device_values,inputInt->size, threads,inputInt->device_offset,inputUnit->device_offset);
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
}

//reduction example followed from: https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
__device__ void count_valid(int * sdata,int * output_timestamp,int* valid, int size, int MaxSize, unsigned int tid, const int i){
    //each thread loads one Element from global to shared memory
    sdata[tid] = 0;

    if (output_timestamp[i] < 0) {
        sdata[tid] = 1;
        //    printf("%d ? %d\n",i,output_timestamp[i]);
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
    if (tid == 0) *valid=*valid+sdata[0];
}
// Device internal sequential merge of small partitions
__device__ void merge_serial(int *a, int *b, int *c,
                             int a_start, int b_start,
                             int vpt, int tidx,
                             int a_len, int b_len){
    int a_i = a_start;
    int b_i = b_start;
    int a_val = a[a_i];
    int b_val = b[b_i];

    bool a_done = false;
    bool b_done = false;

    // Could possibly be optimized since only the last block needs range checks
    // #pragma unroll is also an option according to https://moderngpu.github.io/merge.html
    for(int i = 0; i < vpt; ++i) {
        // Break if last block doesn't fit
        if (a_done && b_done){
            break;
        }

        if (a_done){
            c[tidx*vpt + i] = b_val;
            b_i++;
        }
        else if (b_done){
            c[tidx*vpt + i] = a_val;
            a_i++;
        }
        else if (a_val <= b_val){
            c[tidx*vpt + i] = a_val;
            a_i++;
            if (a_val == b_val){
                // Invalidate b values with overlapping timestamps
                b_val = -1;
                b[b_i] = -1;
            }
        }
        else{
            c[tidx*vpt + i] = b_val;
            b_i++;
        }

        if (a_i >= a_len){
            a_done = true;
        }
        else{
            a_val = a[a_i];
        }

        if (b_i >= b_len){
            b_done = true;
        }
        else{
            b_val = b[b_i];
        }
    }
    __syncthreads();
}


__global__ void last_cuda(int* block_red, int* input_timestamp, int* input_values,int*unit_stream_timestamps,  int* output_timestamps, int* output_values, int intStreamSize, int size, int* offsInt, int* offsUnit){
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int tid = threadIdx.x;
    //shift accordingly to offset


    unit_stream_timestamps += *offsUnit;
    input_timestamp += *offsInt;
    input_values += *offsInt;

    size -= *offsUnit;
    intStreamSize -= *offsInt;
    output_timestamps[i] = INT_MIN;
    output_timestamps += *offsUnit;
    output_values += *offsUnit;
    int out =  INT_MIN;
    __syncthreads(); //should be irrelevant
    if (i<size) {
        int local_unit_timestamp = unit_stream_timestamps[i];
        __shared__ int sdata[1024];
        int L = 0;
        int R = intStreamSize-1;
        int m;
        while (L<=R) {
           __syncthreads();
            m = (int) (L+R)/2;

            if (input_timestamp[m]<local_unit_timestamp){
                L = m + 1;
                out = input_values[m];

                output_timestamps[i] = unit_stream_timestamps[i];
            }
            else if (input_timestamp[m]>=local_unit_timestamp){
                R = m -1;
            }
            else{
                out = input_values[m];
                output_timestamps[i] = unit_stream_timestamps[i];
                break;
            }
        }
       //

        output_values[i] = out;
        block_red[blockIdx.x] = *offsUnit;
        count_valid(sdata,output_timestamps,&block_red[blockIdx.x], 1024,size,tid,i);
    }
}

// working
__global__ void time_cuda(int* input_timestamp, int* output_timestamps, int* output_values,int size, int*offset, int* resultOffset){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    input_timestamp += *offset;
    output_timestamps += *offset;
    output_values += *offset;
    if ( i<size-*offset){
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

__device__ int merge_path(int *a, int *b, int diag, int a_len, int b_len) {
    // Just using UnitStreams for now
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    int begin = max(0, diag - b_len);               // Start of search window
    int end = min(diag, a_len);                     // End of search window
    int mid;

    // Binary search
    while(begin < end){
    
        mid = (end + begin) / 2;
        int a_val = a[mid];
        int b_val = b[diag - 1 - mid];

        if (a_val < b_val) {
            begin = mid + 1;
        }
        else{
            end = mid;
        }
    }
    return begin;
}

// Device internal sequential merge of small partitions
__device__ void merge_serial(int *a, int *b, int *c,
                             int a_start, int b_start,
                             int vpt, int tidx,
                             int a_len, int b_len){

    int a_i = a_start;
    int b_i = b_start;
    int a_val = a[a_i];
    int b_val = b[b_i];
    int size = vpt;

    bool a_done = a_i >= a_len ? true : false;
    bool b_done = b_i >= b_len ? true : false;

    // Could possibly be optimized since only the last block needs range checks
    // #pragma unroll is also an option according to https://moderngpu.github.io/merge.html
    for(int i = 0; i < vpt; ++i) {

        // Break if last block doesn't fit
        if (a_done && b_done){
            break;
        }

        if (a_done){
            c[tidx*vpt + i] = b_val;
            b_i++;
        }
        else if (b_done){
            c[tidx*vpt + i] = a_val;
            a_i++;
        }
        else if (a_val <= b_val){
            c[tidx*vpt + i] = a_val;
            a_i++;
            if (a_val == b_val && (b_i > b_start || tidx == 0)){
                b[b_i] = -1;
            }
           
        }
        else{
            c[tidx*vpt + i] = b_val;
            b_i++;
        }

        if (a_i >= a_len){
            a_done = true;
        }
        else{
            a_val = a[a_i];
        }

        if (b_i >= b_len){
            b_done = true;
        }
        else{
            b_val = b[b_i];
        }
    }

    __syncthreads();

    if (tidx == 0){
        // Thread 0 does not have to check its starting values
        return;
    }

    // Afterwards, threads have to check for overlapping timestamps in their c[] partition!
    // VPT > 1 check not really necessary, we should guarantee that VPT > 1 beforehand, otherwise the mergepath is the full merge anyway
    if (vpt > 1){
        for (int i = 0; i < vpt; i++){
            if (c[tidx*vpt + i - 1] == c[tidx*vpt + i]){
                c[tidx*vpt + i] = -1;
            }
        }
    }
}

// https://stackoverflow.com/questions/30729106/merge-sort-using-cuda-efficient-implementation-for-small-input-arrays
/**
 * See the following paper for parallel merging of sorted arrays:
 * O. Green, R. Mccoll, and D. Bader
 * GPU merge path: a GPU merging algorithm
 * International Conference on Supercomputing
 * November 2014
 * URL: https://www.researchgate.net/publication/254462662_GPU_merge_path_a_GPU_merging_algorithm
 *
 * The paper claims a runtime complexity of O(log n + n/p), p ... # of processors
 */
void merge(UnitStream *s1, UnitStream *s2, UnitStream *result, int threads){

    int block_size = 1;
    int blocks = 1;

    // Export to function maybe -> reusable (e.g. with a config struct)
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

    for (int bl=1; bl <= MAX_BLOCKS*1000; bl++){
        blocks = bl;
        if (bl*block_size > threads){
            break;
        }
    }

    threads = blocks*block_size;

    // Using the pseudo-code in the paper
    int a_len = sizeof(s1->host_timestamp) / sizeof(s1->host_timestamp[0]);
    int b_len = sizeof(s2->host_timestamp) / sizeof(s2->host_timestamp[0]);
    memset(result->host_timestamp, -1, (a_len + b_len) * sizeof(int));

    // cudaMalloc Timestamp arrays
    s1->copy_to_device();
    s2->copy_to_device();
    result->copy_to_device();

    int sha_memsize = (a_len + b_len) * sizeof(int);

    // 3, 2, 1, go
    merge_cuda<<<blocks, block_size, sha_memsize>>>(s1->device_timestamp, s2->device_timestamp, result->device_timestamp, threads, s1->size, s2->size);

    // Copy back results
    result->copy_to_host();
    
    printf("After Merge\n");
    printf("S1: -----------------------------------------\n");
    for (int i = 0; i < s1->size; i++){
        printf("%i, ", s1->host_timestamp[i]);
    }
    printf("\n-----------------------------------------------\n");

    printf("S2: -----------------------------------------\n");
    for (int i = 0; i < s2->size; i++){
        printf("%i, ", s2->host_timestamp[i]);
    }
    printf("\n-----------------------------------------------\n");

    printf("Result: -----------------------------------------\n");
    for (int i = 0; i < result->size; i++){
        printf("%i, ", result->host_timestamp[i]);
    }
    printf("\n-----------------------------------------------\n");
    
}

// https://moderngpu.github.io/merge.html
// https://github.com/moderngpu/moderngpu/blob/V1.1/include/device/ctamerge.cuh
__global__ void merge_cuda(int *a, int *b, int *c, int threads, int a_len, int b_len){

    const int i = threadIdx.x + blockIdx.x * blockDim.x;        // Thread ID

    int vpt = ceil((double)(a_len + b_len) / (double)threads);  // Values per thread
    int diag = i * vpt;                                         // Binary search constraint

    int intersect = merge_path(a, b, diag, a_len, b_len);
    int a_start = intersect;
    int b_start = diag - intersect;

    merge_serial(a, b, c, a_start, b_start, vpt, i, a_len, b_len);
}




