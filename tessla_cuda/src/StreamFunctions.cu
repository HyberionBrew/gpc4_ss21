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

int simp_compare(const void *a, const void *b) { // TODO: Delete when no longer needed
    return ( *(int*)a - *(int*)b );
}

void delay(IntStream *s, UnitStream *r, UnitStream*result, cudaStream_t stream){
    // Prune IntStream s, mark all events that can't possibly trigger because there's a reset event with value -1
    delay_preliminary_prune(s, r, stream);

    // Allocate arrays for search and set reset-UnitStream as first input
    // New output events in each iteration are bounded by size of r
    int *prevResultsTimestamps = (int*) malloc(r->size * sizeof(int));
    memcpy(prevResultsTimestamps, r->host_timestamp, r->size * sizeof(int));
    UnitStream prevResults(prevResultsTimestamps, r->size);
    prevResults.copy_to_device();
    *prevResults.host_offset = (int) r->size;

    int *tempResultsTimestamps = (int*) malloc(r->size * sizeof(int));
    UnitStream tempResults(tempResultsTimestamps, r->size);
    tempResults.copy_to_device();

    int resultIndex = 0; // TODO: Change?
    int prevResultsCount = r->size; // TODO: Change?
    *result->host_offset = (int) result->size; // TODO: Change?

    // Iteratively search for new output events
    while (prevResultsCount > 0) {
        int threads = prevResultsCount;
        int block_size = 1;
        int blocks = 1;
        calcThreadsBlocks(threads, &block_size, &blocks);

        printf("Scheduled delay() with <<<%d,%d>>>, %i threads \n",blocks,block_size, threads);
        delay_cuda<<<blocks, block_size, 0, stream>>>(s->device_timestamp, s->device_values, prevResults.device_timestamp, tempResults.device_timestamp, threads, s->size, s->device_offset, prevResults.device_offset, tempResults.device_offset, stream);
        tempResults.copy_to_host(); 

        // Merge output events into existing output events
        // Sort tempResults to find actual new events (> -1)
        // TODO: Use parallel sort, parallel merge and count_valid
        qsort(tempResults.host_timestamp, threads, sizeof(int), simp_compare); // TODO: Use parallel sort
        int firstResult = -1;
        for (int i = 0; i < threads; i++) {
            //printf("tempResults.host_timestamp[%i] == %i\n", i, tempResults.host_timestamp[i]);
            if (firstResult == -1 && tempResults.host_timestamp[i] > -1)
                firstResult = i;
            if (tempResults.host_timestamp[i] > 0) {
                // Add tempResults to result. TODO: Change?
                result->host_timestamp[resultIndex] = tempResults.host_timestamp[i];
                *result->host_offset -= 1;
                resultIndex++;
            }
        }

        if (firstResult == -1) {
            prevResultsCount = 0;
            break; // TODO: ?
        }

        // Switch prevResults and tempResults to continue search with newly found timestamps
        prevResultsCount = threads - firstResult;
        UnitStream temp = prevResults;
        prevResults = tempResults;
        tempResults = temp;
        *prevResults.host_offset = prevResults.size - prevResultsCount;
        *tempResults.host_offset = 0;
    }

    // TODO: Sort & prune duplicate result
    qsort(result->host_timestamp, result->size, sizeof(int), simp_compare); // TODO: Use parallel sort
    result->copy_to_device();   // Test copies back from device, but result is only on host right now
    printf("SEARCH DONE, %i Results\n", resultIndex);

    // Cleanup
    prevResults.free_device();
    tempResults.free_device();
    free(prevResultsTimestamps);
    free(tempResultsTimestamps);
}

void delay_preliminary_prune(IntStream *s, UnitStream *r, cudaStream_t stream) {
    int threads = (int) s->size;
    int block_size = 1;
    int blocks = 1;
    calcThreadsBlocks(threads,&block_size,&blocks);
    
    printf("Scheduled delay_preliminary_prune() with <<<%d,%d>>>, %i threads \n",blocks,block_size, threads);
    delay_cuda_preliminary_prune<<<blocks, block_size, 0, stream>>>(s->device_timestamp, s->device_values, r->device_timestamp, threads, r->size, s->device_offset, r->device_offset, stream);
}

__global__ void delay_cuda_preliminary_prune(int *inputIntTimestamps, int *inputIntValues, int *resetTimestamps, int size, int resetSize, int *offset, int *resetOffset, cudaStream_t stream) {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int tid = threadIdx.x;
    inputIntTimestamps += *offset;
    inputIntValues += *offset;
    resetTimestamps += *resetOffset;

    int m = 0;
    int lastIntElement = size;
    
    while (resetTimestamps[m] <= inputIntTimestamps[i] && m <= resetSize) {
        m++; // TODO: Use binary search
    }
    if (inputIntTimestamps[i] + inputIntValues[i] > resetTimestamps[m])
        inputIntValues[i] = -1;

}


// binary search
// on failure returns INT_MIN
// returns position of the Element with value x
__device__ int lookUpElement(int size,int searchValue, int * input_timestamp){
    int L = 0;
    int R = size;
    int m = INT_MIN;
    int out = INT_MIN;
    //TODO! APPLY OFFSET DUE TO INVALID WEIGHTS

    while (L<=R) {
        // is this needed? TODO! check and discuss
        //maybe it helps? CHECK!
        __syncthreads();
        m = (int) (L+R)/2;
        if (input_timestamp[m]<searchValue){
            L = m + 1;
        }
        else if (input_timestamp[m]>searchValue){
            R = m -1;
        }
        else{
            out = m;
            break;
        }
    }
    return out;
}



__global__ void delay_cuda(int *inputIntTimestamps, int *inputIntValues, int *resetTimestamps, int *results, int size, int inputSize, int *inputOffset, int *resetOffset, int* resultOffset,cudaStream_t stream) {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    inputIntTimestamps += *inputOffset;
    inputIntValues += *inputOffset;
    resetTimestamps += *resetOffset;
    results += *resultOffset;

    // For each tempEvent, check if there's a matching (valid) event in IntStream s
    int index = lookUpElement(inputSize, resetTimestamps[i], inputIntTimestamps);
    if (index != INT_MIN && inputIntValues[index] != -1) {
        results[i] = inputIntTimestamps[index] + inputIntValues[index];
    } else {
        results[i] = -1;
    }
}

__device__ void delay_cuda_rec(){

}
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
 /*
void merge(int *s1_h, int *s2_h, int threads){

    /*for (size_t i = 0; i < threads; i++){
        a_diag[i] = a_len;
        b_diag[i] = b_len;
    }*//*

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
    //TODO! check how many MAX_BLOCKS
    for (int bl=1; bl <= MAX_BLOCKS*1000; bl++){
        blocks = bl;
        if (bl*block_size > threads){
            break;
        }
    }

    // Using the pseudo-code in the paper
    int a_len = sizeof(s1_h) / sizeof(s1_h[0]);
    int b_len = sizeof(s2_h) / sizeof(s2_h[0]);
    int *a_diag;
    int *b_diag;
    int *s1_d, *s2_d, *out_d;

    // Diag arrays for the algorithm
    cudaMalloc((void**) &a_diag, threads*sizeof(int));
    cudaMalloc((void**) &b_diag, threads*sizeof(int));

    // cudaMalloc Timestamp arrays
    cudaMalloc((void**) &s1_d, a_len*sizeof(int));
    cudaMalloc((void**) &s2_d, b_len*sizeof(int));
    cudaMalloc((void**) &out_d, (a_len+b_len)*sizeof(int));

    // Copy input arrays to device
    cudaMemcpy(s1_d, s1_h, a_len*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(s2_d, s2_h, a_len*sizeof(int), cudaMemcpyHostToDevice);

    // 3, 2, 1, go
    merge_cuda<<<blocks, block_size>>>(s1_d, s2_d, out_d);

    // Copy back results
    int out_size = (a_len + b_len) * sizeof(int);
    int *out_h = (int*)malloc(out_size);
    cudaMemcpy(out_h, out_d, out_size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < out_size; i++){
        std::cout << out_h[i] << ", ";
    }
    std::cout << std::endl;
}

__device__ merge_path(int *a, int *b, int diag, int a_len, int b_len) {
    // Just using UnitStreams for now
    int begin = max(0, diag - b_len);               // Start of search window
    int end = min(diag, a_len);                     // End of search window

    // Binary search
    while(begin < end){
        int mid = (end - begin) / 2;
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
}*/


// https://moderngpu.github.io/merge.html
// https://github.com/moderngpu/moderngpu/blob/V1.1/include/device/ctamerge.cuh
/*
__global__ void merge_cuda(int *a, int *b, int *c, int threads){
    // Thread
    const int i = threadIdx.x + blockIdx.x * blockDim.x;

    int a_len = sizeof(a) / sizeof(a[0]);           // Length of input a
    int b_len = sizeof(b) / sizeof(b[0]);           // Length of input b
    int vpt = (a_len + b_len) / threads;            // Values per thread
    int diag = i * vpt;                             // Binary search constraint

    int intersect = merge_path(a, b, diag);
    int a_start = intersect;
    int b_start = diag - intersect;

    merge_serial(a, b, c, a_start, b_start, vpt, i);
}*/




