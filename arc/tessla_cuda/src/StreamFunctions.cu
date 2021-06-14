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

    int m = lookUpNextElement(resetSize, inputIntTimestamps[i], resetTimestamps);
    if (m > -1 && inputIntTimestamps[i] + inputIntValues[i] > resetTimestamps[m])
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

// Binary search looking for next highest timestamp instead of exact match
__device__ int lookUpNextElement(int size, int searchValue, int *timestamps) {
    int L = 0;
    int R = size - 1;
    int m = INT_MIN;
    int out = INT_MIN;
    //TODO! APPLY OFFSET DUE TO INVALID WEIGHTS

    if (timestamps[size-1] > searchValue) {
        while (L<=R) {
            m = (int) (L+R)/2;
            if (timestamps[m] <= searchValue) {
                L = m + 1;
            } else {
                out = m;
                R = m - 1;
            }
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
    cudaDeviceSynchronize();
    //TODO! check that no expection is thrown at launch!
    last_cuda<<<blocks,block_size,0,stream>>>(inputInt->device_timestamp, inputInt->device_values, inputUnit->device_timestamp,result->device_timestamp,result->device_values,inputInt->size, threads,inputInt->device_offset,inputUnit->device_offset);
    cudaDeviceSynchronize();
    calculate_offset<<<blocks, block_size, 0, stream>>>(result->device_timestamp,result->device_offset, threads);
    printf("Scheduled last() with <<<%d,%d>>> \n",blocks,block_size);
}

//reduction example followed from: https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
//calculates the number of non valid timestamps
__global__ void calculate_offset(int* timestamps, int* offset, int size){
    __shared__ int sdata[1024];// each thread loadsone element from global to shared memunsigned 

    int tid = threadIdx.x;
    unsigned int i= blockIdx.x*blockDim.x+ threadIdx.x;
    int block_offset = 0;
    sdata[tid] = 0;

    if (i < size){
         //printf(" timestamp %d \n",*(timestamps+i));
         if (*(timestamps+i) < 0){
            sdata[tid] = 1;
         }
    }
    __syncthreads();


    for(unsigned int s=1; s < blockDim.x; s *= 2) {
        int index = 2 * s * tid;
        if (index < blockDim.x) {
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }
    __syncthreads();
    if(tid == 0){ 
        block_offset = sdata[0];
        atomicAdd(offset, block_offset);
    }
    

}
__global__ void last_cuda(int* input_timestamp, int* input_values,int*unit_stream_timestamps,  int* output_timestamps, int* output_values, int intStreamSize, int size, int* offsInt, int* offsUnit){
    
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    
    //shift accordingly to offset
    unit_stream_timestamps += *offsUnit;
    input_timestamp += *offsInt;
    input_values += *offsInt;
        int local_unit_timestamp;
    if (i < size){
        output_timestamps[i] = -1;
        local_unit_timestamp = unit_stream_timestamps[i];
    }
    size -= *offsUnit;
    intStreamSize -= *offsInt;

    output_timestamps += *offsUnit;
    output_values += *offsUnit;
    int out =  -2;


    //Search for the timestamp per thread

    int L = 0;
    int R = intStreamSize-1;
    int m;
    __syncthreads();
    if (i<size) {

        while (L<=R) {
           //__syncthreads();
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
    }
    //all have their respective out values
    //the output_values array has been successfully filled
    //now the threads perform an and reduction starting at 0 going to size
    __syncthreads();
    if (i < size){
        if (out <0){
            //printf("out %d \n",out);
        }
        output_values[i] = out;
        if (i < 40){
            //printf("thread: %d \n",i);
            //printf("out value %d\n", out);
        }
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

/**
 * MergePath, also used for lift
 */
__device__ int merge_path(int *x, int *y, int diag, int x_len, int y_len) {
    // Just using UnitStreams for now
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    int begin = max(0, diag - y_len);               // Start of search window
    int end = min(diag, x_len);                     // End of search window
    int mid;

    // Binary search
    while(begin < end){
    
        mid = (end + begin) / 2;
        int x_ts = x[mid];
        int y_ts = y[diag - 1 - mid];

        if (x_ts < y_ts) {
            begin = mid + 1;
        }
        else{
            end = mid;
        }
    }
    return begin;
}

typedef void (*lift_op) (int*, int*, int*);

typedef void (*lift_func) ( int*, int*, 
                            int*, int*, 
                            int*, int*,
                            int*, int*,
                            bool, bool, lift_op);

__device__ void lift_add(int *a, int *b, int *result){
    *result = *a + *b;
}
__device__ void lift_sub(int *a, int *b, int *result){
    *result = *a - *b;
}
__device__ void lift_mul(int *a, int *b, int *result){
    *result = *a * *b;
}
__device__ void lift_div(int *a, int *b, int *result){
    if (*b != 0){
        *result = *a / *b;
    }
    else{
        // For now, dividing by 0 results in 0
        *result = 0;
    }
}
__device__ void lift_mod(int *a, int *b, int *result){
    if (*b != 0){
        *result = *a % *b;
    }
    else{
        // For now, dividing by 0 results in 0
        *result = 0;
    }
}

__device__ void lift_value( int *x_ts, int *y_ts,
                            int *x_v, int *y_v,
                            int *x_i, int *y_i, 
                            int *out_ts, int *out_v,
                            bool x_done, bool y_done, lift_op op){

    if (x_ts[*x_i] != y_ts[*y_i]){
        // If timestamps don't match, result timestamp is invalid (-1)
        *out_ts = -1;

        if (*x_i < *y_i || y_done){
            (*x_i)++;
        }
        else{
            (*y_i)++;
        }
    }
    else{
        // If they match, result timestamp is x/y timestamp and result value is the result of the lift function
        *out_ts = x_ts[*x_i];
        // Specific value based lift operation
        op(&x_v[*x_i], &y_v[*y_i], out_v);

        if (!x_done){
            (*x_i)++;
        }
        if (!y_done){
            (*y_i)++;
        }
    }
}

__device__ void lift_merge( int *x_ts, int *y_ts,
                            int *x_v, int *y_v,
                            int *x_i, int *y_i, 
                            int *out_ts, int *out_v,
                            bool x_done, bool y_done, lift_op op){

    if (x_ts[*x_i] <= y_ts[*y_i] && !x_done){
        *out_ts = x_ts[*x_i];
        *out_v = x_v[*x_i];
        (*x_i)++;
    }
    else{
        *out_ts = y_ts[*y_i];
        *out_v = y_v[*y_i];
        (*y_i)++;
    }
}

__device__ lift_func lift_funcs[] = { lift_value, lift_merge };
__device__ lift_op lift_ops[] = { lift_add, lift_sub, lift_mul, lift_div, lift_mod };

// Device internal sequential processing of small partitions
__device__ void lift_partition( int *x_ts, int *y_ts, int *out_ts,
                                int *x_v, int *y_v, int *out_v,
                                int x_start, int y_start,
                                int vpt, int tidx,
                                int x_len, int y_len,
                                lift_func fct, lift_op op, 
                                int* valid, int *invalid){

    int x_i = x_start;
    int y_i = y_start;
    int size = vpt;

    bool x_done = x_i >= x_len ? true : false;
    bool y_done = y_i >= y_len ? true : false;

    // Could possibly be optimized since only the last block needs range checks
    // #pragma unroll is also an option according to https://moderngpu.github.io/merge.html
    for(int i = 0; i < vpt; i++) {
        // Break if last block doesn't fit
        if (x_done && y_done){
            break;
        }

        int offset = (tidx*vpt) + i;

        fct(x_ts, y_ts, 
            x_v, y_v,
            &x_i, &y_i, 
            out_ts+offset, out_v+offset,
            x_done, y_done, op);

        if (x_i >= x_len){
            x_done = true;
        }
        if (y_i >= y_len){
            y_done = true;
        }
        if ((x_i + y_i) - (x_start + y_start) >= vpt){
            x_done = true;
            y_done = true;
        }
    }

    // Count valid/invalid timestamps per partition
    for (int i = 0; i < vpt && tidx*vpt+i < x_len+y_len; i++){
        if (out_ts[tidx*vpt+i] == -1){
            invalid[tidx]++;
        }
        else{
            valid[tidx]++;
        }
    }

    __syncthreads();

    // Afterwards, threads have to check for overlapping timestamps in their c[] partition in case of merge
    // VPT > 1 check not really necessary, we should guarantee that VPT > 1 beforehand, otherwise the mergepath is the full merge anyway
    if (vpt > 1 && fct == lift_merge){
        for (int i = 0; i < vpt && tidx*vpt+i < x_len+y_len; i++){
            int l = max(tidx*vpt + i - 1, 0);
            int r = max(tidx*vpt + i, 1);
            if (out_ts[l] == out_ts[r]){
                out_ts[r] = -1;
                invalid[tidx]++;
                // Decrement valid, since it is at maximum due to incrementations before
                valid[tidx]--;
            }
        }
        __syncthreads();
    }
}

// Information about MergePath
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


/**
 * Lift
 */
void lift(IntStream *x, IntStream *y, IntStream *result, int threads, int op){
    int block_size = 0;
    int blocks = 0;
    calcThreadsBlocks(threads, &block_size, &blocks);

    threads = (blocks) * (block_size);

    result->host_timestamp = (int*)malloc((x->size+y->size)*sizeof(int));
    result->host_values = (int*)malloc((x->size+y->size)*sizeof(int));
    result->size = x->size+y->size;
    memset(result->host_timestamp, -1, result->size * sizeof(int));

    // cudaMalloc Timestamp arrays
    x->copy_to_device();
    y->copy_to_device();
    result->copy_to_device();

    // Array to count valid timestamps
    int *valid_h = (int*)malloc(threads*sizeof(int));
    int *valid_d;
    memset(valid_h, 0, threads*sizeof(int));
    cudaMalloc((int**)&valid_d, threads*sizeof(int));
    cudaMemcpy(valid_d, valid_h, threads*sizeof(int), cudaMemcpyHostToDevice);

    // Array to count invalid timestamps
    int *invalid_h = (int*)malloc(threads*sizeof(int));
    int *invalid_d;
    memset(invalid_h, 0, threads*sizeof(int));
    cudaMalloc((int**)&invalid_d, threads*sizeof(int));
    cudaMemcpy(invalid_d, invalid_h, threads*sizeof(int), cudaMemcpyHostToDevice);

    // Array to copy the result to ... needed for offset calculations
    int *out_ts_cpy;
    int *out_v_cpy;
    cudaMalloc((int**)&out_ts_cpy, result->size*sizeof(int));
    cudaMalloc((int**)&out_v_cpy, result->size*sizeof(int));

    // 3, 2, 1, go
    lift_cuda<<<blocks, block_size>>>(   x->device_timestamp, y->device_timestamp, 
                                            result->device_timestamp, 
                                            x->device_values, y->device_values,
                                            result->device_values,
                                            threads, x->size, y->size, 
                                            op, valid_d, invalid_d, 
                                            out_ts_cpy, out_v_cpy, result->device_offset);

    // Copy back results
    result->copy_to_host();
}

__global__ void lift_cuda(  int *x_ts, int *y_ts, int *out_ts, 
                            int *x_v, int *y_v, int *out_v,
                            int threads, int x_len, int y_len, 
                            int op, int *valid, int *invalid,
                            int *out_ts_cpy, int *out_v_cpy, int *invalid_offset){

    const int tidx = threadIdx.x + blockIdx.x * blockDim.x;        // Thread ID

    int vpt = ceil((double)(x_len + y_len) / (double)threads);  // Values per thread
    int diag = tidx * vpt;                                         // Binary search constraint

    int intersect = merge_path(x_ts, y_ts, diag, x_len, y_len);
    int x_start = intersect;
    int y_start = diag - intersect;

    // Split op into merge vs. value function and specific value operation
    int fct = 0;
    if (op == MRG){
        op = 0;
        fct = 1;
    }

    memset(out_ts_cpy, -1, (x_len+y_len)*sizeof(int));
    
    lift_partition( x_ts, y_ts, out_ts_cpy,
                    x_v, y_v, out_v_cpy,
                    x_start, y_start, vpt, tidx, 
                    x_len, y_len, lift_funcs[fct], lift_ops[op], 
                    valid, invalid);

    // Each thread can now add up all valid/invalid timestamps and knows how to place their valid timestamps
    int cuml_invalid = 0;
    for (int i = 0; i < threads; i++){
        cuml_invalid += invalid[i];
    }

    int cuml_valid_before = 0;
    for (int i = 0; i < tidx; i++){
        cuml_valid_before += valid[i];
    }

    int vals_before = cuml_invalid + cuml_valid_before;
    int valid_cnt = 0;

    for (int i = 0; i < vpt && tidx*vpt+i < x_len+y_len; i++){
        if (out_ts_cpy[tidx*vpt+i] != -1){
            out_ts[vals_before+valid_cnt] = out_ts_cpy[tidx*vpt+i];
            out_v[vals_before+valid_cnt] = out_v_cpy[tidx*vpt+i];
            valid_cnt++;
        }
    }
    if (tidx == 0) {
        *invalid_offset = cuml_invalid;
    }
    __syncthreads();
}