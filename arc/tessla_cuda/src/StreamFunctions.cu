//
// Created by fabian on 28.05.21.
//
#include <cuda_runtime.h>
#include <sys/time.h>
#include <assert.h>
#include "main.cuh"
#include "helper.cuh"
#include "GPUStream.cuh"
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

__global__ void badsort(int* data, int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (data[j] > data[j+1]) {
                int temp = data[j];
                data[j] = data[j+1];
                data[j+1] = temp;
            }
        }
    }
}

__device__ void calcThreadsBlocks_device(int threads, int *block_size, int*blocks){
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

__global__ void delay_iteration(int* reset_timestamps, int* reset_offset, int reset_size, int* inputInt_timestamps, int* inputInt_values, int* inputInt_offset, int inputInt_size, int* result_timestamps, int* result_offset, int result_size, cudaStream_t stream) {
    cudaError_t lastCudaError;
    
    // Clear results
    memset(result_timestamps, -1, result_size * sizeof(int));
    // Allocate memory for temporary iteration results
    int* tempResults_offset = 0;
    lastCudaError = cudaMalloc((void**)&tempResults_offset, sizeof(int));
    if (lastCudaError == cudaErrorMemoryAllocation) {
        printf("Error allocating for tempResults_offset\n");
    }
    memset(tempResults_offset, 0, sizeof(int));
    int* tempResults = 0;
    lastCudaError = cudaMalloc((void**)&tempResults, reset_size * sizeof(int));
    if (lastCudaError == cudaErrorMemoryAllocation) {
        printf("Error allocating for tempResults\n");
    }
    memset(tempResults, -1, reset_size * sizeof(int));

    int resultCount = 0;

    int prevResultsCount = reset_size;
    while (prevResultsCount > 0) {
        int threads = prevResultsCount;
        int block_size = 0;
        int blocks = 0;
        calcThreadsBlocks_device(threads, &block_size, &blocks);

        delay_cuda<<<blocks, block_size, 0, stream>>>(inputInt_timestamps, inputInt_values, reset_timestamps, tempResults, threads, inputInt_size, inputInt_offset, reset_offset, tempResults_offset, stream);
        cudaDeviceSynchronize();

        cdp_simple_quicksort<<<1, 1, 0, stream>>>(tempResults, 0, threads - 1, 0);
        cudaDeviceSynchronize();

        calculate_offset<<<blocks, block_size, 0, stream>>>(tempResults + *tempResults_offset, tempResults_offset, threads);
        cudaDeviceSynchronize();
        
        prevResultsCount = threads - (*tempResults_offset - *reset_offset);

        if (prevResultsCount > 0) {
            memcpy(result_timestamps + resultCount, tempResults + *tempResults_offset, prevResultsCount * sizeof(int));
            resultCount += prevResultsCount;
        }

        int* temp_timestamps = reset_timestamps;
        int* temp_offset = reset_offset;
        reset_timestamps = tempResults;
        reset_offset = tempResults_offset;
        tempResults = temp_timestamps;
        tempResults_offset = temp_offset;
        *tempResults_offset = *reset_offset;

    }
    cdp_simple_quicksort<<<1, 1, 0, stream>>>(result_timestamps, 0, result_size - 1, 0);
    cudaDeviceSynchronize();
    int threads = result_size;
    int block_size = 0;
    int blocks = 0;
    calcThreadsBlocks_device(threads, &block_size, &blocks);

    *result_offset = 0;
    //calculate_offset<<<blocks, block_size, 0, stream>>>(result_timestamps, result_offset, threads);
    *result_offset = result_size - resultCount;
}

std::shared_ptr<GPUUnitStream> delay(std::shared_ptr<GPUIntStream> s, std::shared_ptr<GPUUnitStream> r, cudaStream_t stream) {
    std::shared_ptr<GPUIntStream> s_prune = std::make_shared<GPUIntStream>(*s, true);
    std::shared_ptr<GPUUnitStream> r_prune = std::make_shared<GPUUnitStream>(*r, true);

    // Prune GPUIntStream s, mark all events that can't possibly trigger because there's a reset event with value -1
    delay_preliminary_prune(s_prune, r_prune, stream);

    // Prepare original input data and result output
    std::shared_ptr<GPUUnitStream> prevResults = std::make_shared<GPUUnitStream>(*r_prune, true);
    std::shared_ptr<GPUUnitStream> result = std::make_shared<GPUUnitStream>(s->size, true);

    // Launch actual iterative algorithm on device
    delay_iteration<<<1, 1, 0, stream>>>(prevResults->device_timestamp, prevResults->device_offset, prevResults->size, s_prune->device_timestamp, s_prune->device_values, s_prune->device_offset, s_prune->size, result->device_timestamp, result->device_offset, result->size, stream);
    cudaDeviceSynchronize();

    return result;
}

/**
 * Removes all timestamps that cannot cause delay due to reset events. Caution: Has side effects on input streams.
 * @param s Integer input stream
 * @param r Unit input stream
 * @param stream CUDA stream number
 */
void delay_preliminary_prune(std::shared_ptr<GPUIntStream> s, std::shared_ptr<GPUUnitStream> r, cudaStream_t stream) {
    int threads = (int) s->size;
    int block_size = 1;
    int blocks = 1;
    calcThreadsBlocks(threads,&block_size,&blocks);
    
    printf("Scheduled delay_preliminary_prune() with <<<%d,%d>>>, %i threads \n",blocks,block_size, threads);
    delay_cuda_preliminary_prune<<<blocks, block_size, 0, stream>>>(s->device_timestamp, s->device_values, r->device_timestamp, threads, r->size, s->device_offset, r->device_offset, stream);
    cudaDeviceSynchronize();
}

__global__ void delay_cuda_preliminary_prune(int *inputIntTimestamps, int *inputIntValues, int *resetTimestamps, int size, int resetSize, int *offset, int *resetOffset, cudaStream_t stream) {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    inputIntTimestamps += *offset;
    inputIntValues += *offset;
    resetTimestamps += *resetOffset;

    if (i < size) { 
        int m = lookUpNextElement(resetSize, inputIntTimestamps[i], resetTimestamps);
        if (m > -1 && inputIntTimestamps[i] + inputIntValues[i] > resetTimestamps[m]) {
            inputIntValues[i] = -1;
        }
    }

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

    if (i < size) {
        // For each tempEvent, check if there's a matching (valid) event in IntStream s
        int index = lookUpElement(inputSize, resetTimestamps[i], inputIntTimestamps);
        if (index != INT_MIN && inputIntValues[index] != -1) {
            results[i] = inputIntTimestamps[index] + inputIntValues[index];
        } else {
            results[i] = -1;
        }
    }
}

__device__ void delay_cuda_rec(){

}


// https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#numa-best-practices
// ADD stream argument to enable multiple kernels in parallel (10.5. Concurrent Kernel Execution)
// Note:Low Medium Priority: Use signed integers rather than unsigned integers as loop counters.
std::shared_ptr<GPUIntStream> time(std::shared_ptr<GPUIntStream> input, cudaStream_t stream){
    int threads = input->size;
    int block_size = 1;
    int blocks = 1;
    calcThreadsBlocks(threads,&block_size,&blocks);
    // Create new stream on device the size of the input stream
    std::shared_ptr<GPUIntStream> result = std::make_shared<GPUIntStream>(input->size, true);
    // Fire off the actual calculation
    time_cuda<<<blocks,block_size,0,stream>>>(input->device_timestamp, result->device_timestamp, result->device_values, threads,input->device_offset,result->device_offset);
    printf("Scheduled time() with <<<%d,%d>>> \n",blocks,block_size);
    return result;
};




std::shared_ptr<GPUIntStream> last(std::shared_ptr<GPUIntStream> inputInt, std::shared_ptr<GPUUnitStream> inputUnit, cudaStream_t stream) {
    int threads = (int) inputUnit->size;
    int block_size = 1;
    int blocks = 1;
    calcThreadsBlocks(threads, &block_size, &blocks);

    // Create new stream on devicewith the size of the unit input stream
    std::shared_ptr<GPUIntStream> result = std::make_shared<GPUIntStream>(inputUnit->size, true);

    // Fire off the CUDA calculation
    last_cuda<<<blocks, block_size, 0, stream>>>(inputInt->device_timestamp, inputInt->device_values,
                                                 inputUnit->device_timestamp, result->device_timestamp,
                                                 result->device_values, inputInt->size, threads,
                                                 inputInt->device_offset, inputUnit->device_offset);
    //TODO! comment out
    printf("beforecalc\n");
    calcThreadsBlocks(threads,&block_size,&blocks);
    printf("aftercalc\n");
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
    calculate_offset<<<blocks, block_size, 0, stream>>>(result->device_timestamp, result->device_offset, threads);
    printf("Scheduled last() with <<<%d,%d>>> \n", blocks, block_size);
    return result;
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
    if (i<size-*offset){
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
    //assert(*b != 0 && "Divide by zero error in lift_div");
    if (*b == 0){
        printf("DIVISION BY ZERO\n");
        *result = 0;
        return;
    }
    *result = *a / *b;
}
__device__ void lift_mod(int *a, int *b, int *result){
    //assert(*b != 0 && "Divide by zero error in lift_mod");
    if (*b == 0){
        printf("MODULO BY ZERO\n");
        *result = 0;
        return;
    }
    *result = *a % *b;
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
        // If they match, result timestamp is x/y timestamp and result value is the cresult of the lift function
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

    if (x_ts[*x_i] <= y_ts[*y_i] && !x_done || y_done){
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
        if (out_ts[tidx*vpt+i] < 0){
            invalid[tidx]++;
        }
        else{
            valid[tidx]++;
        }
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
std::shared_ptr<GPUIntStream> lift(std::shared_ptr<GPUIntStream> x, std::shared_ptr<GPUIntStream> y, int op){
    int block_size = 0;
    int blocks = 0;
    int x_offset = *(x->host_offset);
    int y_offset = *(y->host_offset);
    int len_offset = x_offset+y_offset;
    int threads = (x->size-x_offset) + (y->size-y_offset);
    calcThreadsBlocks(threads, &block_size, &blocks);

    threads = (blocks) * (block_size);

    // Create Result
    std::shared_ptr<GPUIntStream> result(new GPUIntStream());
    result->size = x->size + y->size;
    result->host_timestamp = (int*)malloc(result->size * sizeof(int));
    result->host_values = (int*)malloc(result->size * sizeof(int));
    memset(result->host_timestamp, -1, result->size);
    memset(result->host_values, 0, result->size);
    result->copy_to_device(false);

    // Array to count valid timestamps
    int *valid_h = (int*)malloc(threads*sizeof(int));
    int *valid_d;
    memset(valid_h, 0, threads*sizeof(int));
    cudaMalloc((int**)&valid_d, threads*sizeof(int));

    // Array to count invalid timestamps
    int *invalid_h = (int*)malloc(threads*sizeof(int));
    int *invalid_d;
    memset(invalid_h, 0, threads*sizeof(int));
    cudaMalloc((int**)&invalid_d, threads*sizeof(int));

    // Array to copy the result to ... needed for offset calculations
    int *out_ts_cpy;
    int *out_v_cpy;
    cudaMalloc((int**)&out_ts_cpy, result->size*sizeof(int));
    cudaMalloc((int**)&out_v_cpy, result->size*sizeof(int));

    cudaDeviceSynchronize();

    // 3, 2, 1, go
    lift_cuda<<<blocks, block_size>>>(  x->device_timestamp, y->device_timestamp, 
                                        result->device_timestamp, 
                                        x->device_values, y->device_values,
                                        result->device_values,
                                        threads, (x->size), (y->size),
                                        op, valid_d, invalid_d, 
                                        out_ts_cpy, out_v_cpy, result->device_offset,
                                        x->device_offset, y->device_offset);
                            
    cudaDeviceSynchronize();

    if (op == MRG){
        inval_multiples_merge<<<blocks, block_size>>> ( op, threads,
                                                        x->size, y->size,
                                                        x->device_offset, y->device_offset,
                                                        out_ts_cpy+len_offset, invalid_d, valid_d);
        cudaDeviceSynchronize();
    }

    // Move invalid timestamps to front and set correct offset
    remove_invalid<<<blocks, block_size>>>( threads, invalid_d, valid_d, 
                                            x->size, y->size,
                                            out_ts_cpy, result->device_timestamp,
                                            out_v_cpy, result->device_values,
                                            x->device_offset, y->device_offset,
                                            result->device_offset, op);

    cudaDeviceSynchronize();

    // Free arrays
    // Something's not quite right yet
    
    //cudaFree(out_ts_cpy);
    //cudaFree(out_v_cpy);
    //cudaFree(invalid_d);
    //cudaFree(valid_d);
    //free(valid_h);
    //free(invalid_h);

    return result; 
}

__global__ void inval_multiples_merge(  int op, int threads,
                                        int x_len, int y_len,
                                        int *x_offset_d, int *y_offset_d,
                                        int *out_ts, int *invalid, int *valid){

    // If op is merge, check for double timestamps

    int x_offset = *x_offset_d;
    int y_offset = *y_offset_d;

    const int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    int len_offset = x_offset + y_offset;
    int vpt = ceil((double)((x_len + y_len)-len_offset) / (double)threads); // Values per thread 

    for (int i = 0; i < vpt && tidx*vpt+i < (x_len+y_len)-len_offset; i++){
        int l = max(tidx*vpt + i - 1, 0);
        int r = max(tidx*vpt + i, 1);

        if (out_ts[l] == out_ts[r]){
            out_ts[r] = -1;
            invalid[tidx]++;
            // Decrement valid, since it is at maximum due to incrementations before
            valid[tidx]--;
        }
    }
}

__global__ void remove_invalid( int threads, int *invalid, int *valid, 
                                int x_len, int y_len, 
                                int *out_ts_cpy, int *out_ts, 
                                int *out_v_cpy, int *out_v,
                                int *x_offset_d, int *y_offset_d,
                                int *result_offset, int op){

    int x_offset = *x_offset_d;
    int y_offset = *y_offset_d;

    const int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    int len_offset = x_offset + y_offset;
    int vpt = ceil((double)((x_len + y_len)-len_offset) / (double)threads); // Values per thread 

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

    for (int i = 0; i < vpt && tidx*vpt+i < (x_len+y_len)-len_offset; i++){
        if (out_ts_cpy[tidx*vpt+i+len_offset] >= 0){
            out_ts[vals_before+valid_cnt+len_offset] = out_ts_cpy[tidx*vpt+i+len_offset];
            out_v[vals_before+valid_cnt+len_offset] = out_v_cpy[tidx*vpt+i+len_offset];
            valid_cnt++;
        }
    }

    // Only one thread does this
    if (tidx == 0) {
        (*result_offset) = cuml_invalid+len_offset;
    }
}

__global__ void lift_cuda(  int *x_ts, int *y_ts, int *out_ts, 
                            int *x_v, int *y_v, int *out_v,
                            int threads, int x_len, int y_len, 
                            int op, int *valid, int *invalid,
                            int *out_ts_cpy, int *out_v_cpy, int *invalid_offset,
                            int *x_offset, int *y_offset){

    const int tidx = threadIdx.x + blockIdx.x * blockDim.x;         // Thread ID

    int xo = (*x_offset);
    int yo = (*y_offset);

    int len_offset = xo+yo;

    int vpt = ceil((double)((x_len + y_len)-len_offset) / (double)threads);     // Values per thread
    int diag = tidx * vpt;                                                      // Binary search constraint

    int intersect = merge_path(x_ts+xo, y_ts+yo, diag, x_len-xo, y_len-yo);
    int x_start = intersect;
    int y_start = diag - intersect;

    // Split op into merge vs. value function and specific value operation
    int fct = 0;
    if (op == MRG){
        op = 0;
        fct = 1;
    }
    
    if (tidx*vpt < (x_len+y_len)-len_offset) {
        int mems = min(vpt, ((x_len+y_len)-len_offset)-tidx*vpt);
        memset(out_ts_cpy+len_offset+tidx*vpt, -1, mems*sizeof(int));
    }

    lift_partition( x_ts+xo, y_ts+yo, out_ts_cpy+len_offset,
                    x_v+xo, y_v+yo, out_v_cpy+len_offset,
                    x_start, y_start, vpt, tidx, 
                    x_len-xo, y_len-yo, lift_funcs[fct], lift_ops[op],
                    valid, invalid);

}

std::shared_ptr<GPUIntStream> slift(std::shared_ptr<GPUIntStream> x, std::shared_ptr<GPUIntStream> y, int op){
    
    // Merge fast path
    if (op == MRG){
        return lift(x,y,MRG);
    }
    // Fast path for 1/2 empty stream(s)
    if (x->size == 0 || y->size == 0){
        int *e_ts = (int*)malloc(0);
        int *e_v = (int*)malloc(0);
        std::shared_ptr<GPUIntStream> empty(new GPUIntStream(e_ts, e_v, 0));
        empty->size = 0;
        empty->copy_to_device();
        return empty;
    }

    int *x_ts = (int*)malloc(x->size*sizeof(int));
    int *y_ts = (int*)malloc(y->size*sizeof(int));
    memcpy(x_ts, x->host_timestamp, x->size*sizeof(int));
    memcpy(y_ts, y->host_timestamp, y->size*sizeof(int));

    // xy ... y is the unit stream
    int *xy_ts = (int*)malloc(y->size*sizeof(int));
    int *yx_ts = (int*)malloc(x->size*sizeof(int));
    int *xy_v = (int*)malloc(y->size*sizeof(int));
    int *yx_v = (int*)malloc(x->size*sizeof(int));

    // Maybe cudaMemset?
    memset(xy_ts, -1, y->size*sizeof(int));
    memset(yx_ts, -1, x->size*sizeof(int));

    // Make Unit streams from Int streams for last()
    std::shared_ptr<GPUUnitStream> x_unit(new GPUUnitStream(x_ts, x->size, *(x->host_offset)));
    std::shared_ptr<GPUUnitStream> y_unit(new GPUUnitStream(y_ts, y->size, *(y->host_offset)));

    x_unit->copy_to_device();
    y_unit->copy_to_device();

    std::shared_ptr<GPUIntStream> last_xy = last(x, y_unit, 0);
    std::shared_ptr<GPUIntStream> last_yx = last(y, x_unit, 0);
    cudaDeviceSynchronize();

    // Fixes some bug, but WHY
    last_yx->copy_to_host();
    last_xy->copy_to_host();

    std::shared_ptr<GPUIntStream> x_prime = lift(x, last_xy, MRG);
    std::shared_ptr<GPUIntStream> y_prime = lift(y, last_yx, MRG);
    cudaDeviceSynchronize();

    std::shared_ptr<GPUIntStream> result = lift(x_prime, y_prime, op);
    cudaDeviceSynchronize();

    x_prime->free_device();
    x_prime->free_host();
    y_prime->free_device();
    y_prime->free_host();

    x_unit->free_device();
    x_unit->free_host();
    y_unit->free_device();
    y_unit->free_host();

    // MEMORY BUG WHEN FREEING LAST XY/YX

    return result;
}

// Scan from slides
__global__ void assign_vals(int *input, int *result_v, int *result_ts, int *input_offset, int *result_offset, int size){
    
    const int tidx = threadIdx.x + blockIdx.x * blockDim.x;

    if (input[0] == 0 && tidx < size){
        result_v[tidx] = tidx;
        (*result_offset)++;
    }
    else if (tidx < size){
        result_v[tidx] = tidx;
        //(*result_offset)++;
    }
    if (tidx == 0){
        memcpy(result_ts + 1, input, (size-1)*sizeof(int));
    }
    return;
}

std::shared_ptr<GPUIntStream> count(std::shared_ptr<GPUUnitStream> input){
    int threads = input->size + 1;
    int block_size = 1;
    int blocks = 1;
    calcThreadsBlocks(threads,&block_size,&blocks);

    std::shared_ptr<GPUIntStream> result(new GPUIntStream());

    cudaMalloc((void **) &result->device_timestamp, (input->size + 1)*sizeof(int));
    cudaMalloc((void **) &result->device_values, (input->size + 1)*sizeof(int));
    cudaMalloc((void **) &result->device_offset, sizeof(int));

    assign_vals<<<blocks, block_size>>>( input->device_timestamp, result->device_values, result->device_timestamp,
                                        input->device_offset, result->device_offset, 
                                        input->size+1);

    return result;
}