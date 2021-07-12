//
// Created by fabian on 28.05.21.
//

#ifndef TESSLA_CUDA_STREAMFUNCTIONS_CUH
#define TESSLA_CUDA_STREAMFUNCTIONS_CUH

// Lift operations
#define ADD 0
#define SUB 1
#define MUL 2
#define DIV 3
#define MOD 4
#define MRG 5

#include <cuda_runtime.h>
#include "GPUStream.cuh"
#include <memory>

// time operation for unit stream
int simp_compare(const void *a, const void *b); // TODO: Delete when no longer needed
void calcThreadsBlocks(int threads, int *block_size, int*blocks);
std::shared_ptr<GPUIntStream> time(std::shared_ptr<GPUIntStream> input, cudaStream_t stream);
std::shared_ptr<GPUIntStream> last(std::shared_ptr<GPUIntStream> s, std::shared_ptr<GPUUnitStream> r, cudaStream_t stream);
std::shared_ptr<GPUUnitStream> delay(std::shared_ptr<GPUIntStream> s, std::shared_ptr<GPUUnitStream> r, cudaStream_t stream);
void delay_preliminary_prune(std::shared_ptr<GPUIntStream> s, std::shared_ptr<GPUUnitStream> r, cudaStream_t stream);
std::shared_ptr<GPUIntStream> slift(std::shared_ptr<GPUIntStream> x, std::shared_ptr<GPUIntStream> y, int op);
std::shared_ptr<GPUIntStream> lift(std::shared_ptr<GPUIntStream> x, std::shared_ptr<GPUIntStream> y, int op);
std::shared_ptr<GPUIntStream> count(std::shared_ptr<GPUUnitStream> input);
__global__ void assign_vals(int *input, int *result, int *input_offset, int *result_offset, int size);
__global__ void time_cuda(int* input_timestamp, int* output_timestamps, int* output_values, int size, int*offs, int* resultOffse);
__global__ void last_cuda(int* input_timestamp, int* input_values,int*unit_stream_timestamps,  int* output_timestamps, int* output_values,int IntStreamSize, int size, int* offsInt, int* offsUnit);
__global__ void final_reduce(int* block_red,int size,int* offset);

__global__ void delay_cuda_preliminary_prune(int *inputIntTimestamps, int *inputIntValues, int *resetTimestamps, int size, int resetSize, int *offset, int *resetOffset, cudaStream_t stream);
__global__ void delay_cuda(int *inputIntTimestamps, int *inputIntValues, int *resetTimestamps, int *results, int size, int inputSize, int *inputOffset, int *resetOffset, int* resultOffset, int maxTimestamp, cudaStream_t stream);
__device__ int lookUpElement(int size,int searchValue, int * input_timestamp);
__device__ int lookUpNextElement(int size,int searchValue, int * timestamps);
__global__ void calculate_offset(int* timestamps, int* offset, int size);
__global__ void lift_cuda(  int *x_ts, int *y_ts, int *out_ts, 
                            int *x_v, int *y_v, int *out_v,
                            int threads, int x_len, int y_len, 
                            int op, int *valid, int *invalid,
                            int *out_ts_cpy, int *out_v_cpy, int *invalid_offset,
                            int *x_offset, int *y_offset);
                            
__global__ void remove_invalid( int threads, int *invalid, int *valid, 
                                int x_len, int y_len, 
                                int *out_ts_cpy, int *out_ts, 
                                int *out_v_cpy, int *out_v,
                                int *x_offset_d, int *y_offset_d,
                                int *result_offset, int op);

__global__ void inval_multiples_merge(  int op, int threads,
                                        int x_len, int y_len,
                                        int *x_offset_d, int *y_offset_d,
                                        int *out_ts, int *invalid, int *valid);

__device__ void selection_sort( int *data, int left, int right );
__global__ void cdp_simple_quicksort(int *data, int left, int right, int depth );
#endif //TESSLA_CUDA_STREAMFUNCTIONS_CUH
