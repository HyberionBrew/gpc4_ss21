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
#include "Stream.cuh"
// time operation for unit stream
int simp_compare(const void *a, const void *b); // TODO: Delete when no longer needed
void calcThreadsBlocks(int threads, int *block_size, int*blocks);
void time(IntStream *s, IntStream *result, cudaStream_t stream);
void last(IntStream *d, UnitStream *r, IntStream *result,cudaStream_t stream);
void delay(IntStream *s, UnitStream *r, UnitStream*result, cudaStream_t stream);
void delay_preliminary_prune(IntStream *s, UnitStream *r, cudaStream_t stream);
void lift(IntStream *x, IntStream *y, IntStream *result, int threads, int op);
__global__ void time_cuda(int* input_timestamp, int* output_timestamps, int* output_values, int size, int*offs, int* resultOffse);
__global__ void last_cuda(int* input_timestamp, int* input_values,int*unit_stream_timestamps,  int* output_timestamps, int* output_values,int IntStreamSize, int size, int* offsInt, int* offsUnit);
__global__ void final_reduce(int* block_red,int size,int* offset);

__global__ void delay_cuda_preliminary_prune(int *inputIntTimestamps, int *inputIntValues, int *resetTimestamps, int size, int resetSize, int *offset, int *resetOffset, cudaStream_t stream);
__global__ void delay_cuda(int *inputIntTimestamps, int *inputIntValues, int *resetTimestamps, int *results, int size, int inputSize, int *inputOffset, int *resetOffset, int* resultOffset, cudaStream_t stream);
__device__ int lookUpElement(int size,int searchValue, int * input_timestamp);
__device__ int lookUpNextElement(int size,int searchValue, int * timestamps);
__global__ void calculate_offset(int* timestamps, int* offset, int size);
__global__ void lift_cuda(  int *x_ts, int *y_ts, int *out_ts, 
                            int *x_v, int *y_v, int *out_v,
                            int threads, int x_len, int y_len, 
                            int op, int *valid, int *invalid,
                            int *out_ts_cpy, int *out_v_cpy, int *invalid_offset);
#endif //TESSLA_CUDA_STREAMFUNCTIONS_CUH
