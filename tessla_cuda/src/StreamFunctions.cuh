//
// Created by fabian on 28.05.21.
//

#ifndef TESSLA_CUDA_STREAMFUNCTIONS_CUH
#define TESSLA_CUDA_STREAMFUNCTIONS_CUH
#include <cuda_runtime.h>
#include "Stream.cuh"
// time operation for unit stream
void time(IntStream *s, IntStream *result, cudaStream_t stream);
void last(IntStream *d, UnitStream *r, IntStream *result,cudaStream_t stream);
void delay(IntStream *s, UnitStream *r, UnitStream*result, cudaStream_t stream);
__global__ void time_cuda(int* input_timestamp, int* output_timestamps, int* output_values, int size, int*offs, int* resultOffse);
__global__ void last_cuda(int* block_red, int* input_timestamp, int* input_values,int*unit_stream_timestamps,  int* output_timestamps, int* output_values,int IntStreamSize, int size, int* offsInt, int* offsUnit);
__global__ void final_reduce(int* block_red,int size,int* offset);

__global__ void delay_cuda(int* input_timestamp, int* input_values,int*unit_stream_timestamps,  int* output_timestamps, int* offsInt, int* offsUnit, int*offsResult,int size);
__device__ int lookUpElement(int size,int searchValue, int * input_timestamp);
#endif //TESSLA_CUDA_STREAMFUNCTIONS_CUH
