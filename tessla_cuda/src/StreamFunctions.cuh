//
// Created by fabian on 28.05.21.
//

#ifndef TESSLA_CUDA_STREAMFUNCTIONS_CUH
#define TESSLA_CUDA_STREAMFUNCTIONS_CUH
#include "Stream.cuh"
// time operation for unit stream
void time(IntStream *s, IntStream *result, cudaStream_t stream);
void last(IntStream *d, UnitStream *r, IntStream *result,cudaStream_t stream);

__global__ void time_cuda(int* input_timestamp, int* output_timestamps, int* output_values, int size, int*offs, int* resultOffse);
__global__ void last_cuda(int* block_red, int* input_timestamp, int* input_values,int*unit_stream_timestamps,  int* output_timestamps, int* output_values,int IntStreamSize, int size);
__global__ void final_reduce(int* block_red,int size,int* offset);
#endif //TESSLA_CUDA_STREAMFUNCTIONS_CUH
