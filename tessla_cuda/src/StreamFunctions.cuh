//
// Created by fabian on 28.05.21.
//

#ifndef TESSLA_CUDA_STREAMFUNCTIONS_CUH
#define TESSLA_CUDA_STREAMFUNCTIONS_CUH
#include "Stream.cuh"
// time operation for unit stream
void time(IntStream s, IntStream result);
__global__ void time_cuda(int* input_timestamp, int* output_timestamps, int* output_values, int size);
#endif //TESSLA_CUDA_STREAMFUNCTIONS_CUH
