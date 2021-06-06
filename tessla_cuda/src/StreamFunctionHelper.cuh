//
// Created by fabian on 06.06.21.
//

#ifndef TESSLA_CUDA_STREAMFUNCTIONHELPER_CUH
#define TESSLA_CUDA_STREAMFUNCTIONHELPER_CUH
void calcThreadsBlocks(int threads, int *block_size, int*blocks);
__global__ void final_reduce(int* block_red,int size,int* offset);
__device__ void count_valid(int * sdata,int * output_timestamp,int* valid, int size, int MaxSize, unsigned int tid, const int i);
__device__ void merge_serial(int *a, int *b, int *c,
                             int a_start, int b_start,
                             int vpt, int tidx,
                             int a_len, int b_len);
#endif //TESSLA_CUDA_STREAMFUNCTIONHELPER_CUH
