//
// Created by fabian on 06.06.21.
//

#ifndef TESSLA_CUDA_STREAMFUNCTIONHELPER_CUH
#define TESSLA_CUDA_STREAMFUNCTIONHELPER_CUH
void calcThreadsBlocks(int threads, int *block_size, int*blocks);
__global__ void final_reduce(int* block_red,int size,int* offset);
__device__ void count_valid(int * sdata,int * output_timestamp,int* valid, int size, int MaxSize, unsigned int tid, const int i);
__device__ void lift_partition( int *x_ts, int *y_ts, int *out_ts,
                                int *x_v, int *y_v, int *out_v,
                                int x_start, int y_start,
                                int vpt, int tidx,
                                int x_len, int y_len,
                                lift_func fct, lift_op op);
                                
#endif //TESSLA_CUDA_STREAMFUNCTIONHELPER_CUH
