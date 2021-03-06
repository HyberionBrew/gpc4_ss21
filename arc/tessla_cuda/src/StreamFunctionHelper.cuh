//
// Created by fabian on 06.06.21.
//

#ifndef TESSLA_CUDA_STREAMFUNCTIONHELPER_CUH
#define TESSLA_CUDA_STREAMFUNCTIONHELPER_CUH

typedef void (*lift_op) (int*, int*, int*);

typedef void (*lift_func) ( int*, int*, 
                            int*, int*, 
                            int*, int*,
                            int*, int*,
                            bool, bool, lift_op);

void calcThreadsBlocks(int threads, int *block_size, int*blocks);
__global__ void final_reduce(int* block_red,int size,int* offset);
__device__ void count_valid(int * sdata,int * output_timestamp,int* valid, int size, int MaxSize, unsigned int tid, const int i);
__device__ void lift_partition( int *x_ts, int *y_ts, int *out_ts,
                                int *x_v, int *y_v, int *out_v,
                                int x_start, int y_start,
                                int vpt, int tidx,
                                int x_len, int y_len,
                                lift_func fct, lift_op op, 
                                int *valid, int *invalid);
                                
__device__ void selection_sort( int *data, int left, int right );
__global__ void cdp_simple_quicksort(int *data, int left, int right, int depth );
#endif //TESSLA_CUDA_STREAMFUNCTIONHELPER_CUH
