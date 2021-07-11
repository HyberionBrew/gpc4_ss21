/**
 * @file ImmediateFunctions.cuh
 * @author Fabian Posch
 * @date 2021.07.09
 * @brief Immediate stream operations header
 *
 * Header file containing immediate stream operation library functions.
 * Implementation done using vanilla CUDA.
 * The CUDA kernel headers are not to be used.
 *
 */

#ifndef ARC_IMMEDIATEFUNCTIONS_CUH
#define ARC_IMMEDIATEFUNCTIONS_CUH

#include "GPUStream.cuh"
#include <memory>
#include <cuda_runtime.h>

using namespace std;

// Calling interface
shared_ptr<GPUIntStream> add_imm(shared_ptr<GPUIntStream> input, size_t imm, cudaStream_t stream);
shared_ptr<GPUIntStream> mul_imm (shared_ptr<GPUIntStream> input, size_t imm, cudaStream_t stream);
shared_ptr<GPUIntStream> sub_imm (shared_ptr<GPUIntStream> input, size_t imm, cudaStream_t stream);
shared_ptr<GPUIntStream> sub_inv_imm (shared_ptr<GPUIntStream> input, size_t imm, cudaStream_t stream);
shared_ptr<GPUIntStream> div_imm (shared_ptr<GPUIntStream> input, size_t imm, cudaStream_t stream);
shared_ptr<GPUIntStream> div_inv_imm (shared_ptr<GPUIntStream> input, size_t imm, cudaStream_t stream);
shared_ptr<GPUIntStream> mod_imm (shared_ptr<GPUIntStream> input, size_t imm, cudaStream_t stream);
shared_ptr<GPUIntStream> mod_inv_imm (shared_ptr<GPUIntStream> input, size_t imm, cudaStream_t stream);

// CUDA operations. Do not use externally
__global__ void add_imm_cuda (const int *input_val, int imm, int *output_val);
__global__ void mul_imm_cuda (const int *input_val, int imm, int *output_val);
__global__ void sub_imm_cuda (const int *input_val, int imm, int *output_val);
__global__ void sub_inv_imm_cuda (const int *input_val, int imm, int *output_val);
__global__ void div_imm_cuda (const int *input_val, int imm, int *output_val);
__global__ void div_inv_imm_cuda (const int *input_val, int imm, int *output_val);
__global__ void mod_imm_cuda (const int *input_val, int imm, int *output_val);
__global__ void mod_inv_imm_cuda (const int *input_val, int imm, int *output_val);

#endif //ARC_IMMEDIATEFUNCTIONS_CUH
