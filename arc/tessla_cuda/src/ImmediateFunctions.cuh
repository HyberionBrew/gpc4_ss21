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

#endif //ARC_IMMEDIATEFUNCTIONS_CUH
