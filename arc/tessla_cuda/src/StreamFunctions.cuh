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
#include <memory>
#include "GPUStream.cuh"

// Calling interface
std::shared_ptr<GPUIntStream> time(std::shared_ptr<GPUIntStream> input, cudaStream_t stream);
std::shared_ptr<GPUIntStream> time(std::shared_ptr<GPUUnitStream> input, cudaStream_t stream);
std::shared_ptr<GPUIntStream> last(std::shared_ptr<GPUIntStream> s, std::shared_ptr<GPUUnitStream> r, cudaStream_t stream);
std::shared_ptr<GPUUnitStream> delay(std::shared_ptr<GPUIntStream> s, std::shared_ptr<GPUUnitStream> r, cudaStream_t stream);
std::shared_ptr<GPUIntStream> slift(std::shared_ptr<GPUIntStream> x, std::shared_ptr<GPUIntStream> y, int op);
std::shared_ptr<GPUIntStream> lift(std::shared_ptr<GPUIntStream> x, std::shared_ptr<GPUIntStream> y, int op);
std::shared_ptr<GPUIntStream> count(std::shared_ptr<GPUUnitStream> input);

#endif //TESSLA_CUDA_STREAMFUNCTIONS_CUH
