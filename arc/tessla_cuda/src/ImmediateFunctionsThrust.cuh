//
// Created by klaus on 09.07.21.
//

#ifndef ARC_IMMEDIATEFUNCTIONSTHRUST_CUH
#define ARC_IMMEDIATEFUNCTIONSTHRUST_CUH

#include "GPUStream.cuh"
#include <memory>
#include <cuda_runtime.h>

using namespace std;

// Calling interface
shared_ptr<GPUIntStream> add_imm_thrust(shared_ptr<GPUIntStream> input, size_t imm);
shared_ptr<GPUIntStream> mul_imm_thrust (shared_ptr<GPUIntStream> input, size_t imm);
shared_ptr<GPUIntStream> sub_imm_thrust (shared_ptr<GPUIntStream> input, size_t imm);
shared_ptr<GPUIntStream> sub_inv_imm_thrust (shared_ptr<GPUIntStream> input, size_t imm);
shared_ptr<GPUIntStream> div_imm_thrust (shared_ptr<GPUIntStream> input, size_t imm);
shared_ptr<GPUIntStream> div_inv_imm_thrust (shared_ptr<GPUIntStream> input, size_t imm);
shared_ptr<GPUIntStream> mod_imm_thrust (shared_ptr<GPUIntStream> input, size_t imm);
shared_ptr<GPUIntStream> mod_inv_imm_thrust (shared_ptr<GPUIntStream> input, size_t imm);

#endif //ARC_IMMEDIATEFUNCTIONSTHRUST_CUH
