//
// Created by fabian on 09/07/2021.
//

#include "ImmediateFunctions.cuh"
#include "StreamFunctionHelper.cuh"

/**
 * Add an immediate value to a stream.
 * @param input Input stream
 * @param imm Immediate value
 * @param stream Number of parallel CUDA operations
 * @return Modified integer stream
 */
shared_ptr<GPUIntStream> add_imm (shared_ptr<GPUIntStream> input, size_t imm, cudaStream_t stream) {
    // Calculate the GPU requirements
    int threads = input->size;
    int block_size = 1;
    int blocks = 1;
    calcThreadsBlocks(threads,&block_size,&blocks);

    // Calculate using GPU
    std::shared_ptr<GPUIntStream> result = std::make_shared<GPUIntStream>(input->size, true);
    add_imm_cuda <<<blocks, block_size, 0, stream>>> (input->device_values, imm, result->device_values);

    return result;
}

/**
 * Kernel; Add an immediate value to a stream.
 * @param input_ts input timestamp memory location
 * @param input_val input value memory location
 * @param imm immediate value
 * @param output_ts output timestamp memory location
 * @param output_val output value memory location
 */
__global__ void add_imm_cuda (const int *input_val, int imm, int *output_val) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    output_val[i] = input_val[i] + imm;
}




/**
 * Multiply an immediate value onto a stream.
 * @param input Input stream
 * @param imm Immediate value
 * @param stream Number of parallel CUDA operations
 * @return Modified integer stream
 */
shared_ptr<GPUIntStream> mul_imm (shared_ptr<GPUIntStream> input, size_t imm, cudaStream_t stream) {
    // Calculate the GPU requirements
    int threads = input->size;
    int block_size = 1;
    int blocks = 1;
    calcThreadsBlocks(threads,&block_size,&blocks);

    // Calculate using GPU
    std::shared_ptr<GPUIntStream> result = std::make_shared<GPUIntStream>(input->size, true);
    mul_imm_cuda <<<blocks, block_size, 0, stream>>> (input->device_values, imm, result->device_values);

    return result;
}

/**
 * Kernel; Multiply an immediate value onto a stream.
 * @param input_ts input timestamp memory location
 * @param input_val input value memory location
 * @param imm immediate value
 * @param output_ts output timestamp memory location
 * @param output_val output value memory location
 */
__global__ void mul_imm_cuda (const int *input_val, int imm, int *output_val) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    output_val[i] = input_val[i] * imm;
}




/**
 * Subtract an immediate value from a stream.
 * @param input Input stream
 * @param imm Immediate value
 * @param stream Number of parallel CUDA operations
 * @return Modified integer stream
 */
shared_ptr<GPUIntStream> sub_imm (shared_ptr<GPUIntStream> input, size_t imm, cudaStream_t stream) {
    // Calculate the GPU requirements
    int threads = input->size;
    int block_size = 1;
    int blocks = 1;
    calcThreadsBlocks(threads,&block_size,&blocks);

    // Calculate using GPU
    std::shared_ptr<GPUIntStream> result = std::make_shared<GPUIntStream>(input->size, true);
    sub_imm_cuda <<<blocks, block_size, 0, stream>>> (input->device_values, imm, result->device_values);

    return result;
}

/**
 * Kernel; Subtract an immediate value from a stream.
 * @param input_ts input timestamp memory location
 * @param input_val input value memory location
 * @param imm immediate value
 * @param output_ts output timestamp memory location
 * @param output_val output value memory location
 */
__global__ void sub_imm_cuda (const int *input_val, int imm, int *output_val) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    output_val[i] = input_val[i] - imm;
}




/**
 * Subtract a stream from an immediate value.
 * @param input Input stream
 * @param imm Immediate value
 * @param stream Number of parallel CUDA operations
 * @return Modified integer stream
 */
shared_ptr<GPUIntStream> sub_inv_imm (shared_ptr<GPUIntStream> input, size_t imm, cudaStream_t stream) {
    // Calculate the GPU requirements
    int threads = input->size;
    int block_size = 1;
    int blocks = 1;
    calcThreadsBlocks(threads,&block_size,&blocks);

    // Calculate using GPU
    std::shared_ptr<GPUIntStream> result = std::make_shared<GPUIntStream>(input->size, true);
    sub_inv_imm_cuda <<<blocks, block_size, 0, stream>>> (input->device_values, imm, result->device_values);

    return result;
}

/**
 * Kernel; Subtract a stream from an immediate value.
 * @param input_ts input timestamp memory location
 * @param input_val input value memory location
 * @param imm immediate value
 * @param output_ts output timestamp memory location
 * @param output_val output value memory location
 */
__global__ void sub_inv_imm_cuda (const int *input_val, int imm, int *output_val) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    output_val[i] = imm - input_val[i];
}




/**
 * Divide a stream by an immediate value
 * @param input Input stream
 * @param imm Immediate value
 * @param stream Number of parallel CUDA operations
 * @return Modified integer stream
 */
shared_ptr<GPUIntStream> div_imm (shared_ptr<GPUIntStream> input, size_t imm, cudaStream_t stream) {
    // Calculate the GPU requirements
    int threads = input->size;
    int block_size = 1;
    int blocks = 1;
    calcThreadsBlocks(threads,&block_size,&blocks);

    // Calculate using GPU
    std::shared_ptr<GPUIntStream> result = std::make_shared<GPUIntStream>(input->size, true);
    div_imm_cuda <<<blocks, block_size, 0, stream>>> (input->device_values, imm, result->device_values);

    return result;
}

/**
 * Kernel; Divide a stream by an immediate value
 * @param input_ts input timestamp memory location
 * @param input_val input value memory location
 * @param imm immediate value
 * @param output_ts output timestamp memory location
 * @param output_val output value memory location
 */
__global__ void div_imm_cuda (const int *input_val, int imm, int *output_val) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    output_val[i] = input_val[i] / imm;
}




/**
 * Divide an immediate value by a stream
 * @param input Input stream
 * @param imm Immediate value
 * @param stream Number of parallel CUDA operations
 * @return Modified integer stream
 */
shared_ptr<GPUIntStream> div_inv_imm (shared_ptr<GPUIntStream> input, size_t imm, cudaStream_t stream) {
    // Calculate the GPU requirements
    int threads = input->size;
    int block_size = 1;
    int blocks = 1;
    calcThreadsBlocks(threads,&block_size,&blocks);

    // Calculate using GPU
    std::shared_ptr<GPUIntStream> result = std::make_shared<GPUIntStream>(input->size, true);
    div_inv_imm_cuda <<<blocks, block_size, 0, stream>>> (input->device_values, imm, result->device_values);

    return result;
}

/**
 * Kernel; Divide an immediate value by a stream
 * @param input_ts input timestamp memory location
 * @param input_val input value memory location
 * @param imm immediate value
 * @param output_ts output timestamp memory location
 * @param output_val output value memory location
 */
__global__ void div_inv_imm_cuda (const int *input_val, int imm, int *output_val) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    output_val[i] = imm / input_val[i];
}




/**
 * Calculate a stream modulo an immediate value
 * @param input Input stream
 * @param imm Immediate value
 * @param stream Number of parallel CUDA operations
 * @return Modified integer stream
 */
shared_ptr<GPUIntStream> mod_imm (shared_ptr<GPUIntStream> input, size_t imm, cudaStream_t stream) {
    // Calculate the GPU requirements
    int threads = input->size;
    int block_size = 1;
    int blocks = 1;
    calcThreadsBlocks(threads,&block_size,&blocks);

    // Calculate using GPU
    std::shared_ptr<GPUIntStream> result = std::make_shared<GPUIntStream>(input->size, true);
    mod_imm_cuda <<<blocks, block_size, 0, stream>>> (input->device_values, imm, result->device_values);

    return result;
}

/**
 * Kernel; Calculate a stream modulo an immediate value
 * @param input_ts input timestamp memory location
 * @param input_val input value memory location
 * @param imm immediate value
 * @param output_ts output timestamp memory location
 * @param output_val output value memory location
 */
__global__ void mod_imm_cuda (const int *input_val, int imm, int *output_val) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    output_val[i] = input_val[i] % imm;
}




/**
 * Calculate an immediate value modulo a stream
 * @param input Input stream
 * @param imm Immediate value
 * @param stream Number of parallel CUDA operations
 * @return Modified integer stream
 */
shared_ptr<GPUIntStream> mod_inv_imm (shared_ptr<GPUIntStream> input, size_t imm, cudaStream_t stream) {
    // Calculate the GPU requirements
    int threads = input->size;
    int block_size = 1;
    int blocks = 1;
    calcThreadsBlocks(threads,&block_size,&blocks);

    // Calculate using GPU
    std::shared_ptr<GPUIntStream> result = std::make_shared<GPUIntStream>(input->size, true);
    mod_inv_imm_cuda <<<blocks, block_size, 0, stream>>> (input->device_values, imm, result->device_values);

    return result;
}

/**
 * Kernel; Calculate an immediate value modulo a stream
 * @param input_ts input timestamp memory location
 * @param input_val input value memory location
 * @param imm immediate value
 * @param output_ts output timestamp memory location
 * @param output_val output value memory location
 */
__global__ void mod_inv_imm_cuda (const int *input_val, int imm, int *output_val) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    output_val[i] = imm % input_val[i];
}