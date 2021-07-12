//
// Created by klaus on 09.07.21.
//

#include "ImmediateFunctionsThrust.cuh"
#include "StreamFunctionHelper.cuh"
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>


enum ImmOp {
    ADD,
    MUL,
    DIVI,
    DIVII,
    SUBI,
    SUBII,
    MODI,
    MODII
};

struct immFunctor {
    size_t imm;
    ImmOp op;
    immFunctor(size_t _imm, ImmOp _op) {
        imm = _imm;
        op = _op;
    }
    __host__ __device__
    int operator()(int input) const {
        switch (op) {
            case ADD:
                return input + imm;
            case MUL:
                return input * imm;
            case SUBI:
                return input - imm;
            case SUBII:
                return imm - input;
            case DIVI:
                return input / imm;
            case DIVII:
                return imm / input;
            case MODI:
                return input % imm;
            case MODII:
                return imm % input;
        }
    }
};

shared_ptr<GPUIntStream> exec_imm_op(shared_ptr<GPUIntStream> input, size_t imm, ImmOp op) {
    // prepare result
    shared_ptr<GPUIntStream> result = make_shared<GPUIntStream>();
    cudaMalloc((void**) result->device_timestamp, input->size*sizeof(int));
    cudaMalloc((void**) result->device_values, input->size*sizeof(int));
    result->device_offset = input->device_offset;

    // copy timestamps
    auto input_ts = thrust::device_pointer_cast(input->device_timestamp);
    auto result_ts = thrust::device_pointer_cast(result->device_timestamp);
    thrust::copy_n(input_ts, input->size, result_ts);

    // get pointers and transform stream
    auto offset = thrust::device_pointer_cast(input->device_offset);
    auto input_vals_start = thrust::device_pointer_cast(input->device_values+*offset);
    auto input_vals_end = thrust::device_pointer_cast(input->device_values+ input->size);
    auto result_vals = thrust::device_pointer_cast(input->device_values + *offset);
    immFunctor f(imm, op);
    thrust::transform(input_vals_start, input_vals_end, result_vals, f);
    return result;
}

shared_ptr<GPUIntStream> add_imm_thrust(shared_ptr<GPUIntStream> input, size_t imm) {
    return exec_imm_op(input, imm, ADD);
}

shared_ptr<GPUIntStream> mul_imm_thrust (shared_ptr<GPUIntStream> input, size_t imm) {
    return exec_imm_op(input, imm, MUL);
}

shared_ptr<GPUIntStream> sub_imm_thrust (shared_ptr<GPUIntStream> input, size_t imm) {
    return exec_imm_op(input, imm, SUBI);
}

shared_ptr<GPUIntStream> sub_inv_imm_thrust (shared_ptr<GPUIntStream> input, size_t imm) {
    return exec_imm_op(input, imm, SUBII);
}

shared_ptr<GPUIntStream> div_imm_thrust (shared_ptr<GPUIntStream> input, size_t imm) {
    return exec_imm_op(input, imm, DIVI);
}

shared_ptr<GPUIntStream> div_inv_imm_thrust (shared_ptr<GPUIntStream> input, size_t imm){
    return exec_imm_op(input, imm, DIVII);
}

shared_ptr<GPUIntStream> mod_imm_thrust (shared_ptr<GPUIntStream> input, size_t imm){
    return exec_imm_op(input, imm, MODI);
}

shared_ptr<GPUIntStream> mod_inv_imm_thrust (shared_ptr<GPUIntStream> input, size_t imm){
    return exec_imm_op(input, imm, MODII);
}
