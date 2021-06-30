//
// Created by fabian on 28.06.17.
//

#ifndef TESSLA_CUDA_STREAMFUNCTIONSTHRUST_CUH
#define STREAMFUNCTIONSTHRUST_CUH
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
enum operations {add,substract,multiply,divide,modulo,merge};

std::shared_ptr<GPUIntStream> last_thrust(std::shared_ptr<GPUIntStream>  inputInt, std::shared_ptr<GPUUnitStream>  inputUnit, cudaStream_t stream);
std::shared_ptr<GPUIntStream> slift_thrust(std::shared_ptr<GPUIntStream> inputInt1, std::shared_ptr<GPUIntStream> inputInt2, operations op, cudaStream_t stream);
#endif //TESSLA_CUDA_STREAMFUNCTIONSTHRUST_CUH
