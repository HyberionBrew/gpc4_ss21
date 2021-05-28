//
// Created by fabian on 28.05.21.
//

#ifndef TESSLA_CUDA_HELPER_CUH
#define TESSLA_CUDA_HELPER_CUH
#include <stdio.h>
#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
    }                                                                          \
}

#endif //TESSLA_CUDA_HELPER_CUH
