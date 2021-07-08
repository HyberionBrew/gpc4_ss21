include_directories(${PROJECT_SOURCE_DIR}/tessla_cuda/src)


add_library(tessla_cuda
        StreamFunctions.cuh
        GPUReader.cuh GPUReader.cu
        Stream.cuh Stream.cu
        StreamFunctions.cu
        StreamFunctionHelper.cuh StreamFunctionHelper.cu
        helper.cuh
        device_information.cuh
        StreamFunctionsThrust.cuh StreamFunctionsThrust.cu
)

add_executable(main
        StreamFunctions.cuh
        GPUReader.cuh GPUReader.cu
        Stream.cuh Stream.cu
        StreamFunctions.cu
        StreamFunctionHelper.cuh StreamFunctionHelper.cu
        helper.cuh
        device_information.cuh
        StreamFunctionsThrust.cuh StreamFunctionsThrust.cu
        main.cu main.cuh
        )

# Set CUDA target architectures
set_property(TARGET tessla_cuda PROPERTY CUDA_ARCHITECTURES 61 75)