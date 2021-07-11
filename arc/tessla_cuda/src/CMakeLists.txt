include_directories(${PROJECT_SOURCE_DIR}/tessla_cuda/src)


add_library(tessla_cuda
        StreamFunctions.cuh
        GPUReader.cuh GPUReader.cu
        GPUStream.cuh GPUStream.cu
        GPUWriter.cuh GPUWriter.cu
        StreamFunctions.cu
        ImmediateFunctions.cu
        StreamFunctionHelper.cuh StreamFunctionHelper.cu
        helper.cuh
        device_information.cuh
        StreamFunctionsThrust.cuh StreamFunctionsThrust.cu
)

# Set CUDA target architectures
set_property(TARGET tessla_cuda PROPERTY CUDA_ARCHITECTURES 61 75)
set_target_properties(tessla_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)