#include "StreamFunctions.cuh"
#include "StreamFunctionsThrust.cuh"
#include "helper.cuh"
#include "Stream.cuh"
#include <thrust/device_ptr.h>

void last_thrust(IntStream *inputInt, UnitStream *inputUnit, IntStream *result, cudaStream_t stream){
    //first cast device pointers to thrust pointers
    thrust::device_ptr<int> inputInt_timestamps(inputInt->device_timestamp);
    thrust::device_ptr<int> inputInt_values(inputInt->device_values);
    thrust::device_ptr<int> inputUnit_timestamps(inputUnit->device_timestamp);
    if (!result->onDevice) {
        //TODO! where do we free this?
        int sizeAllocated = inputUnit->size * sizeof(int);
        result->size = inputUnit->size;
        result->host_timestamp = (int *) malloc(inputUnit->size * sizeof(int));
        result->host_values = (int *) malloc(inputUnit->size * sizeof(int));
        memset(result->host_timestamp, 0, sizeAllocated);
        memset(result->host_values, 0, sizeAllocated);
        result->copy_to_device(false);
    }
    thrust::device_ptr<int> result_values(result->device_values);
    thrust::device_ptr<int> result_timestamps(result->device_timestamp);
    
    //cast back
    result->device_timestamp = thrust::raw_pointer_cast(result_timestamps);
    result->device_values = thrust::raw_pointer_cast(result_values);

}