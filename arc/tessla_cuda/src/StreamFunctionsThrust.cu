#include "StreamFunctions.cuh"
#include "StreamFunctionsThrust.cuh"
#include "helper.cuh"
#include "Stream.cuh"
#include <thrust/device_ptr.h>
#include <iostream>
#include <thrust/functional.h>
#include <thrust/gather.h>

struct is_larger_zero
{
  __host__ __device__
  bool operator()(const int x)
  {
    return x > -1;
  }
};

void last_thrust(GPUIntStream *inputInt, GPUUnitStream *inputUnit, GPUIntStream *result, cudaStream_t stream){
    
    //first cast device pointers to thrust pointers
    auto offsetInt = thrust::device_pointer_cast(inputInt->device_offset);
    auto offsetUnit = thrust::device_pointer_cast(inputUnit->device_offset);
    //shift for offset
    auto inputInt_timestamps = thrust::device_pointer_cast(inputInt->device_timestamp+*offsetInt);
    auto inputInt_values = thrust::device_pointer_cast(inputInt->device_values+*offsetInt);
    auto inputUnit_timestamps = thrust::device_pointer_cast(inputUnit->device_timestamp+*offsetUnit);
    
    //Standard guard
    if (!result->onDevice) {
        int sizeAllocated = inputUnit->size * sizeof(int);
        result->size = inputUnit->size;
        result->host_timestamp = (int *) malloc(inputUnit->size * sizeof(int));
        result->host_values = (int *) malloc(inputUnit->size * sizeof(int));
        memset(result->host_timestamp, 0, sizeAllocated);
        memset(result->host_values, 0, sizeAllocated);
        result->copy_to_device(false);
    }

    auto result_values = thrust::device_pointer_cast(result->device_values);
    auto result_timestamps = thrust::device_pointer_cast(result->device_timestamp);
    auto result_offs = thrust::device_pointer_cast(result->device_offset);
    
    //fill those that are not part of the current calc (since they are invalid) with -1
    thrust::fill(result_values,result_values+*offsetUnit,-1);
    thrust::fill(result_timestamps,result_timestamps+*offsetUnit,-1);
    
    //now only look at valid region
    result_values = thrust::device_pointer_cast(result->device_values+*offsetUnit);
    result_timestamps = thrust::device_pointer_cast(result->device_timestamp+*offsetUnit);
    
    //Actual algorithm starts here!
    thrust::lower_bound(inputInt_timestamps, inputInt_timestamps+inputInt->size-*offsetInt,
                    inputUnit_timestamps, inputUnit_timestamps+inputUnit->size-*offsetUnit, 
                    result_timestamps,
                    thrust::less<int>());
    //decrement by -1
    thrust::transform(result_timestamps,
                  result_timestamps+result->size-*offsetUnit,
                  thrust::make_constant_iterator((1)),
                  result_timestamps,
                  thrust::minus<int>());
    
    //calculate new additional offset
    *result_offs = thrust::count(result_timestamps, result_timestamps+result->size-*offsetUnit, -1);
    
    thrust::gather(result_timestamps,result_timestamps+result->size-*offsetUnit,
                    inputInt_values,
                    result_values);

    //USE COPY_N ! otherwise unsafe
    thrust::copy_n(inputUnit_timestamps+ *result_offs, result->size-*result_offs-*offsetUnit, 
                    result_timestamps+*result_offs);
    
    //final offset calculation
    *result_offs = *result_offs+*offsetUnit;

}