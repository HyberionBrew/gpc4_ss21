#include "StreamFunctions.cuh"
#include "StreamFunctionsThrust.cuh"
#include "helper.cuh"
#include "Stream.cuh"
#include <thrust/device_ptr.h>
#include <iostream>
#include <thrust/functional.h>

void last_thrust(IntStream *inputInt, UnitStream *inputUnit, IntStream *result, cudaStream_t stream){
    //first cast device pointers to thrust pointers
    thrust::device_ptr<int> inputInt_timestamps(inputInt->device_timestamp);
    thrust::device_ptr<int> inputInt_values(inputInt->device_values);
    thrust::device_ptr<int> inputUnit_timestamps(inputUnit->device_timestamp);
  
  
    thrust::device_ptr<int> offsetInt(inputInt->device_offset);
    thrust::device_ptr<int> offsetUnit(inputUnit->device_offset);
  
  
    thrust::device_vector<double> inputInt_timestamps_vec(inputInt_timestamps, inputInt_timestamps + inputInt->size); 
    thrust::device_vector<double> inputUnit_timestamps_vec(inputUnit_timestamps, inputUnit_timestamps + inputUnit->size); 
    thrust::device_vector<double> inputInt_timestamps_vec(inputInt_timestamps, inputInt_timestamps + inputInt->size); 
    thrust::device_vector<double> inputUnit_timestamps_vec(inputUnit_timestamps, inputUnit_timestamps + inputUnit->size); 
  
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
    thrust::device_vector<double> result_timestamps_vec(result_timestamps, result_timestamps + result->size); 
    
    
    //actuall algo
    //calculate where before where the ints could be inserted
    thrust::lower_bound(inputInt_timestamps_vec.begin()+*offsetInt, inputInt_timestamps_vec.end(),
                    inputUnit_timestamps_vec.begin()+*offsetUnit, inputUnit_timestamps_vec.end(), 
                    result_timestamps_vec.begin(),
                    thrust::less<int>());
    for(int i = 0; i < result_timestamps_vec.size(); i++)
        std::cout << "D[" << i << "] = " << result_timestamps_vec[i] << std::endl;
    //decrement by -1
    //thrust::for_each(thrust::device, vec.begin(), vec.end(), _1 -= val);
    //Thrust permutation iterator to set each to index as currently in timestamps (timestamps-1)
    typedef thrust::device_vector<float>::iterator ElementIterator;
    typedef thrust::device_vector<int>::iterator   IndexIterator;
    //https://thrust.github.io/doc/classthrust_1_1permutation__iterator.html
    //thrust::permutation_iterator<ElementIterator,IndexIterator> iter(values.begin(), indices.begin());


    //cast back
    result->device_timestamp = thrust::raw_pointer_cast(result_timestamps);
    result->device_values = thrust::raw_pointer_cast(result_values);

}