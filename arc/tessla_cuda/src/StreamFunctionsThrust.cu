#include "StreamFunctions.cuh"
#include "StreamFunctionsThrust.cuh"
#include "helper.cuh"
#include "Stream.cuh"
#include <thrust/device_ptr.h>
#include <iostream>
#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/device_vector.h>
#include <thrust/count.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/unique.h>
#include <thrust/merge.h>
#include <thrust/functional.h>
#include <thrust/pair.h>

struct is_larger_zero
{
  __host__ __device__
  bool operator()(const int x)
  {
    return x > -1;
  }
};

std::shared_ptr<GPUIntStream> last_thrust(std::shared_ptr<GPUIntStream> inputInt, std::shared_ptr<GPUUnitStream>  inputUnit, cudaStream_t stream){
    
    //first cast device pointers to thrust pointers
    auto offsetInt = thrust::device_pointer_cast(inputInt->device_offset);
    auto offsetUnit = thrust::device_pointer_cast(inputUnit->device_offset);
    //shift for offset
    auto inputInt_timestamps = thrust::device_pointer_cast(inputInt->device_timestamp+*offsetInt);
    auto inputInt_values = thrust::device_pointer_cast(inputInt->device_values+*offsetInt);
    auto inputUnit_timestamps = thrust::device_pointer_cast(inputUnit->device_timestamp+*offsetUnit);
    //Standard guard

    std::shared_ptr<GPUIntStream> result = std::make_shared<GPUIntStream>();
    int sizeAllocated = inputUnit->size * sizeof(int);
    result->size = inputUnit->size;
    result->host_timestamp = (int *) malloc(inputUnit->size * sizeof(int));
    result->host_values = (int *) malloc(inputUnit->size * sizeof(int));

    // Check if we have enough memory left
    if (result->host_values == nullptr || result->host_timestamp == nullptr) {
        throw std::runtime_error("Out of memory.");
    }

    memset(result->host_timestamp, 0, sizeAllocated);
    memset(result->host_values, 0, sizeAllocated);
    result->copy_to_device(false);



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
    int rs = thrust::count(result_timestamps, result_timestamps+result->size-*offsetUnit, -1);
    
    thrust::gather(result_timestamps,result_timestamps+result->size-*offsetUnit,
                    inputInt_values,
                    result_values);

    //USE COPY_N ! otherwise unsafe
    thrust::copy_n(inputUnit_timestamps+ rs, result->size-rs-*offsetUnit, 
                    result_timestamps+rs);
    
    //final offset calculation
    rs = rs+*offsetUnit;
    thrust::fill(result_offs, result_offs + sizeof(int), (int) rs);
    return result;
}


struct is_smaller
{

  int _thresh;
    is_smaller(int thresh) : _thresh(thresh) { }

  __host__ __device__
  const bool operator()(const int &x) 
  {
    return x <= _thresh;
  }
};

thrust::device_vector<int> cross_streams(thrust::device_ptr<int> inputInt1_timestamps, thrust::device_ptr<int> inputInt2_timestamps,thrust::device_ptr<int> inputInt1_values,thrust::device_ptr<int> inputInt2_values,int size1,int size2){
    //*inputInt1_timestamps
    int input_shift =0;
    if (size1 > 0){
      int result = thrust::count_if(inputInt2_timestamps, inputInt2_timestamps+size2, is_smaller(*inputInt1_timestamps));
      //lower 0 guard
      input_shift = std::max(0,result-1);
    }
    else{
      input_shift = 0;
    }
    thrust::device_vector<int> fit_1(size2-input_shift);

    thrust::lower_bound(inputInt1_timestamps, inputInt1_timestamps+size1,
                      inputInt2_timestamps+input_shift, inputInt2_timestamps+size2,
                      fit_1.begin(),
                      thrust::less_equal<int>());
        //decrement by -1
    thrust::transform(fit_1.begin(),
                  fit_1.end(),
                  thrust::make_constant_iterator((1)),
                  fit_1.begin(),
                  thrust::minus<int>());
    //check if no value is before
    if(fit_1.size() == 0){
      thrust::device_vector<int> res(0);
      return res;
    }
    else if (fit_1.begin()[0] == -1){
      thrust::device_vector<int> res(0);
      return res;
    }

    //for(int i = 0; i < fit_1.size(); i++) {
    //   std::cout << "Fit[" << i << "] = " << fit_1[i] << std::endl;
    //}


    //printf("size_1: %d \n",size1);
    //printf("size_2: %d \n",size2);
    int size2_new = size2-input_shift;


    int out_of_range = thrust::count(fit_1.begin(), fit_1.end(), size1);
    thrust::fill(fit_1.end()-out_of_range,fit_1.end(),size1-1);
    thrust::device_vector<int> added_1(size2_new);
    thrust::gather(fit_1.begin(),fit_1.end(),
                  inputInt1_values,
                  added_1.begin());

    //for(int i = 0; i < added_1.size(); i++) {
    //   std::cout << "ADD[" << i << "] = " << added_1[i] << std::endl;
    //}

   if (size2 <= size1){
        //first add up for all valid values
        //printf("taken\n");
        //count values that would be outside of range, i.e. all values == size
        //printf("%d \n", out_of_range);
        thrust::transform(added_1.begin(), added_1.end(), inputInt2_values+input_shift,added_1.begin(), thrust::plus<int>());
        //thrust::transform(added_1.end()-out_of_range, added_1.end(), thrust::make_constant_iterator(*(inputInt2_values+size2)),added_1.end()-out_of_range, thrust::plus<int>());
        //now fill up remaining
        //thrust::transform(fit_1[size_fit1], fit_2,fit_2 +size_fit1, op);
        //inputInt2_values +result_values
    }
    else{
      //size_inputInt1 > size_inputIn2
      thrust::transform(added_1.begin(), added_1.end(), inputInt2_values+input_shift,added_1.begin(), thrust::plus<int>());
    }
 
    /*for(int i = 0; i < added_1.size(); i++) {
        std::cout << "ADDFInal[" << i << "] = " << added_1[i] << std::endl;
    }*/
  return added_1;
}

typedef thrust::tuple<int, int> tuple_t;
struct tupleEqual
{
  __host__ __device__
    bool operator()(tuple_t x, tuple_t y)
    {
      return ( (x.get<0>()== y.get<0>()) && (x.get<1>() == y.get<1>()) );
    }
};
//TODO! only supports adds
std::shared_ptr<GPUIntStream> slift_thrust(std::shared_ptr<GPUIntStream> inputInt1, std::shared_ptr<GPUIntStream> inputInt2,cudaStream_t stream){
    /*PREAMBLE*/
    auto offsetInt1 = thrust::device_pointer_cast(inputInt1->device_offset);
    auto offsetInt2 = thrust::device_pointer_cast(inputInt2->device_offset);
    //shift for offset
    auto inputInt1_timestamps = thrust::device_pointer_cast(inputInt1->device_timestamp+*offsetInt1);
    auto inputInt1_values = thrust::device_pointer_cast(inputInt1->device_values+*offsetInt1);
    auto inputInt2_timestamps = thrust::device_pointer_cast(inputInt2->device_timestamp+*offsetInt2);
    auto inputInt2_values = thrust::device_pointer_cast(inputInt2->device_values+*offsetInt2);
    //Standard guard
    //TODO! but in function @ cleanup!
    //std::shared_ptr<GPUIntStream> result = std::make_shared<GPUIntStream>();
   /* int sizeAllocated = (inputInt1->size+inputInt2->size)*sizeof(int);
    result->size = inputInt1->size+ inputInt2->size;
    result->host_timestamp = (int *) malloc(result->size * sizeof(int));
    result->host_values = (int *) malloc(result->size * sizeof(int));
    memset(result->host_timestamp, 0, sizeAllocated);
    memset(result->host_values, 0, sizeAllocated);
    printf("host offs %d \n", *result->host_offset);
    result->copy_to_device(false);
    auto result_values = thrust::device_pointer_cast(result->device_values);
    auto result_timestamps = thrust::device_pointer_cast(result->device_timestamp);
    
    int * x_h = (int *) malloc(sizeof(int));
    int * x_d;
    memset(x_h,0,sizeof(int));

    CHECK(cudaMalloc((int**)&x_d, sizeof(int)));
    //CHECK(cudaMemcpy(x_h, x_d, sizeof(int), cudaMemcpyHostToDevice));

    auto result_offs = thrust::device_pointer_cast(x_d);
    //auto result_offs = thrust::device_pointer_cast(result->device_offset);
    printf("result offs = %d \n",*result_offs);
    *result_offs = 0;
    printf("result offs = %d \n",*result_offs);
    //*result_offs = *offsetInt1 + *offsetInt2;*/
    std::shared_ptr<GPUIntStream> result = std::make_shared<GPUIntStream>();
    int sizeAllocated = (inputInt1->size +inputInt2->size)* sizeof(int);
    result->size = inputInt1->size +inputInt2->size;
    result->host_timestamp = (int *) malloc(sizeAllocated);
    result->host_values = (int *) malloc(sizeAllocated);

    // Check if we have enough memory left
    if (result->host_values == nullptr || result->host_timestamp == nullptr) {
        throw std::runtime_error("Out of memory.");
    }

    memset(result->host_timestamp, 0, sizeAllocated);
    memset(result->host_values, 0, sizeAllocated);
    result->copy_to_device(false);
    auto result_values = thrust::device_pointer_cast(result->device_values);
    auto result_timestamps = thrust::device_pointer_cast(result->device_timestamp);
    auto result_offs = thrust::device_pointer_cast(result->device_offset);
    printf("result_values = %d \n",result_timestamps[0]);
    printf("result offs = %d \n",result_offs[0]);

    //fill those that are not part of the current calc (since they are invalid) with -1
    thrust::fill(result_values,result_values+*result_offs,-1);
    thrust::fill(result_timestamps,result_timestamps+*result_offs,-1);

    //now only look at valid region
    //result_values = thrust::device_pointer_cast(result->device_values+*result_offs);
    //result_timestamps = thrust::device_pointer_cast(result->device_timestamp+*result_offs);

    /*FINISHED PREAMBLE*/

    //add one stream up

    //calc lower bounds
    int size_inputInt2 = inputInt2->size-*offsetInt2;
    int size_inputInt1 = inputInt1->size-*offsetInt1;

    thrust::device_vector<int> add = cross_streams(inputInt1_timestamps, inputInt2_timestamps,inputInt1_values,inputInt2_values, size_inputInt1,size_inputInt2);
    int shift_timestamps1 = size_inputInt2-add.size();
    thrust::device_vector<int> add2 = cross_streams(inputInt2_timestamps, inputInt1_timestamps,inputInt2_values,inputInt1_values, size_inputInt2,size_inputInt1);
    int shift_timestamps = size_inputInt1-add2.size();
    //inputInt2_timestamps += shift_timestamps1;
    //inputInt1_timestamps += shift_timestamps;
    //print both streams for debugging

    //now shift timestamps accordingly
    //merge into new vector array

    thrust::device_vector<int> merged_timestamps(add.size()+add2.size());

    thrust::device_vector<int> merged_values(add.size()+add2.size());
    thrust::merge_by_key(inputInt1_timestamps+shift_timestamps,inputInt1_timestamps+size_inputInt1,
                        inputInt2_timestamps+shift_timestamps1,inputInt2_timestamps+size_inputInt2,
                        add2.begin(), add.begin(),merged_timestamps.begin(),merged_values.begin());

    thrust::device_vector<int> timestamps_res(add.size()+add2.size());
    thrust::device_vector<int> values_res(add.size()+add2.size());
    //TODO! one of the following could in theroy be ommited 
    thrust::fill(timestamps_res.begin(),timestamps_res.end(),-1);
    thrust::fill(values_res.begin(),values_res.end(),-1);
    /*for (int i=0 ; i < merged_values.size();i++){
      std::cout << "merged_vals[" << i << "] = " <<merged_values[i] << std::endl;
    }*/
  
    thrust::pair<thrust::device_vector<int>::iterator, thrust::device_vector<int>::iterator> new_end = thrust::unique_by_key(merged_timestamps.begin(),merged_timestamps.end(),merged_values.begin());

    int length = thrust::distance(merged_timestamps.begin(),new_end.first);//thrust::count(values_res.begin(),values_res.end(),-1);

    printf("result size %d",result->size);
     printf("result legth %d",length);
    int rs = result->size-length;
    printf("result_ insied %d \n",rs);
    //result_offs[0] = rs;
    printf("result_ insied %d \n",rs );
    thrust::copy(merged_timestamps.begin(), merged_timestamps.end(), 
                  result_timestamps+rs);
    thrust::copy(merged_values.begin(), merged_values.end(), 
              result_values+rs);

    /*for (int i=rs ; i < result->size;i++){
      std::cout << "Final[" << i << "] = " <<result_timestamps[i] <<" | "<< result_values[i] << std::endl;
    }*/
    thrust::fill(result_offs, result_offs + sizeof(int), (int) rs);
    //std::cout <<"timestap1" <<result_timestamps[rs] << std::endl;
    std::cout <<"res offset" <<result_offs[0]<< std::endl;

    return result;

}