#include "StreamFunctions.cuh"
#include "StreamFunctionsThrust.cuh"
#include "helper.cuh"
#include "GPUStream.cuh"
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
#include <thrust/scan.h>

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

struct is_zero
{
  __host__ __device__
  bool operator()(const int x)
  {
    return (x == 0);
  }
};

//is used by slift_thrust()
thrust::device_vector<int> cross_streams(thrust::device_ptr<int> inputInt1_timestamps, thrust::device_ptr<int> inputInt2_timestamps,thrust::device_ptr<int> inputInt1_values,thrust::device_ptr<int> inputInt2_values,int size1,int size2,operations op, bool swap){
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


    int size2_new = size2-input_shift;

    //the
    int out_of_range = thrust::count(fit_1.begin(), fit_1.end(), size1);
    thrust::fill(fit_1.end()-out_of_range,fit_1.end(),size1-1);
    thrust::device_vector<int> added_1(size2_new);
    thrust::gather(fit_1.begin(),fit_1.end(),
                  inputInt1_values,
                  added_1.begin());

    int zeros = 0;
    if (swap){
      switch(op){
        case TH_OP_add:
          thrust::transform(added_1.begin(), added_1.end(), inputInt2_values+input_shift,added_1.begin(), thrust::plus<int>());
          break;
        case TH_OP_subtract:
          thrust::transform(added_1.begin(), added_1.end(), inputInt2_values+input_shift,added_1.begin(), thrust::minus<int>());
          break;
        case TH_OP_divide:
          zeros = thrust::count_if(inputInt2_timestamps+input_shift, inputInt2_timestamps+input_shift+(added_1.end() -added_1.begin()),is_zero());
          if (zeros!= 0) throw std::runtime_error("Division by Zero error");
          thrust::transform(added_1.begin(), added_1.end(), inputInt2_values+input_shift,added_1.begin(), thrust::divides<int>());
          break;
        case TH_OP_multiply:
          thrust::transform(added_1.begin(), added_1.end(), inputInt2_values+input_shift,added_1.begin(), thrust::multiplies<int>());
          break;
        case TH_OP_modulo:
          zeros = thrust::count_if(inputInt2_timestamps+input_shift, inputInt2_timestamps+input_shift+(added_1.end() -added_1.begin()),is_zero());
          if (zeros!= 0) throw std::runtime_error("Division by Zero error");
          thrust::transform(added_1.begin(), added_1.end(), inputInt2_values+input_shift,added_1.begin(), thrust::modulus<int>());
          break;
      }
    }
    else{
      switch(op){
        case TH_OP_add:
          thrust::transform(inputInt2_values+input_shift, inputInt2_values+input_shift+(added_1.end() -added_1.begin()),added_1.begin(), added_1.begin(), thrust::plus<int>());
          break;
        case TH_OP_subtract:
          thrust::transform(inputInt2_values+input_shift, inputInt2_values+input_shift+(added_1.end() -added_1.begin()),added_1.begin(), added_1.begin(), thrust::minus<int>());
          break;
        case TH_OP_divide:
          zeros = thrust::count_if(added_1.begin(),added_1.end(),is_zero());
          if (zeros!= 0) throw std::runtime_error("Division by Zero error");
          thrust::transform(inputInt2_values+input_shift, inputInt2_values+input_shift+(added_1.end() -added_1.begin()),added_1.begin(), added_1.begin(),  thrust::divides<int>());
          break;
        case TH_OP_multiply:
          thrust::transform(inputInt2_values+input_shift, inputInt2_values+input_shift+(added_1.end() -added_1.begin()),added_1.begin(), added_1.begin(),  thrust::multiplies<int>());
          break;
        case TH_OP_modulo:
          zeros = thrust::count_if(added_1.begin(),added_1.end(),is_zero());
          if (zeros!= 0) throw std::runtime_error("Division by Zero error");
          thrust::transform(inputInt2_values+input_shift, inputInt2_values+input_shift+(added_1.end() -added_1.begin()),added_1.begin(), added_1.begin(),  thrust::modulus<int>());
          break;
      }     
    }


  
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
std::shared_ptr<GPUIntStream> slift_thrust(std::shared_ptr<GPUIntStream> inputInt1, std::shared_ptr<GPUIntStream> inputInt2,operations op, cudaStream_t stream){
    /*PREAMBLE*/
    auto offsetInt1 = thrust::device_pointer_cast(inputInt1->device_offset);
    auto offsetInt2 = thrust::device_pointer_cast(inputInt2->device_offset);
    //shift for offset
    auto inputInt1_timestamps = thrust::device_pointer_cast(inputInt1->device_timestamp+*offsetInt1);
    auto inputInt1_values = thrust::device_pointer_cast(inputInt1->device_values+*offsetInt1);
    auto inputInt2_timestamps = thrust::device_pointer_cast(inputInt2->device_timestamp+*offsetInt2);
    auto inputInt2_values = thrust::device_pointer_cast(inputInt2->device_values+*offsetInt2);
    //Standard guard
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

    //fill those that are not part of the current calc (since they are invalid) with -1
    thrust::fill(result_values,result_values+*result_offs,-1);
    thrust::fill(result_timestamps,result_timestamps+*result_offs,-1);

    /*FINISHED PREAMBLE*/

    //calc lower bounds
    int size_inputInt2 = inputInt2->size-*offsetInt2;
    int size_inputInt1 = inputInt1->size-*offsetInt1;

    //fast path for merge
    if (op==TH_OP_merge){
       thrust::device_vector<int> merged_timestamps(size_inputInt2+size_inputInt1);

       thrust::device_vector<int> merged_values(size_inputInt2+size_inputInt1);
       thrust::merge_by_key(inputInt1_timestamps,inputInt1_timestamps+size_inputInt1,
                        inputInt2_timestamps,inputInt2_timestamps+size_inputInt2,
                        inputInt1_values, inputInt2_values,merged_timestamps.begin(),merged_values.begin());
    
      thrust::pair<thrust::device_vector<int>::iterator, thrust::device_vector<int>::iterator> new_end = thrust::unique_by_key(merged_timestamps.begin(),merged_timestamps.end(),merged_values.begin());

      int length = thrust::distance(merged_timestamps.begin(),new_end.first);
      int rs = result->size-length;
      thrust::copy(merged_timestamps.begin(), merged_timestamps.end(), 
                    result_timestamps+rs);
      thrust::copy(merged_values.begin(), merged_values.end(), 
                result_values+rs);

      /*for (int i=rs ; i < result->size;i++){
        std::cout << "Final[" << i << "] = " <<result_timestamps[i] <<" | "<< result_values[i] << std::endl;
      }*/
      thrust::fill(result_offs, result_offs + sizeof(int), (int) rs);
      return result;
    }


    thrust::device_vector<int> add = cross_streams(inputInt1_timestamps, inputInt2_timestamps,inputInt1_values,inputInt2_values, size_inputInt1,size_inputInt2,op,true);
    int shift_timestamps1 = size_inputInt2-add.size();
    thrust::device_vector<int> add2 = cross_streams(inputInt2_timestamps, inputInt1_timestamps,inputInt2_values,inputInt1_values, size_inputInt2,size_inputInt1,op,false);
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
  
    /*for (int i=0 ; i < merged_values.size();i++){
      std::cout << "merged_vals[" << i << "] = " <<merged_values[i] << std::endl;
    }*/
  
    thrust::pair<thrust::device_vector<int>::iterator, thrust::device_vector<int>::iterator> new_end = thrust::unique_by_key(merged_timestamps.begin(),merged_timestamps.end(),merged_values.begin());

    int length = thrust::distance(merged_timestamps.begin(),new_end.first);//thrust::count(values_res.begin(),values_res.end(),-1);

    //printf("result size %d",result->size);
    //printf("result legth %d",length);
    int rs = result->size-length;
    //printf("result_ insied %d \n",rs);
    //result_offs[0] = rs;
    //printf("result_ insied %d \n",rs );
    thrust::copy(merged_timestamps.begin(), merged_timestamps.end(), 
                  result_timestamps+rs);
    thrust::copy(merged_values.begin(), merged_values.end(), 
              result_values+rs);

    /*for (int i=rs ; i < result->size;i++){
      std::cout << "Final[" << i << "] = " <<result_timestamps[i] <<" | "<< result_values[i] << std::endl;
    }*/
    thrust::fill(result_offs, result_offs + sizeof(int), (int) rs);
    //std::cout <<"timestap1" <<result_timestamps[rs] << std::endl;
    //std::cout <<"res offset" <<result_offs[0]<< std::endl;
    return result;

}

struct delay_invalidate : public thrust::binary_function<int, int, int>
{
  int maxTimestamp;
  delay_invalidate(int max): maxTimestamp(max) {};
  __host__ __device__
  int operator()(int output, int nextReset) {
    if (nextReset < output || maxTimestamp < output)
      return -1;
    return output;
  }
};

struct delay_invalidated_filter {
  __host__ __device__
  bool operator()(thrust::tuple<int, int> t) {
    return t.get<1>() == -1;
  }
};

std::shared_ptr<GPUUnitStream> delay_thrust(std::shared_ptr<GPUIntStream> inputDelay, std::shared_ptr<GPUUnitStream> inputReset, cudaStream_t stream) {
    auto offsetDelay = thrust::device_pointer_cast(inputDelay->device_offset);
    auto offsetReset = thrust::device_pointer_cast(inputReset->device_offset);

    auto inputDelay_timestamps = thrust::device_pointer_cast(inputDelay->device_timestamp + *offsetDelay);
    auto inputDelay_values = thrust::device_pointer_cast(inputDelay->device_values + *offsetDelay);
    auto inputReset_timestamps = thrust::device_pointer_cast(inputReset->device_timestamp + *offsetReset);

    // Purge delay values that will never produce results
    // Find next reset event for each delay event
    thrust::device_vector<int> nextResets_indices(inputDelay->size - *offsetDelay);
    thrust::lower_bound(inputReset_timestamps, inputReset_timestamps + inputReset->size - *offsetReset,
                        inputDelay_timestamps, inputDelay_timestamps + inputDelay->size - *offsetDelay,
                        nextResets_indices.begin(),
                        thrust::less_equal<int>());
    thrust::device_vector<int> nextResets_timestamps(inputDelay->size - *offsetDelay);
    thrust::gather(nextResets_indices.begin(), nextResets_indices.end(),
                   inputReset_timestamps,
                   nextResets_timestamps.begin());
    // Calculate output timestamps for delay events
    thrust::device_vector<int> delay_outputs(inputDelay->size - *offsetDelay);
    thrust::transform(inputDelay_timestamps, inputDelay_timestamps + inputDelay->size - *offsetDelay,
                      inputDelay_values,
                      delay_outputs.begin(),
                      thrust::plus<int>());
    
    int maxTimestamp = inputDelay->host_timestamp[inputDelay->size - 1] > inputReset->host_timestamp[inputReset->size - 1] ? inputDelay->host_timestamp[inputDelay->size - 1] : inputReset->host_timestamp[inputReset->size - 1];
    thrust::device_vector<int> delay_timestamps(inputDelay->size - *offsetDelay);
    thrust::copy(inputDelay_timestamps, inputDelay_timestamps + inputDelay->size - *offsetDelay,
                 delay_timestamps.begin());
    thrust::transform(delay_outputs.begin(), delay_outputs.end(),
                      nextResets_timestamps.begin(),
                      delay_outputs.begin(),
                      delay_invalidate(maxTimestamp));

    auto inputOutput_begin = thrust::make_zip_iterator(thrust::make_tuple(delay_timestamps.begin(), delay_outputs.begin()));
    auto inputOutput_end = thrust::make_zip_iterator(thrust::make_tuple(delay_timestamps.end(), delay_outputs.end()));
    
    // Remove invalidated
    auto inputOutput_filteredEnd = thrust::remove_if(inputOutput_begin, inputOutput_end, delay_invalidated_filter());
    //std::cout << "removed " << inputOutput_end - inputOutput_filteredEnd << " elements" << std::endl;

    // Iteratively generate new outputs
    thrust::device_vector<int> iteration_input(inputReset->size);
    thrust::device_vector<int> iteration_output(inputReset->size);
    thrust::device_vector<int> results(inputDelay->size);
    thrust::device_vector<int> iteration_output_timestamps(inputReset->size); // don't care about those
    thrust::copy_n(inputReset_timestamps, inputReset->size, iteration_input.begin());

    thrust::detail::normal_iterator<thrust::device_ptr<int>> iteration_input_end = iteration_input.begin() + inputReset->size;
    thrust::detail::normal_iterator<thrust::device_ptr<int>> results_end = results.begin();
    int input_count = inputReset->size;
    while (input_count > 0) {
      // Calculate output timestamps generated by input events
      thrust::pair<thrust::detail::normal_iterator<thrust::device_ptr<int>>, thrust::detail::normal_iterator<thrust::device_ptr<int>>> end = thrust::set_intersection_by_key(
                                      delay_timestamps.begin(), delay_timestamps.begin() + (inputOutput_filteredEnd - inputOutput_begin),
                                      iteration_input.begin(), iteration_input_end,
                                      delay_outputs.begin(), 
                                      iteration_output_timestamps.begin(),
                                      iteration_output.begin()
                                      );
      // Use timestamps not yet present in results as input for next iteration
      iteration_input_end = thrust::set_difference(iteration_output.begin(), end.second,
                                                    results.begin(), results_end,
                                                    iteration_input.begin());
      // Add new timestamps to results
      results_end = thrust::set_union(results.begin(), results_end,
                                      iteration_input.begin(), iteration_input_end,
                                      results.begin());

      input_count = iteration_input_end - iteration_input.begin();
    }

    std::shared_ptr<GPUUnitStream> result = std::make_shared<GPUUnitStream>();
    int sizeAllocated = (results_end - results.begin()) * sizeof(int);
    result->size = results_end - results.begin();
    result->host_timestamp = (int*) malloc((results_end - results.begin()) * sizeof(int));
    
    if (result->host_timestamp == nullptr) {
        throw std::runtime_error("Out of memory.");
    }

    memset(result->host_timestamp, 0, sizeAllocated);
    result->copy_to_device(false);
    thrust::copy(results.begin(), results_end, result->device_timestamp);

    return result;
}

std::shared_ptr<GPUIntStream> count_thrust(std::shared_ptr<GPUUnitStream> input){
  auto offset = thrust::device_pointer_cast(input->device_offset);
  auto input_ts = thrust::device_pointer_cast(input->device_timestamp+*offset);

  std::shared_ptr<GPUIntStream> result = std::make_shared<GPUIntStream>();

  // Increase size by 1 to handle potentially new timestamp 0
  int size_alloc = (input->size + 1) * sizeof(int);
  result->size = size_alloc;
  cudaMalloc((void **) result->device_timestamp, (size_alloc)*sizeof(int));
  cudaMalloc((void **) result->device_values, (size_alloc)*sizeof(int));

  // create helper array for prefix sum
  int *helper;
  cudaMalloc((void **) helper, (input->size+1)*sizeof(int));
  auto helper_start = thrust::device_pointer_cast(helper+*offset);
  auto helper_end = thrust::device_pointer_cast(helper+input->size+1);
  thrust::fill(helper_start, helper_end, 1);

  // set first timestamp to 0 and then copy timestamps from input
  auto result_ts = thrust::device_pointer_cast(result->device_timestamp);
  result_ts[0] = 0;
  // copy device timestamps
  thrust::copy_n(input->device_timestamp, input->size, result->device_timestamp+1);

  // destination is at offset - 1, handle accordingly in if statement
  auto dest = thrust::device_pointer_cast(result->device_values+ *offset - 1);

  if (input_ts[0] == 0){
    // Result timestamps start with value 1 => inclusive prefix sum and same event count as input
    thrust::inclusive_scan(helper_start + 1, helper_end, dest + 1);

    // set device offset accordingly
    result->device_offset = input->device_offset + 1; 
  } else {
    // Result timestamps start with value 0 => exclusive prefix sum and one event more than input
    thrust::exclusive_scan(helper_start, helper_end, dest);

    // set device offset accordingly
    result->device_offset = input->device_offset; 
  }

  // free helper
  cudaFree(helper);

  return result;
}