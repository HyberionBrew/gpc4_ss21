//
// Created by fabian on 28.05.21.
//

#ifndef TESSLA_CUDA_STREAM_CUH
#define TESSLA_CUDA_STREAM_CUH

/**
 * Stream class, can contain either an IntStream or a UnitStream
 */
/*class Stream {
public:
    virtual void copy_to_device()=0;
    virtual void copy_to_host()=0;
    size_t size;
    bool OnDevice;
};*/

class IntStream{
public:
    //DO NOT USE DESTRUCTOR! LEADS TO ERROR IN CONJUNCTION WITH
    //CHECK(cudaDeviceReset());
    size_t size;
    bool OnDevice;
    //two pointers one to value one to timestamp
    int * host_timestamp;
    int * host_values;
    int * device_timestamp;
    int * device_values;

    IntStream(int *timestamp,int *value, size_t size);
    //just allocate on host
    IntStream(size_t size);

    void copy_to_device();
    void copy_to_host();
    void free_device();
    void free_host();
    void print();
};

class UnitStream  {
public:
    size_t size;
    bool OnDevice;
    int * host_timestamp;
    int * device_timestamp;
    UnitStream(int *timestamp, size_t size);
    void copy_to_device();
    void copy_to_host();
    void free_device();
    void print();
};
#endif //TESSLA_CUDA_STREAM_CUH
