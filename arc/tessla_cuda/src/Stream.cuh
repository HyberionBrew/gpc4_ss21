//
// Created by fabian on 28.05.21.
//

#ifndef TESSLA_CUDA_STREAM_CUH
#define TESSLA_CUDA_STREAM_CUH

/**
 * Stream class, can contain either an GPUIntStream or a GPUUnitStream
 */
/*class Stream {
public:
    virtual void copy_to_device()=0;
    virtual void copy_to_host()=0;
    size_t size;
    bool OnDevice;
};*/

class GPUIntStream{
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

    int * host_offset;
    int * device_offset;
    bool onDevice;
    GPUIntStream(int *timestamp, int *value, size_t size);
    GPUIntStream();
    GPUIntStream(int *timestamp, int *value, size_t size, int offs);
    //just allocate on host

    void copy_to_device(bool valid);
    void copy_to_device();
    void copy_to_host();
    void free_device();
    GPUIntStream(bool deviceOnly, size_t size);
    void free_host();
    void print();
};

class GPUUnitStream  {
public:
    size_t size;
    bool onDevice;
    int * host_timestamp;
    int * device_timestamp;
    int * host_offset;
    int * device_offset;
    GPUUnitStream();
    GPUUnitStream(int *timestamp, size_t size);
    GPUUnitStream(int *timestamp, size_t size, int offs);
    void copy_to_device();
    void copy_to_device(bool valid);
    void copy_to_host();
    void free_host();
    void free_device();
    void print();
};
#endif //TESSLA_CUDA_STREAM_CUH
