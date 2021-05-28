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

    size_t size;
    bool OnDevice;
    //two pointers one to value one to timestamp
    int * timestamp_host;
    int * value_host;
    int * timestamp_device;
    int * value_device;

    IntStream(int *timestamp,int *value, size_t size);
    //just allocate on host
    IntStream(size_t size);
    ~IntStream();
    void copy_to_device();
    void copy_to_host();
};

class UnitStream  {
public:
    size_t size;
    bool OnDevice;
    int * timestamp_host;
    int * timestamp_device;
    UnitStream(int *timestamp, size_t size, bool OnDevice);
    void copy_to_device();
    void copy_to_host();
};
#endif //TESSLA_CUDA_STREAM_CUH
