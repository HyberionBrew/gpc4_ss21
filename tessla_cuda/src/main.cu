// The boilerplatte code is taken from tuwel

#include <cuda_runtime.h>
#include <sys/time.h>
#include <cuda_profiler_api.h>
#include "main.cuh"
#include "helper.cuh"
#include "Stream.cuh"
#include "StreamFunctions.cuh"

void experimental_time(){

}



int main(int argc, char **argv) {

    printf("%s Starting...\n", argv[0]);
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    //might wanna derive MAX_THREADS and so on from here! TODO!
    printf("Using Device %d: %s\n", dev, deviceProp.name);

    //create & allocate experimental streams
    //still working for size = 1024*1024*10
    int size = 1024*1024;

    int sizeAllocated = (size_t)size * sizeof(int);
    int * host_timestamp = (int *) malloc(size * sizeof(int));
    int * host_value = (int *) malloc(size* sizeof(int));
    int * host_timestamp2 = (int *) malloc(size * sizeof(int));
    int * host_value2 = (int *) malloc(size* sizeof(int));
    for (int i = 0; i< size;i++) {
        *(host_timestamp2+i) = i;
        *(host_value2+i) = i;
    }
    //initially empty stream values
    int * host_timestampOut = (int *) malloc(size * sizeof(int));
    int * host_valueOut = (int *) malloc(size* sizeof(int));
    int * host_timestampOut2 = (int *) malloc(size * sizeof(int));
    int * host_valueOut2 = (int *) malloc(size* sizeof(int));
    CHECK(cudaProfilerStart());

    memset(host_timestampOut2,0,sizeAllocated);
    memset(host_valueOut2,0,sizeAllocated);
    memset(host_timestampOut,0,sizeAllocated);
    memset(host_valueOut,0,sizeAllocated);

    IntStream inputStream(host_timestamp,host_value,size);
    IntStream outputStream(host_timestampOut,host_valueOut,size);
    IntStream inputStream2(host_timestamp2,host_value2,size);
    IntStream outputStream2(host_timestampOut2,host_valueOut2,size);
    // create streams for parallel kernel launches
    int MAX_STREAMS = 16; // check if this is really max
    // I think we can +2 streams for in/out sync? but not sure

    // IMPORTANT! NO CONCURRENCY BETWEEN KERNELS POSSIBLE if:
    // 3.2.6.5.4. Implicit Synchronization https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
    //a page-locked host memory allocation,
    //a device memory allocation,
    //a device memory set,
    //a memory copy between two addresses to the same device memory,
    //any CUDA command to the NULL stream,
    //        a switch between the L1/shared memory configurations described in Compute Capability 3.x and Compute Capability 7.x.

    cudaStream_t stream[MAX_STREAMS];
    for (int i = 0; i < MAX_STREAMS; ++i)
        cudaStreamCreate(&stream[i]);

    //end config
    inputStream.copy_to_device();
    outputStream.copy_to_device();
    inputStream2.copy_to_device();
    outputStream2.copy_to_device();
    time(&inputStream, &outputStream, stream[0]);
    time(&inputStream2, &outputStream2, stream[1]);
    //copy back and output
    //outputStream.print();
    outputStream.copy_to_host();
    //outputStream.print();
    outputStream2.free_device();
    inputStream2.free_device();
    outputStream.free_device();
    inputStream.free_device();

    free(host_timestampOut2);
    free(host_valueOut2);
    free(host_value2);
    free(host_timestamp2);

    free(host_timestampOut);
    free(host_valueOut);
    free(host_value);
    free(host_timestamp);

    for (int i = 0; i < MAX_STREAMS; ++i)
        cudaStreamDestroy(stream[i]);
    //not working with destructor! --> I think we just shouldn't use destructor
    CHECK(cudaProfilerStop());
    CHECK(cudaDeviceReset());


    return(0);
}
