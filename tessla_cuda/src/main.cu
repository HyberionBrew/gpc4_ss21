// The boilerplatte code is taken from tuwel

#include <cuda_runtime.h>
#include <sys/time.h>
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
    printf("Using Device %d: %s\n", dev, deviceProp.name);

    //create & allocate experimental streams
    int size = 10;

    int sizeAllocated = (size_t)size * sizeof(int);
    int * host_timestamp = (int *) malloc(size * sizeof(int));
    int * host_value = (int *) malloc(size* sizeof(int));
    for (int i = 0; i< size;i++) {
        *(host_timestamp+i) = i;
        *(host_value+i) = i;
    }
    //initially empty stream values
    int * host_timestampOut = (int *) malloc(size * sizeof(int));
    int * host_valueOut = (int *) malloc(size* sizeof(int));


    memset(host_timestampOut,0,sizeAllocated);
    memset(host_valueOut,0,sizeAllocated);

    IntStream inputStream(host_timestamp,host_value,size);
    IntStream outputStream(host_timestampOut,host_valueOut,size);

    inputStream.copy_to_device();
    outputStream.copy_to_device();

    time(&inputStream, &outputStream);

    //copy back and ouput
    outputStream.print();

    outputStream.copy_to_host();
    outputStream.print();
    outputStream.free_device();
    inputStream.free_device();
    free(host_timestampOut);
    free(host_valueOut);
    free(host_value);
    free(host_timestamp);
    CHECK(cudaDeviceReset()); //not working with destructor! --> I think we just shouldn't use destructor
    return(0);
}
