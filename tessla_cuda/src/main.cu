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
    //call experimental stream function time

    int size = 10;

    printf("SIZE: %d \n",(size_t)size * sizeof(int));
    int sizeAllocated = (size_t)size * sizeof(int);
    int * timestamp = (int *) malloc(size * sizeof(int));
    int * value = (int *) malloc(size* sizeof(int));
    for (int i = 0; i< size;i++) {
        *(timestamp+i) = i;
        *(value+i) = i;
    }
    int * result = (int *) malloc(size* sizeof(int));

    int * timestampOut = (int *) malloc(size * sizeof(int));
    int * valueOut = (int *) malloc(size* sizeof(int));
    printf("SIZE: %d \n",(size_t)size * sizeof(int));
    memset(timestampOut,0,sizeAllocated);
    memset(valueOut,0,sizeAllocated);

    IntStream inputStream(timestamp,value,10);
    IntStream outputStream(timestampOut,valueOut,10);

    inputStream.copy_to_device();

    outputStream.copy_to_device();

   time(inputStream, outputStream);

   CHECK(cudaDeviceSynchronize());
   CHECK(cudaGetLastError());
   //printf("outputStream: %d \n",outputStream.timestamp_host[i]);
   //outputStream.copy_to_host();
   printf("ptr %d \n",outputStream.timestamp_device);
    CHECK(cudaMemcpy(result, outputStream.timestamp_device, sizeAllocated, cudaMemcpyDeviceToHost));
   for (int i = 0; i< size;i++) {
       printf("output: %d \n",outputStream.timestamp_host[i]);
   }
   //look at destructur cudaFree()
   //inputStream.free();
   //print(outputStream);
   //outputStream.free();*/
    free(timestamp);
    free(value);
    //CHECK(cudaDeviceReset()); //not working because of destructor
    return(0);
}
