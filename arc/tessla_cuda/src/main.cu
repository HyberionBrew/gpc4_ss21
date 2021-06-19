// The boilerplatte code is taken from tuwel

#include <cuda_runtime.h>
#include <sys/time.h>
#include <cuda_profiler_api.h>
#include "main.cuh"
#include "helper.cuh"
#include "Stream.cuh"
#include "StreamFunctions.cuh"
#include "StreamFunctionsThrust.cuh"
void experimental_time(){

}

void test_slift(){
    printf("slift test\n");
    int sx = 50;
    int sy = 100;
    int *x_v = (int*)malloc(sx*sizeof(int));
    int *y_v = (int*)malloc(sy*sizeof(int));
    int *x_ts = (int*)malloc(sx*sizeof(int));
    int *y_ts = (int*)malloc(sy*sizeof(int));

    //int *res_ts = (int*)malloc((sx+sy)*sizeof(int));
    //int *res_v = (int*)malloc((sx+sy)*sizeof(int));

    for (int i=0; i<sx; i++){
        x_ts[i] = i;
        x_v[i] = 2*i;
    }
    
    for (int i=0; i<sy; i++){
        y_ts[i] = i;
        y_v[i] = i;
    }

    IntStream x(x_ts, x_v, sx);
    IntStream y(y_ts, y_v, sy);
    IntStream res;

    x.copy_to_device();
    y.copy_to_device();

    slift(&x,&y,&res, ADD);
    res.copy_to_host();

    x.print();
    y.print();
    res.print();
}


void experimental_last(){
    printf("--------- experimental last start -----------\n");
    int size = 5;

    int sizeAllocated = (size_t)size * sizeof(int);
    int * host_timestamp = (int *) malloc(size * sizeof(int));
    int * host_unit_timestamp = (int *) malloc(size * sizeof(int));
    int * host_value = (int *) malloc(size* sizeof(int));


    for (int i = 0; i< size;i++) {
        *(host_timestamp+i) = i+10;
        *(host_unit_timestamp+i) = i;
        *(host_value+i) = i;
    }

    *(host_timestamp) = INT_MIN;
    *(host_timestamp+1) = 6;
    *(host_timestamp+2) = 8;
    *(host_value) = INT_MIN;
    *(host_value+1) = 3;
    *(host_value+2) = 6;

    *(host_unit_timestamp) = INT_MIN;
    *(host_unit_timestamp+1) = 2;
    *(host_unit_timestamp+2) = 4;
    *(host_unit_timestamp+3) = 5;
    *(host_unit_timestamp+4) = 9;

    int * host_timestampOut = (int *) malloc(size * sizeof(int));
    int * host_valueOut = (int *) malloc(size* sizeof(int));

    memset(host_timestampOut,0,sizeAllocated);
    memset(host_valueOut,0,sizeAllocated);

    IntStream inputStream(host_timestamp,host_value, size,1);
    IntStream outputStream(host_timestampOut,host_valueOut,size);
    UnitStream inputUnitStream(host_unit_timestamp,size,1);
    inputStream.print();
    inputUnitStream.print();
    //device copies
    inputStream.copy_to_device();
    inputUnitStream.copy_to_device();
    outputStream.copy_to_device();

    last(&inputStream, &inputUnitStream, &outputStream, 0);

    outputStream.copy_to_host();
    outputStream.print();

    //frees
    outputStream.free_device();
    inputStream.free_device();
    inputUnitStream.free_device();

    free(host_unit_timestamp);
    free(host_timestampOut);
    free(host_valueOut);
    free(host_value);
    free(host_timestamp);
    printf("--------- experimental last end -----------\n");
}

int main(int argc, char **argv) {

    printf("%s Starting...\n", argv[0]);
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    //might wanna derive MAX_THREADS and so on from here! TODO!
    printf("Using Device %d: %s\n", dev, deviceProp.name);

    test_slift();

    /*
    //create & allocate experimental streams
    //still working for size = 1024*1024*10
    int size = 5;
    experimental_last();
    int sizeAllocated = (size_t)size * sizeof(int);
    int * host_timestamp = (int *) malloc(size * sizeof(int));
    int * host_unit_timestamp = (int *) malloc(size * sizeof(int));
    int * host_value = (int *) malloc(size* sizeof(int));
    //int * host_timestamp2 = (int *) malloc(size * sizeof(int));
    //int * host_value2 = (int *) malloc(size* sizeof(int));



    for (int i = 0; i< size;i++) {
        *(host_timestamp+i) = i;
        *(host_unit_timestamp+i) = i;
        *(host_value+i) = i;
    }

    /**(host_timestamp) = 3;
    *(host_timestamp+1) = 6;
    *(host_timestamp+2) = 8;
    *(host_value) = 1;
    *(host_value+1) = 3;
    *(host_value+2) = 6;*/

   /*
    *(host_unit_timestamp) = 0;
    *(host_unit_timestamp+1) = 2;
    *(host_unit_timestamp+2) = 4;
    *(host_unit_timestamp+3) = 5;
    *(host_unit_timestamp+4) = 9;

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

    IntStream inputStream(host_timestamp,host_value, size);
    IntStream outputStream(host_timestampOut,host_valueOut,size);
    IntStream outputStream2(host_timestampOut2,host_valueOut2,size);
    //IntStream outputStream2(host_timestampOut2,host_valueOut2,size);
    UnitStream inputUnitStream(host_unit_timestamp,size);
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
    //inputStream.print();
    inputUnitStream.copy_to_device();
    outputStream.copy_to_device();
    //outputStream.copy_to_device();
    //inputStream2.copy_to_device();
    outputStream2.copy_to_device();
    //time(&inputStream, &outputStream, stream[0]);
    //inputUnitStream.print();
    //inputStream.print();
    last(&inputStream, &inputUnitStream, &outputStream, stream[0]);
    time(&outputStream,&outputStream2, stream[0]);
    //copy back and output
    //printf("time \n");
    outputStream2.copy_to_host();
    outputStream2.print();

    outputStream.copy_to_host();
    outputStream.print();



    //inputStream2.free_device();
    outputStream.free_device();
    outputStream2.free_device();
    inputStream.free_device();
    inputUnitStream.free_device();

    //free(host_timestampOut2);
    //free(host_valueOut2);
    //free(host_value2);
    //free(host_timestamp2);
    free(host_unit_timestamp);
    free(host_timestampOut);
    free(host_valueOut);
    free(host_value);
    free(host_timestamp);

    for (int i = 0; i < MAX_STREAMS; ++i)
        cudaStreamDestroy(stream[i]);
    //not working with destructor! --> I think we just shouldn't use destructor
    CHECK(cudaProfilerStop());
    CHECK(cudaDeviceReset());

    */
    return(0);
}
