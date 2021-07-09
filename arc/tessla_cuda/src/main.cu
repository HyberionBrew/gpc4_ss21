// The boilerplatte code is taken from tuwel

#include <cuda_runtime.h>
#include <sys/time.h>
#include <cuda_profiler_api.h>
#include "main.cuh"
#include "helper.cuh"
#include "GPUStream.cuh"
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

    std::shared_ptr<GPUIntStream> x(new GPUIntStream(x_ts, x_v, sx));
    std::shared_ptr<GPUIntStream> y(new GPUIntStream(y_ts, y_v, sy));

    x->copy_to_device();
    y->copy_to_device();

    std::shared_ptr<GPUIntStream> res = slift(x,y, ADD);
    res->copy_to_host();

    x->print();
    y->print();
    res->print();
}

int main(int argc, char **argv) {

    printf("%s Starting...\n", argv[0]);
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);

    test_slift();

    return(0);
}
