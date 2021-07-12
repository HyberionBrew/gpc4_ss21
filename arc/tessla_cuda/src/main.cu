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

void test_count(){
    printf("count() test\n");

    int *input_1 = (int*)malloc(5*sizeof(int));
    int *input_2 = (int*)malloc(5*sizeof(int));

    input_1[0] = 0;
    input_1[1] = 1;
    input_1[2] = 2;
    input_1[3] = 3;
    input_1[4] = 4;

    input_2[0] = 1;
    input_2[1] = 2;
    input_2[2] = 3;
    input_2[3] = 4;
    input_2[4] = 5;

    std::shared_ptr<GPUUnitStream> inp_1(new GPUUnitStream(input_1, 5));
    std::shared_ptr<GPUUnitStream> inp_2(new GPUUnitStream(input_2, 5));

    printf("made inputs\n");

    inp_1->copy_to_device();

    std::shared_ptr<GPUIntStream> res_1 = count(inp_1);
    std::shared_ptr<GPUIntStream> res_2 = count(inp_2);
    res_1->host_offset = (int*)malloc(sizeof(int));
    res_1->host_timestamp = (int*)malloc(6*sizeof(int));
    res_1->host_values = (int*)malloc(6*sizeof(int));
    res_1->size = 6;

    res_2->host_offset = (int*)malloc(sizeof(int));
    res_2->host_timestamp = (int*)malloc(6*sizeof(int));
    res_2->host_values = (int*)malloc(6*sizeof(int));
    res_2->size = 6;

    res_1->copy_to_host();
    res_2->copy_to_host();

    printf("RESULT 1:\n");
    res_1->print();

    printf("RESULT 2:\n");
    res_2->print();
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

    //test_slift();
    test_count();

    return(0);
}
