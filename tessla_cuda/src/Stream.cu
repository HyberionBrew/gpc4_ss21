//
// Created by fabian on 28.05.21.
//
#include <cuda_runtime.h>

#include "Stream.cuh"
#include "helper.cuh"

IntStream::IntStream(int *timestamp,int *value, size_t size) {
    this->host_timestamp = timestamp;
    this->host_values = value;
    this->size = size;
}

//DEVICE ONLY dont use
IntStream::IntStream(bool deviceOnly, size_t size) {
    if (deviceOnly) {
        int sizeAllocate = this->size * sizeof(int);
        this->size = size;
        CHECK(cudaMalloc((int**)&this->device_timestamp, sizeAllocate));
        CHECK(cudaMalloc((int**)&this->device_values, sizeAllocate));
    }else{
        printf("U are using this function wrong, just creates uninitalized stream ONLY on device (i.e. can not be copied back)");
        exit(1);
    }

}


void IntStream::print() {
    printf("IntStream\n");
    printf("t|value\n");
    for (int i = 0; i< this->size;i++) {
        printf("%d|%d \n",this->host_timestamp[i],this->host_values[i]);
    }
    printf("end IntStream\n");
}

void IntStream::free_device(){
    CHECK(cudaFree(this->device_timestamp));
    CHECK(cudaFree(this->device_values));
}

//TODO! implement Staged concurrent copy and execute
//https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#memory-optimizations
// i.e. maybe have a function that doesnt just copy but also performs function?
void IntStream::copy_to_device(){
    int sizeAllocate = this->size * sizeof(int);

    CHECK(cudaMalloc((int**)&this->device_timestamp, sizeAllocate));
    CHECK(cudaMalloc((int**)&this->device_values, sizeAllocate));
    //Aync copying - However excectution of the kernel weaits for it to complete! (since default stream 0 is used!)
    // However CPU continues
    CHECK(cudaMemcpy(this->device_timestamp, this->host_timestamp, sizeAllocate, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(this->device_values, this->host_values, sizeAllocate, cudaMemcpyHostToDevice));
}

void IntStream::copy_to_host() {
    int sizeAllocate = this->size * sizeof(int);
    //dest,src
    printf("%d\n",this->host_values);
    memset(this->host_values, 0, sizeAllocate);
    memset(this->host_timestamp,  0, sizeAllocate);
    CHECK(cudaMemcpy(this->host_values, this->device_values, sizeAllocate, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(this->host_timestamp, this->device_timestamp, sizeAllocate, cudaMemcpyDeviceToHost));
}


void UnitStream::print() {
    printf("UnitStream\n");
    printf("t\n");
    for (int i = 0; i< this->size;i++) {
        printf("%d \n",this->host_timestamp[i]);
    }
    printf("end UnitStream\n");
}


UnitStream::UnitStream(int*timestamp,size_t size) {
    this->host_timestamp = timestamp;
    this->size = size;
}


void UnitStream::free_device(){
    CHECK(cudaFree(this->device_timestamp));
}

void UnitStream::copy_to_device(){
    int sizeAllocate = this->size * sizeof(int);
    CHECK(cudaMalloc((int**)&this->device_timestamp, sizeAllocate));
    printf("th %d \n",this->device_timestamp);
    printf("th %d \n",this->host_timestamp);
    CHECK(cudaMemcpy(this->device_timestamp, this->host_timestamp, sizeAllocate, cudaMemcpyHostToDevice));

}

void UnitStream::copy_to_host() {
    int sizeAllocate = this->size * sizeof(int);
    //dest,src
    memset(this->host_timestamp,  0, sizeAllocate);
    CHECK(cudaMemcpy(this->host_timestamp, this->device_timestamp, sizeAllocate, cudaMemcpyDeviceToHost));
}