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

IntStream::IntStream(size_t size) {
    this->size = size;
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

UnitStream::UnitStream(int*timestamp,size_t size, bool OnDevice) {
    this->device_timestamp = timestamp;
    this->size = size;
    this->OnDevice = false;
}

void IntStream::copy_to_device(){
    int sizeAllocate = this->size * sizeof(int);
    //this->device_timestamp = (int*)malloc(sizeAllocate);
    //this->device_values = (int*)malloc(sizeAllocate);
    //memset(this->device_timestamp,0,sizeAllocate);
    //memset(this->device_values,0,sizeAllocate);
    CHECK(cudaMalloc((int**)&this->device_timestamp, sizeAllocate));
    CHECK(cudaMalloc((int**)&this->device_values, sizeAllocate));
    CHECK(cudaMemcpy(this->device_timestamp, this->host_timestamp, sizeAllocate, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(this->device_values, this->host_values, sizeAllocate, cudaMemcpyHostToDevice));

}

void IntStream::copy_to_host() {
    int sizeAllocate = this->size * sizeof(int);
    //dest,src
    memset(this->host_values, 0, sizeAllocate);
    memset(this->host_timestamp,  0, sizeAllocate);
    CHECK(cudaMemcpy(this->host_values, this->device_values, sizeAllocate, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(this->host_timestamp, this->device_timestamp, sizeAllocate, cudaMemcpyDeviceToHost));

}