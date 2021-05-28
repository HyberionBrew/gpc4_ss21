//
// Created by fabian on 28.05.21.
//
#include <cuda_runtime.h>

#include "Stream.cuh"
#include "helper.cuh"

IntStream::IntStream(int *timestamp,int *value, size_t size) {
    this->timestamp_host = timestamp;
    this->value_host = value;
    this->size = size;
}

IntStream::IntStream(size_t size) {
    this->size = size;
}

IntStream::~IntStream(){
    CHECK(cudaFree(this->timestamp_device));
    CHECK(cudaFree(this->value_device));
}

UnitStream::UnitStream(int*timestamp,size_t size, bool OnDevice) {
    this->timestamp_device = timestamp;
    this->size = size;
    this->OnDevice = false;
}

void IntStream::copy_to_device(){
    int sizeAllocate = this->size * sizeof(int);
    //this->timestamp_device = (int*)malloc(sizeAllocate);
    //this->value_device = (int*)malloc(sizeAllocate);
    //memset(this->timestamp_device,0,sizeAllocate);
    //memset(this->value_device,0,sizeAllocate);
    CHECK(cudaMalloc((int**)&this->timestamp_device, sizeAllocate));
    printf("ptr %d \n",timestamp_device);
    CHECK(cudaMalloc((int**)&this->value_device, sizeAllocate));
    CHECK(cudaMemcpy(this->timestamp_device, this->timestamp_host, sizeAllocate, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(this->value_device, this->value_host, sizeAllocate, cudaMemcpyHostToDevice));

}

void IntStream::copy_to_host() {
    int sizeAllocate = this->size * sizeof(int);
    //dest,src
    memset(this->value_host, 0, sizeAllocate);
    memset(this->timestamp_host,  0, sizeAllocate);
    CHECK(cudaMemcpy(this->value_host, this->value_device, sizeAllocate, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(this->timestamp_host, this->timestamp_device, sizeAllocate, cudaMemcpyDeviceToHost));

}