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
    //TODO! this is per object tf. problems
    this->host_offset = (int *) malloc(size* sizeof(int));
    memset( this->host_offset,0,sizeof(int));

}

//DEVICE ONLY dont use
IntStream::IntStream(bool deviceOnly, size_t size) {
    if (deviceOnly) {
        int sizeAllocate = this->size * sizeof(int);
        this->size = size;
        CHECK(cudaMalloc((int**)&this->device_timestamp, sizeAllocate));
        CHECK(cudaMalloc((int**)&this->device_values, sizeAllocate));
    }
    else{
        printf("U are using this function wrong, just creates uninitalized stream ONLY on device (i.e. can not be copied back)");
        exit(1);
    }
}

void IntStream::print() {
    printf("IntStream\n");
    printf("t|value\n");
    for (int i = *this->host_offset; i< this->size;i++) {
        printf("%d|%d \n",this->host_timestamp[i],this->host_values[i]);
    }
    printf("end IntStream\n");
}

void IntStream::free_device(){
    CHECK(cudaFree(this->device_timestamp));
    CHECK(cudaFree(this->device_values));
    CHECK(cudaFree(this->device_offset));
    free(this->host_offset);
}

//TODO! implement Staged concurrent copy and execute
//https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#memory-optimizations
// i.e. maybe have a function that doesnt just copy but also performs function?
void IntStream::copy_to_device(){
    int sizeAllocate = this->size * sizeof(int);

    CHECK(cudaMalloc((int**)&this->device_timestamp, sizeAllocate));
    CHECK(cudaMalloc((int**)&this->device_values, sizeAllocate));
    CHECK(cudaMalloc((int**)&this->device_offset, sizeof(int)));
    // Async copying - However excectution of the kernel waits for it to complete! (since default stream 0 is used!)
    // However CPU continues
    CHECK(cudaMemcpy(this->device_offset, this->host_offset, sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(this->device_timestamp, this->host_timestamp, sizeAllocate, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(this->device_values, this->host_values, sizeAllocate, cudaMemcpyHostToDevice));

}

void IntStream::copy_to_host() {
    int sizeAllocate = this->size * sizeof(int);
    //dest,src
    memset(this->host_values, 0, sizeAllocate);
    memset(this->host_timestamp,  0, sizeAllocate);
    memset(this->host_offset,  0, sizeof(int));
    CHECK(cudaMemcpy(this->host_values, this->device_values, sizeAllocate, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(this->host_offset, this->device_offset, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(this->host_timestamp, this->device_timestamp, sizeAllocate, cudaMemcpyDeviceToHost));
    printf("host offset;: %d\n",*this->host_offset);
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
    this->host_offset = (int *) malloc(size* sizeof(int));
    memset( this->host_offset,0,sizeof(int));
}


void UnitStream::free_device(){
    CHECK(cudaFree(this->device_timestamp));
    CHECK(cudaFree(this->device_offset));
    free(this->host_offset);
}

void UnitStream::copy_to_device(){
    int sizeAllocate = this->size * sizeof(int);
    CHECK(cudaMalloc((int**)&this->device_timestamp, sizeAllocate));
    CHECK(cudaMalloc((int**)&this->device_offset, sizeof(int)));
    CHECK(cudaMemcpy(this->device_timestamp, this->host_timestamp, sizeAllocate, cudaMemcpyHostToDevice));

    CHECK(cudaMemcpy(this->device_offset, this->host_offset, sizeof(int), cudaMemcpyHostToDevice));

}

void UnitStream::copy_to_host() {
    int sizeAllocate = this->size * sizeof(int);
    //dest,src
    memset(this->host_timestamp,  0, sizeAllocate);
    memset(this->host_offset,  0, sizeof(int));
    CHECK(cudaMemcpy(this->host_timestamp, this->device_timestamp, sizeAllocate, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(this->host_offset, this->device_offset, sizeof(int), cudaMemcpyDeviceToHost));
}