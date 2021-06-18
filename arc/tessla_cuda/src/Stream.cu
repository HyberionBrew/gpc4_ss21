//
// Created by fabian on 28.05.21.
//
#include <cuda_runtime.h>

#include "Stream.cuh"
#include "helper.cuh"

IntStream::IntStream(int *timestamp,int *value, size_t size, int offs) {
    this->host_timestamp = timestamp;
    this->host_values = value;
    this->size = size;
    //TODO! this is per object tf. problems
    this->host_offset = (int *) malloc(sizeof(int));
    *this->host_offset = offs;
    onDevice =false;

}
IntStream::IntStream(int *timestamp,int *value, size_t size) {
    this->host_timestamp = timestamp;
    this->host_values = value;
    this->size = size;
    //TODO! this is per object tf. problems
    this->host_offset = (int *) malloc(sizeof(int));
    memset( this->host_offset,0,sizeof(int));
    onDevice =false;

}
//DEVICE ONLY dont use
// THIS CAN BE DELETED!
IntStream::IntStream() {
    this->host_offset = (int *) malloc(sizeof(int));
    memset( this->host_offset,0,sizeof(int));
    onDevice =false;
}
UnitStream::UnitStream() {
    this->host_offset = (int *) malloc(sizeof(int));
    memset( this->host_offset,0,sizeof(int));
    onDevice =false;
}

void IntStream::print() {
    printf("IntStream\n");
    printf("t|value\n");
    //
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

void IntStream::free_host(){
    free(this->host_timestamp);
    free(this->host_values);
}

void UnitStream::free_host(){
    free(this->host_timestamp);
}

//TODO! implement Staged concurrent copy and execute
//https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#memory-optimizations
// i.e. maybe have a function that doesnt just copy but also performs function?
void IntStream::copy_to_device(bool valid){
    onDevice =true;
    int sizeAllocate = this->size * sizeof(int);

    CHECK(cudaMalloc((int**)&this->device_timestamp, sizeAllocate));
    CHECK(cudaMalloc((int**)&this->device_values, sizeAllocate));
    CHECK(cudaMalloc((int**)&this->device_offset, sizeof(int)));
    // Async copying - However excectution of the kernel waits for it to complete! (since default stream 0 is used!)
    // However CPU continues
    CHECK(cudaMemcpy(this->device_offset, this->host_offset, sizeof(int), cudaMemcpyHostToDevice));
    if (valid == true) {
        CHECK(cudaMemcpy(this->device_timestamp, this->host_timestamp, sizeAllocate, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(this->device_values, this->host_values, sizeAllocate, cudaMemcpyHostToDevice));
    }
}

void IntStream::copy_to_device(){
    onDevice =true;
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
}


void UnitStream::print() {
    printf("UnitStream\n");
    printf("t\n");
    for (int i = *this->host_offset; i< this->size;i++) {
        printf("%d \n",this->host_timestamp[i]);
    }
    printf("end UnitStream\n");
}


UnitStream::UnitStream(int*timestamp,size_t size, int offs) {
    this->host_timestamp = timestamp;
    this->size = size;
    this->host_offset = (int *) malloc(size* sizeof(int));
    *this->host_offset = offs;
    onDevice =false;
}

UnitStream::UnitStream(int*timestamp,size_t size) {
    this->host_timestamp = timestamp;
    this->size = size;
    this->host_offset = (int *) malloc(size* sizeof(int));
    memset( this->host_offset,0,sizeof(int));
    onDevice =false;
}


void UnitStream::free_device(){
    CHECK(cudaFree(this->device_timestamp));
    CHECK(cudaFree(this->device_offset));
   // free(this->host_timestamp);
    free(this->host_offset);
}

void UnitStream::copy_to_device(){
    onDevice =true;
    int sizeAllocate = this->size * sizeof(int);
    CHECK(cudaMalloc((int**)&this->device_timestamp, sizeAllocate));
    CHECK(cudaMalloc((int**)&this->device_offset, sizeof(int)));
    CHECK(cudaMemcpy(this->device_timestamp, this->host_timestamp, sizeAllocate, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(this->device_offset, this->host_offset, sizeof(int), cudaMemcpyHostToDevice));

}

void UnitStream::copy_to_device(bool valid){
    onDevice =true;
    int sizeAllocate = this->size * sizeof(int);
    CHECK(cudaMalloc((int**)&this->device_timestamp, sizeAllocate));
    CHECK(cudaMalloc((int**)&this->device_offset, sizeof(int)));
    if (valid == true) {
        CHECK(cudaMemcpy(this->device_timestamp, this->host_timestamp, sizeAllocate, cudaMemcpyHostToDevice));
    }
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