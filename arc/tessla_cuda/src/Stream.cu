//
// Created by fabian on 28.05.21.
//
#include <cuda_runtime.h>

#include "Stream.cuh"
#include "helper.cuh"

#include <stdexcept>

GPUIntStream::GPUIntStream(int *timestamp, int *value, size_t size, int offs) {
    this->host_timestamp = timestamp;
    this->host_values = value;
    this->size = size;
    //TODO! this is per object tf. problems
    this->host_offset = (int *) malloc(sizeof(int));
    *this->host_offset = offs;
    onDevice =false;

}
GPUIntStream::GPUIntStream(int *timestamp, int *value, size_t size) {
    this->host_timestamp = timestamp;
    this->host_values = value;
    this->size = size;
    //TODO! this is per object tf. problems
    this->host_offset = (int *) malloc(sizeof(int));
    memset( this->host_offset,0,sizeof(int));
    onDevice =false;

}

/**
 * Create an empty stream of size size and copy over to CUDA device if necessary.
 * Does not set memory on host to all 0. Might contain bogus values.
 * @param size The size of the stream to create
 * @param createOnDevice Allocate the memory on CUDA device
 */
GPUIntStream::GPUIntStream(size_t size, bool createOnDevice) {
    int sizeAllocated = size * sizeof(int);
    this->size = size;
    this->host_timestamp = (int *) malloc(size * sizeof(int));
    this->host_values = (int *) malloc(size * sizeof(int));
    this->host_offset = (int *) malloc(sizeof(int));
    memset( this->host_offset,0,sizeof(int));
    // Check if we have enough memory left
    if (this->host_values == nullptr || this->host_timestamp == nullptr) {
        throw std::runtime_error("Out of memory.");
    }
    if (createOnDevice) this->copy_to_device(false);

}

/**
 * Copy constructor of Int Stream. If onDevice is set, the stream is also copied on the CUDA device
 * @param stream Stream to be copied.
 * @param onDevice Also copy data from CUDA device
 */
GPUIntStream::GPUIntStream(GPUIntStream &stream, bool onDevice) : GPUIntStream(stream.size, onDevice) {
    size_t allocSize = stream.size * sizeof (int);
    // Copy host data
    memcpy(this->host_offset,stream.host_offset,sizeof(int));
    memcpy(this->host_timestamp, stream.host_timestamp, allocSize);
    memcpy(this->host_values, stream.host_values, allocSize);
    // Copy device data
    if (onDevice) {
        this->device_offset = stream.device_offset;
        // TODO make this more high performance
        CHECK(cudaMemcpy(this->device_offset, stream.device_offset, allocSize, cudaMemcpyDeviceToDevice));
        CHECK(cudaMemcpy(this->device_timestamp, stream.device_timestamp, allocSize, cudaMemcpyDeviceToDevice));
        CHECK(cudaMemcpy(this->device_values, stream.device_values, allocSize, cudaMemcpyDeviceToDevice));
    }
}

//DEVICE ONLY dont use
// THIS CAN BE DELETED!
GPUIntStream::GPUIntStream() {
    this->host_offset = (int *) malloc(sizeof(int));
    memset( this->host_offset,0,sizeof(int));
    onDevice =false;
}
GPUUnitStream::GPUUnitStream() {
    this->host_offset = (int *) malloc(sizeof(int));
    memset( this->host_offset,0,sizeof(int));
    onDevice =false;
}

void GPUIntStream::print() {
    printf("GPUIntStream\n");
    printf("t|value\n");
    //
    for (int i = *this->host_offset; i< this->size;i++) {
        printf("%d|%d \n",this->host_timestamp[i],this->host_values[i]);
    }
    printf("end GPUIntStream\n");
}

void GPUIntStream::free_device(){
    CHECK(cudaFree(this->device_timestamp));
    CHECK(cudaFree(this->device_values));
    CHECK(cudaFree(this->device_offset));
    free(this->host_offset);
}

void GPUIntStream::free_host(){
    free(this->host_timestamp);
    free(this->host_values);
}

void GPUUnitStream::free_host(){
    free(this->host_timestamp);
}

//TODO! implement Staged concurrent copy and execute
//https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#memory-optimizations
// i.e. maybe have a function that doesnt just copy but also performs function?
void GPUIntStream::copy_to_device(bool valid){
    onDevice =true;
    int sizeAllocate = this->size * sizeof(int);
    CHECK(cudaMalloc((int**)&this->device_timestamp, sizeAllocate));
    CHECK(cudaMalloc((int**)&this->device_values, sizeAllocate));
    CHECK(cudaMalloc((int**)&this->device_offset, sizeof(int)));
    // Async copying - However excectution of the kernel waits for it to complete! (since default stream 0 is used!)
    // However CPU continues
    CHECK(cudaMemcpy(this->device_offset, this->host_offset, sizeof(int), cudaMemcpyHostToDevice));
    if (valid) {
        CHECK(cudaMemcpy(this->device_timestamp, this->host_timestamp, sizeAllocate, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(this->device_values, this->host_values, sizeAllocate, cudaMemcpyHostToDevice));
    }
}

void GPUIntStream::copy_to_device(){
    copy_to_device(true);
}


void GPUIntStream::copy_to_host() {

    int sizeAllocate = size * sizeof(int);
    //dest,src
    memset(this->host_values, 0, sizeAllocate);
    memset(this->host_timestamp,  0, sizeAllocate);
    memset(this->host_offset,  0, sizeof(int));
    CHECK(cudaMemcpy(this->host_values, this->device_values, sizeAllocate, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(this->host_offset, this->device_offset, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(this->host_timestamp, this->device_timestamp, sizeAllocate, cudaMemcpyDeviceToHost));
}


void GPUUnitStream::print() {
    printf("GPUUnitStream\n");
    printf("t\n");
    for (int i = *this->host_offset; i< this->size;i++) {
        printf("%d \n",this->host_timestamp[i]);
    }
    printf("end GPUUnitStream\n");
}


GPUUnitStream::GPUUnitStream(int*timestamp, size_t size, int offs) {
    this->host_timestamp = timestamp;
    this->size = size;
    this->host_offset = (int *) malloc(size* sizeof(int));
    *this->host_offset = offs;
    onDevice =false;
}

GPUUnitStream::GPUUnitStream(int*timestamp, size_t size) {
    this->host_timestamp = timestamp;
    this->size = size;
    this->host_offset = (int *) malloc(size* sizeof(int));
    memset( this->host_offset,0,sizeof(int));
    onDevice =false;
}

/**
 * Create an empty stream of size size and copy over to CUDA device if necessary.
 * Does not set memory on host to all 0. Might contain bogus values.
 * @param size size The size of the stream to create
 * @param createOnDevice Allocate the memory on CUDA device
 */
GPUUnitStream::GPUUnitStream(size_t size, bool createOnDevice) {
    int sizeAllocated = size * sizeof(int);
    this->size = size;
    this->host_timestamp = (int *) malloc(size * sizeof(int));
    this->host_offset = (int *) malloc(sizeof(int));
    // Check if we have enough memory left
    if (this->host_timestamp == nullptr) {
        throw std::runtime_error("Out of memory.");
    }

    if (createOnDevice) this->copy_to_device(false);
}

/**
 * Copy constructor of Unit Stream. If onDevice is set, the stream is also copied on the CUDA device
 * @param stream Stream to be copied.
 * @param onDevice Also copy data from CUDA device
 */
GPUUnitStream::GPUUnitStream(GPUUnitStream &stream, bool onDevice) : GPUUnitStream(stream.size, onDevice){
    this->host_offset = stream.host_offset;
    memcpy(this->host_timestamp, stream.host_timestamp, stream.size * sizeof (int));

    if (onDevice) {
        this->device_offset = stream.device_offset;
        // TODO make this more high performance
        CHECK(cudaMemcpy(this->device_timestamp, stream.device_timestamp, stream.size * sizeof (int), cudaMemcpyDeviceToDevice));
    }
}


void GPUUnitStream::free_device(){
    CHECK(cudaFree(this->device_timestamp));
    CHECK(cudaFree(this->device_offset));
   // free(this->host_timestamp);
    free(this->host_offset);
}

void GPUUnitStream::copy_to_device(){
    onDevice =true;
    int sizeAllocate = this->size * sizeof(int);
    CHECK(cudaMalloc((int**)&this->device_timestamp, sizeAllocate));
    CHECK(cudaMalloc((int**)&this->device_offset, sizeof(int)));
    CHECK(cudaMemcpy(this->device_timestamp, this->host_timestamp, sizeAllocate, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(this->device_offset, this->host_offset, sizeof(int), cudaMemcpyHostToDevice));

}

void GPUUnitStream::copy_to_device(bool valid){
    onDevice =true;
    int sizeAllocate = this->size * sizeof(int);
    CHECK(cudaMalloc((int**)&this->device_timestamp, sizeAllocate));
    CHECK(cudaMalloc((int**)&this->device_offset, sizeof(int)));
    if (valid == true) {
        CHECK(cudaMemcpy(this->device_timestamp, this->host_timestamp, sizeAllocate, cudaMemcpyHostToDevice));
    }
    CHECK(cudaMemcpy(this->device_offset, this->host_offset, sizeof(int), cudaMemcpyHostToDevice));
}


void GPUUnitStream::copy_to_host() {
    int sizeAllocate = this->size * sizeof(int);
    //dest,src
    memset(this->host_timestamp,  0, sizeAllocate);
    memset(this->host_offset,  0, sizeof(int));
    CHECK(cudaMemcpy(this->host_timestamp, this->device_timestamp, sizeAllocate, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(this->host_offset, this->device_offset, sizeof(int), cudaMemcpyDeviceToHost));
}