#include <cstdio>
#include <iostream>
#include <cuda_runtime.h>


void allocateToDevice(float* x, int size) {
    cudaError_t error = cudaMalloc((void**) &x, size);
    if(error != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(error), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
}

void freeFromDevice(float* x) {
    cudaError_t error = cudaFree(x);
    if(error != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(error), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
}

void copyFromHostToDevice(float* dest, float* src, int size) {
    cudaError_t error = cudaMemcpy(dest, src, size, cudaMemcpyHostToDevice);
    if(error != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(error), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
}


void vecAdd(float* A_h, float* B_h, float* C_h, int n) {
    int size = n * sizeof(float);
    float *A_d, *B_d, *C_d;

    // allocate memory on the device
    allocateToDevice(A_d, size);
    allocateToDevice(B_d, size);
    allocateToDevice(C_d, size);

    // copy from host to device

    // dest, src, size, direction
    copyFromHostToDevice(A_d, A_h, size);
    copyFromHostToDevice(B_d, B_h, size);
    copyFromHostToDevice(C_d, C_h, size);


    // free memory
    freeFromDevice(A_d);
    freeFromDevice(B_d);
    freeFromDevice(C_d);
}

int main() {

    
    return 0;
}
