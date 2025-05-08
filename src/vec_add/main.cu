#include <cstdio>
#include <iostream>
#include <cuda_runtime.h>


void allocateToDevice(float** x, int size) {
    cudaError_t error = cudaMalloc((void**) x, size);
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

void copyFromDeviceToHost(float* dest, float* src, int size) {
    cudaError_t error = cudaMemcpy(dest, src, size, cudaMemcpyDeviceToHost);
    if(error != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(error), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
}

__global__ void addKernel(float* A, float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

void vecAdd(float* A_h, float* B_h, float* C_h, int n) {
    int size = n * sizeof(float);
    float *A_d, *B_d, *C_d;
    
    // allocate memory on the device
    allocateToDevice(&A_d, size);
    allocateToDevice(&B_d, size);
    allocateToDevice(&C_d, size);

    // copy from host to device
    copyFromHostToDevice(A_d, A_h, size);
    copyFromHostToDevice(B_d, B_h, size);
    
    // launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    addKernel<<<blocksPerGrid, threadsPerBlock>>>(A_d, B_d, C_d, n);
    
    // check for errors
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
    
    // copy result from device to host
    copyFromDeviceToHost(C_h, C_d, size);

    // free memory
    freeFromDevice(A_d);
    freeFromDevice(B_d);
    freeFromDevice(C_d);
}

int main() {
    // create host data
    int n = 100;
    int size = n * sizeof(float);
    float *A_h = (float*)malloc(size);
    float *B_h = (float*)malloc(size);
    float *C_h = (float*)malloc(size);

    for(int i = 0; i < n; i++) {
        A_h[i] = i;
        B_h[i] = i;
    }

    // call vecAdd
    vecAdd(A_h, B_h, C_h, n);

    // verify result
    printf("Vector addition results:\n");
    for(int i = 0; i < 10; i++) {  // Print first 10 elements
        printf("%f + %f = %f\n", A_h[i], B_h[i], C_h[i]);
    }

    // free host data
    free(A_h);
    free(B_h);
    free(C_h);
    
    return 0;
}
