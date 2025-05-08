#include "utils.h"

void allocateToDevice(float **x, int size)
{
    cudaError_t result = cudaMalloc((void **)x, size);
    CUDA_CHECK(result);
}

void freeFromDevice(float *x)
{
    cudaError_t result = cudaFree(x);
    CUDA_CHECK(result);
}

void copyFromHostToDevice(float *dest, float *src, int size)
{
    cudaError_t result = cudaMemcpy(dest, src, size, cudaMemcpyHostToDevice);
    CUDA_CHECK(result);
}

void copyFromDeviceToHost(float *dest, float *src, int size)
{
    cudaError_t result = cudaMemcpy(dest, src, size, cudaMemcpyDeviceToHost);
    CUDA_CHECK(result);
}
