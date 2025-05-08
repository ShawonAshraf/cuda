#include <cstdio>
#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>

// Check CUDA errors as a macro
#define CUDA_CHECK(x)                                                                    \
    do                                                                                   \
    {                                                                                    \
        cudaError_t e = x;                                                               \
        if (e != cudaSuccess)                                                            \
        {                                                                                \
            printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
            exit(EXIT_FAILURE);                                                          \
        }                                                                                \
    } while (0)

void allocateToDevice(float **x, int size);

void freeFromDevice(float *x);

void copyFromHostToDevice(float *dest, float *src, int size);

void copyFromDeviceToHost(float *dest, float *src, int size);
