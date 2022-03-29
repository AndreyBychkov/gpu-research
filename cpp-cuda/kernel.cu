
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <time.h>
#include <cmath>

__global__ void matmul_kernel(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            sum += A[row * N + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}


cudaError_t matmul_gpu(const float* lhs, const float* rhs, float* out, unsigned int n);


int main()
{   
    printf("Started\n");
    const int n = 1024*1024*16*16;
    float *a = (float*)malloc(n * sizeof(float));
    float *b = (float*)malloc(n * sizeof(float));
    srand(42);
    #pragma omp parallel for
    for (size_t i = 0; i != n; ++i) {
        a[i] = (rand() % 10) - 10;
        b[i] = (rand() % 10) - 10;
    }
    float *c = (float*)malloc(n * sizeof(float));
    printf("Init finalized\n");
    clock_t begin = clock();
    cudaError_t cudaStatus = matmul_gpu(a, b, c, n);
    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("c[0] = %.1f\n", c[0]);
    printf("time spent = %.6fs\n", time_spent);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
    printf("Press Any Key to Exit\n");
    getchar();
    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t matmul_gpu(const float *lhs, const float *rhs, float *out, unsigned int n)
{
    float *lhs_gpu = 0;
    float *rhs_gpu = 0;
    float *out_gpu = 0;
    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&out_gpu, n * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&lhs_gpu, n * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&rhs_gpu, n * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(lhs_gpu, lhs, n * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(rhs_gpu, rhs, n * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }


    int block_size;
    int grid_size;
    int min_grid_size;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, matmul_kernel, 0, 0);
    grid_size = (n + block_size - 1) / block_size;
    printf("using %d blocks and %d threads per block\n", grid_size, block_size);

    clock_t begin = clock();
    matmul_kernel<<<grid_size, block_size>>>(lhs_gpu, rhs_gpu, out_gpu, static_cast<int>(sqrt(n)));
    cudaDeviceSynchronize();
    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("time spent on kernel = %.6fs\n", time_spent);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    cudaStatus = cudaMemcpy(out, out_gpu, n * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(out_gpu);
    cudaFree(lhs_gpu);
    cudaFree(rhs_gpu);
    
    return cudaStatus;
}
