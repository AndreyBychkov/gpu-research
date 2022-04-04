
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <time.h>
#include <cmath>

const int TILE_SZ = 32;

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

__global__ void matmul_tiled_kernel(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float tile_a[TILE_SZ][TILE_SZ];
    __shared__ float tile_b[TILE_SZ][TILE_SZ];

    float sum = 0.0f;

    for(int i = 0; i < (TILE_SZ + N - 1)/TILE_SZ; ++i) {
        // Fill tile A
        if (TILE_SZ*i + threadIdx.x < N && row < N) {
            tile_a[threadIdx.y][threadIdx.x] = A[row*N + i*TILE_SZ + threadIdx.x];
        } else {
            tile_a[threadIdx.y][threadIdx.x] = 0.0f;
        }
        // Fill tile B
        if (TILE_SZ*i + threadIdx.y < N && col < N) {
            tile_b[threadIdx.y][threadIdx.x] = B[N*(i*TILE_SZ + threadIdx.y) + col];
        } else {
            tile_b[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();
        for (int j = 0; j < TILE_SZ; ++j) {
            sum += tile_a[threadIdx.y][j] * tile_b[j][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] -= sum;
    }
}


cudaError_t matmul_gpu(const float* lhs, const float* rhs, float* out, unsigned int n);
void check(const float* out, int n) {
    for (int i=0; i<n; ++i) {
        if (out[i] > 0.0f) {
            printf("Err: out[%d]\t= %.12f\n", i, out[i]);
        }
    }
}


int main()
{   
    printf("Started\n");
    const int n = 1024*1024*16*16;
    printf("Matrix NxN = %d, total %f+ GB\n", n, 3*n*1e-9 * sizeof(float)/sizeof(char));
    float *a = (float*)malloc(n * sizeof(float));
    float *b = (float*)malloc(n * sizeof(float));
    srand(42);
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
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

    return 0;
}

// Helper function for using CUDA
cudaError_t matmul_gpu(const float *lhs, const float *rhs, float *out, unsigned int n)
{
    float *lhs_gpu = 0;
    float *rhs_gpu = 0;
    float *out_gpu = 0;
    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        return cudaStatus;
    }

    cudaStatus = cudaMalloc((void**)&out_gpu, n * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return cudaStatus;
    }

    cudaStatus = cudaMalloc((void**)&lhs_gpu, n * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return cudaStatus;
    }

    cudaStatus = cudaMalloc((void**)&rhs_gpu, n * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return cudaStatus;
    }

    cudaStatus = cudaMemcpy(lhs_gpu, lhs, n * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        return cudaStatus;
    }

    cudaStatus = cudaMemcpy(rhs_gpu, rhs, n * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        return cudaStatus;
    }

    int block_size = 32;
    int grid_size;
    //int min_grid_size;
    // cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, matmul_kernel, 0, 0);
    grid_size = (static_cast<int>(sqrt(n)) + block_size - 1) / block_size;
    printf("using %d blocks and %d threads per block (%d threads total)\n",
        grid_size*grid_size, block_size*block_size, grid_size*grid_size*block_size*block_size);
    dim3 dim_block(block_size,block_size);
    dim3 dim_grid(grid_size,grid_size);

    clock_t begin = clock();
    matmul_kernel<<<dim_grid, dim_block>>>(lhs_gpu, rhs_gpu, out_gpu, static_cast<int>(sqrt(n)));
    cudaDeviceSynchronize();
    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("time spent on naive kernel = %.6fs,\tPerf = %.6f Gflops\n", time_spent, 2.0*sqrt(n)*sqrt(n)*1e-9*sqrt(n)/time_spent);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "matmul_kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        return cudaStatus;
    }
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching matmul_kernel!\n", cudaStatus);
        return cudaStatus;
    }

    block_size = TILE_SZ;
    grid_size = (static_cast<int>(sqrt(n)) + block_size - 1) / block_size;
    printf("using %d blocks and %d threads per block (%d threads total)\n", 
        grid_size*grid_size, block_size*block_size, grid_size*grid_size*block_size*block_size);
    dim_block = dim3(block_size,block_size);
    dim_grid = dim3(grid_size,grid_size);
    begin = clock();
    matmul_tiled_kernel<<<dim_grid, dim_block>>>(lhs_gpu, rhs_gpu, out_gpu, static_cast<int>(sqrt(n)));
    cudaDeviceSynchronize();
    end = clock();
    time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("time spent on tiled kernel = %.6fs,\tPerf = %.6f Gflops\n", time_spent, 2.0*sqrt(n)*sqrt(n)*1e-9*sqrt(n)/time_spent);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "matmul_tiled_kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        return cudaStatus;
    }
    

    cudaStatus = cudaMemcpy(out, out_gpu, n * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        return cudaStatus;
    }

    check(out, n);
    cudaFree(out_gpu);
    cudaFree(lhs_gpu);
    cudaFree(rhs_gpu);

    return cudaStatus;
}
