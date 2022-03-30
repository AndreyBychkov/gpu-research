//Add if include libraries locally
//#pragma comment(lib, "cublas.lib")
//#pragma comment(lib, "curand.lib")

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <time.h>
#include <curand.h>

#define M 1024
#define N 1014
#define IDX2F(i,j,ld) ((((j)-1)*(ld))+((i)-1))

static __inline__ void modify(cublasHandle_t handle, float* m, int ldm, int n, int p, int q, float alpha, float beta) {
    cublasSscal(handle, n - q + 1, &alpha, &m[IDX2F(p, q, ldm)], ldm);
    cublasSscal(handle, ldm - p + 1, &beta, &m[IDX2F(p, q, ldm)], 1);
}


void GPU_fill_rand(float* A, int nr_rows_A, int nr_cols_A) {
    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());
    curandGenerateUniform(prng, A, nr_rows_A * nr_cols_A);
}

void gpu_blas_mmul(const float* A, const float* B, float* C, const int m, const int k, const int n) {
    int lda = m, ldb = k, ldc = m;
    const float alf = 1;
    const float bet = 0;
    const float* alpha = &alf;
    const float* beta = &bet;
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    cublasDestroy(handle);
}

int main(void) {
    int nr_rows_A, nr_cols_A, nr_rows_B, nr_cols_B, nr_rows_C, nr_cols_C;
    nr_rows_A = nr_cols_A = nr_rows_B = nr_cols_B = nr_rows_C = nr_cols_C = N;
    float* h_A = (float*)malloc(nr_rows_A * nr_cols_A * sizeof(float));
    float* h_B = (float*)malloc(nr_rows_B * nr_cols_B * sizeof(float));
    float* h_C = (float*)malloc(nr_rows_C * nr_cols_C * sizeof(float));

    float* d_A, * d_B, * d_C;
    cudaMalloc(&d_A, nr_rows_A * nr_cols_A * sizeof(float));
    cudaMalloc(&d_B, nr_rows_B * nr_cols_B * sizeof(float));
    cudaMalloc(&d_C, nr_rows_B * nr_cols_B * sizeof(float));

    GPU_fill_rand(d_A, nr_rows_A, nr_cols_A);
    GPU_fill_rand(d_B, nr_rows_B, nr_cols_B);

    clock_t start = clock();
    gpu_blas_mmul(d_A, d_B, d_C, nr_rows_A, nr_cols_A, nr_cols_B);
    clock_t end = clock();
    cudaMemcpy(h_C, d_C, nr_rows_C * nr_cols_C * sizeof(float), cudaMemcpyDeviceToHost);


    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    
    printf("Kernel execution time: %.9f\n", (end - start) * 1.0 / CLOCKS_PER_SEC);
    printf("Press Any Key to Exit\n");
    getchar();
    return EXIT_SUCCESS;
}
