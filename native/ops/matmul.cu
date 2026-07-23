#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace jstorch {
namespace ops {

static cublasHandle_t get_handle() {
    static cublasHandle_t h = nullptr;
    if (!h) cublasCreate(&h);
    return h;
}

extern "C" {

// 2D: [M,K] x [K,N] -> [M,N]
void launch_matmul(const float* a, const float* b, float* c,
    int M, int K, int N, cudaStream_t s) {
    cublasHandle_t h = get_handle();
    cublasSetStream(h, s);
    float alpha = 1.0f, beta = 0.0f;
    // cuBLAS is column-major, so we compute C^T = B^T * A^T
    // which gives row-major C = A * B
    cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
        &alpha, b, N, a, K, &beta, c, N);
}

// Batch matmul: [B,M,K] x [B,K,N] -> [B,M,N]
void launch_bmm(const float* a, const float* b, float* c,
    int B, int M, int K, int N, cudaStream_t s) {
    cublasHandle_t h = get_handle();
    cublasSetStream(h, s);
    float alpha = 1.0f, beta = 0.0f;
    long long strideA = (long long)M * K;
    long long strideB = (long long)K * N;
    long long strideC = (long long)M * N;
    cublasSgemmStridedBatched(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
        &alpha, b, N, strideB, a, K, strideA, &beta, c, N, strideC, B);
}

}

}
}
