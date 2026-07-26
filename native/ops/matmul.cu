#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace jstorch {
namespace ops {

static cublasHandle_t get_handle() {
    static cublasHandle_t h = nullptr;
    if (!h) {
        cublasCreate(&h);
        // Enable TF32 Tensor Core GEMM on Ampere+ (transparent for FP32 interface)
        cublasSetMathMode(h, CUBLAS_TF32_TENSOR_OP_MATH);
    }
    return h;
}

// ── Bias-add kernel: out[b, j] += bias[j] ──────────────────────────────────
__global__ void add_bias_kernel(float* out, const float* bias, int n) {
    // Each thread handles one element; blockIdx.y = batch index
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;
    if (j < n) out[b * n + j] += bias[j];
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

// linear: out [B, out_f] = x [B, in_f] @ w.T  +  bias [out_f]
// w is stored [out_f, in_f] (row-major). One N-API call = GEMM + bias in stream.
void launch_linear(const float* x, const float* w, const float* bias,
    float* out, int B, int in_f, int out_f, cudaStream_t s) {
    cublasHandle_t h = get_handle();
    cublasSetStream(h, s);
    float one = 1.0f, zero = 0.0f;
    // Row-major trick: C[B,out_f] = x[B,in_f] @ w.T[in_f,out_f]
    // cuBLAS col-major: C^T[out_f,B] = w[out_f,in_f]^T * x[B,in_f]^T
    cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, out_f, B, in_f,
                &one, w, in_f, x, in_f, &zero, out, out_f);
    if (bias) {
        dim3 grid((out_f + 255) / 256, B);
        add_bias_kernel<<<grid, 256, 0, s>>>(out, bias, out_f);
    }
}

}

}
}
