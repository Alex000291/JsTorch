// MatMul.cu - Matrix multiplication using cuBLAS
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>

// Global cuBLAS handle (initialized once)
static cublasHandle_t cublas_handle = nullptr;

// Initialize cuBLAS handle (call once at startup)
extern "C" void init_cublas() {
    if (cublas_handle == nullptr) {
        cublasCreate(&cublas_handle);
        cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH);
    }
}

// Cleanup cuBLAS handle
extern "C" void cleanup_cublas() {
    if (cublas_handle != nullptr) {
        cublasDestroy(cublas_handle);
        cublas_handle = nullptr;
    }
}

// ==================== Matrix Multiplication using cuBLAS ====================

/**
 * Compute C = A * B using cuBLAS
 * 
 * A: (M x K)
 * B: (K x N)
 * C: (M x N)
 * 
 * cuBLAS uses column-major order, but we use row-major
 * So we compute: C^T = B^T * A^T
 * Which gives us: C = A * B in row-major
 */
extern "C" void launch_matmul(const float* A, const float* B, float* C,
                             int M, int N, int K, cudaStream_t stream) {
    // Initialize cuBLAS if not already done
    if (cublas_handle == nullptr) {
        init_cublas();
    }
    
    // Set stream for cuBLAS
    cublasSetStream(cublas_handle, stream);
    
    // cuBLAS parameters
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // Perform: C = alpha * A * B + beta * C
    // Since cuBLAS is column-major and we're row-major:
    // C^T = B^T * A^T
    // So we swap A and B, and swap M and N
    cublasSgemm(
        cublas_handle,
        CUBLAS_OP_N,    // B^T is not transposed (in column-major view)
        CUBLAS_OP_N,    // A^T is not transposed (in column-major view)
        N,              // Number of rows of B^T (columns of B in row-major)
        M,              // Number of columns of A^T (rows of A in row-major)
        K,              // Inner dimension
        &alpha,
        B,              // B matrix
        N,              // Leading dimension of B
        A,              // A matrix
        K,              // Leading dimension of A
        &beta,
        C,              // Output matrix
        N               // Leading dimension of C
    );
}

// ==================== Batched Matrix Multiplication ====================

/**
 * Batched matmul: C[i] = A[i] * B[i] for i in [0, batch_size)
 * Each matrix in batch has same dimensions
 */
extern "C" void launch_matmul_batched(const float* A, const float* B, float* C,
                                     int M, int N, int K, int batch_size,
                                     cudaStream_t stream) {
    if (cublas_handle == nullptr) {
        init_cublas();
    }
    
    cublasSetStream(cublas_handle, stream);
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // Stride between matrices in batch
    long long int strideA = M * K;
    long long int strideB = K * N;
    long long int strideC = M * N;
    
    // Strided batched matmul
    cublasSgemmStridedBatched(
        cublas_handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        N,
        M,
        K,
        &alpha,
        B, N, strideB,
        A, K, strideA,
        &beta,
        C, N, strideC,
        batch_size
    );
}

// ==================== Matrix-Vector Multiplication ====================

/**
 * Compute y = A * x
 * A: (M x N)
 * x: (N,)
 * y: (M,)
 */
extern "C" void launch_matvec(const float* A, const float* x, float* y,
                             int M, int N, cudaStream_t stream) {
    if (cublas_handle == nullptr) {
        init_cublas();
    }
    
    cublasSetStream(cublas_handle, stream);
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // y = alpha * A * x + beta * y
    cublasSgemv(
        cublas_handle,
        CUBLAS_OP_T,    // Transpose because cuBLAS is column-major
        N,              // Rows of A^T
        M,              // Columns of A^T
        &alpha,
        A,
        N,              // Leading dimension
        x,
        1,              // Stride of x
        &beta,
        y,
        1               // Stride of y
    );
}

// ==================== Tensor Contraction (einsum-like) ====================

/**
 * For more complex operations like einsum, we can combine multiple cuBLAS calls
 * This is a placeholder for future implementation
 */
extern "C" void launch_einsum(const float* A, const float* B, float* C,
                             const char* equation, cudaStream_t stream) {
    // TODO: Parse einsum equation and dispatch to appropriate cuBLAS routines
    // For now, fallback to matmul for "ij,jk->ik"
}
