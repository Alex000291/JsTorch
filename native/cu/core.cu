// core.cu - GPU kernels for element-wise operations
#include <cuda_runtime.h>

// ==================== Element-wise Addition ====================

__global__ void add_kernel(const float* A, const float* B, float* C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] + B[idx];
    }
}

extern "C" void launch_add(const float* A, const float* B, float* C, 
                          int size, cudaStream_t stream) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    
    add_kernel<<<blocks, threads, 0, stream>>>(A, B, C, size);
}

// ==================== Element-wise Subtraction ====================

__global__ void sub_kernel(const float* A, const float* B, float* C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] - B[idx];
    }
}

extern "C" void launch_sub(const float* A, const float* B, float* C,
                          int size, cudaStream_t stream) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    
    sub_kernel<<<blocks, threads, 0, stream>>>(A, B, C, size);
}

// ==================== Element-wise Multiplication ====================

__global__ void mul_kernel(const float* A, const float* B, float* C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] * B[idx];
    }
}

extern "C" void launch_mul(const float* A, const float* B, float* C,
                          int size, cudaStream_t stream) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    
    mul_kernel<<<blocks, threads, 0, stream>>>(A, B, C, size);
}

// ==================== Element-wise Division ====================

__global__ void div_kernel(const float* A, const float* B, float* C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] / B[idx];
    }
}

extern "C" void launch_div(const float* A, const float* B, float* C,
                          int size, cudaStream_t stream) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    
    div_kernel<<<blocks, threads, 0, stream>>>(A, B, C, size);
}

// ==================== Scalar Multiplication ====================

__global__ void mul_scalar_kernel(const float* input, float scalar, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] * scalar;
    }
}

extern "C" void launch_mul_scalar(const float* input, float scalar, float* output,
                                 int size, cudaStream_t stream) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    
    mul_scalar_kernel<<<blocks, threads, 0, stream>>>(input, scalar, output, size);
}

// ==================== ReLU Activation ====================

__global__ void relu_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

extern "C" void launch_relu(const float* input, float* output,
                           int size, cudaStream_t stream) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    
    relu_kernel<<<blocks, threads, 0, stream>>>(input, output, size);
}

// ==================== Exp ====================

__global__ void exp_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = expf(input[idx]);
    }
}

extern "C" void launch_exp(const float* input, float* output,
                          int size, cudaStream_t stream) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    
    exp_kernel<<<blocks, threads, 0, stream>>>(input, output, size);
}

// ==================== Log ====================

__global__ void log_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = logf(input[idx]);
    }
}

extern "C" void launch_log(const float* input, float* output,
                          int size, cudaStream_t stream) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    
    log_kernel<<<blocks, threads, 0, stream>>>(input, output, size);
}

// ==================== Negate ====================

__global__ void neg_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = -input[idx];
    }
}

extern "C" void launch_neg(const float* input, float* output,
                          int size, cudaStream_t stream) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    
    neg_kernel<<<blocks, threads, 0, stream>>>(input, output, size);
}

// ==================== Sum Reduce ====================

__global__ void sum_reduce_kernel(const float* input, float* output, int size) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (idx < size) ? input[idx] : 0.0f;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

extern "C" void launch_sum(const float* input, float* output, int size, cudaStream_t stream) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    
    // Initialize output to 0
    cudaMemsetAsync(output, 0, sizeof(float), stream);
    
    sum_reduce_kernel<<<blocks, threads, threads * sizeof(float), stream>>>(input, output, size);
}

// ==================== Fill (for zeros/ones) ====================

__global__ void fill_kernel(float* data, float value, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = value;
    }
}

extern "C" void launch_fill(float* data, float value, int size, cudaStream_t stream) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    
    fill_kernel<<<blocks, threads, 0, stream>>>(data, value, size);
}
