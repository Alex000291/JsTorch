#include <cuda_runtime.h>

namespace jstorch {
namespace ops {

constexpr int MAX_DIMS = 8;

template<typename BinaryOp>
__global__ void broadcast_kernel(
    const float* a, const int* a_shape, const int* a_strides,
    const float* b, const int* b_shape, const int* b_strides,
    float* out, const int* out_shape, int ndim, int total_size, BinaryOp op
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;
    
    int indices[MAX_DIMS];
    int tmp = idx;
    for (int i = ndim - 1; i >= 0; i--) {
        indices[i] = tmp % out_shape[i];
        tmp /= out_shape[i];
    }
    
    int a_idx = 0, b_idx = 0;
    for (int i = 0; i < ndim; i++) {
        a_idx += (a_shape[i] == 1 ? 0 : indices[i]) * a_strides[i];
        b_idx += (b_shape[i] == 1 ? 0 : indices[i]) * b_strides[i];
    }
    
    out[idx] = op(a[a_idx], b[b_idx]);
}

struct AddOp { __device__ float operator()(float a, float b) const { return a + b; } };
struct SubOp { __device__ float operator()(float a, float b) const { return a - b; } };
struct MulOp { __device__ float operator()(float a, float b) const { return a * b; } };
struct DivOp { __device__ float operator()(float a, float b) const { return a / b; } };

template<typename BinaryOp>
void launch_broadcast(
    const float* a, const int* a_shape, const int* a_strides,
    const float* b, const int* b_shape, const int* b_strides,
    float* out, const int* out_shape, int ndim, int total_size,
    BinaryOp op, cudaStream_t stream
) {
    int threads = 256;
    int blocks = (total_size + threads - 1) / threads;
    broadcast_kernel<<<blocks, threads, 0, stream>>>(
        a, a_shape, a_strides, b, b_shape, b_strides, out, out_shape, ndim, total_size, op
    );
}

extern "C" {

void launch_broadcast_add(
    const float* a, const int* a_shape, const int* a_strides,
    const float* b, const int* b_shape, const int* b_strides,
    float* out, const int* out_shape, int ndim, int total_size, cudaStream_t stream
) {
    launch_broadcast(a, a_shape, a_strides, b, b_shape, b_strides, out, out_shape, ndim, total_size, AddOp(), stream);
}

void launch_broadcast_sub(
    const float* a, const int* a_shape, const int* a_strides,
    const float* b, const int* b_shape, const int* b_strides,
    float* out, const int* out_shape, int ndim, int total_size, cudaStream_t stream
) {
    launch_broadcast(a, a_shape, a_strides, b, b_shape, b_strides, out, out_shape, ndim, total_size, SubOp(), stream);
}

void launch_broadcast_mul(
    const float* a, const int* a_shape, const int* a_strides,
    const float* b, const int* b_shape, const int* b_strides,
    float* out, const int* out_shape, int ndim, int total_size, cudaStream_t stream
) {
    launch_broadcast(a, a_shape, a_strides, b, b_shape, b_strides, out, out_shape, ndim, total_size, MulOp(), stream);
}

void launch_broadcast_div(
    const float* a, const int* a_shape, const int* a_strides,
    const float* b, const int* b_shape, const int* b_strides,
    float* out, const int* out_shape, int ndim, int total_size, cudaStream_t stream
) {
    launch_broadcast(a, a_shape, a_strides, b, b_shape, b_strides, out, out_shape, ndim, total_size, DivOp(), stream);
}

}

}
}
