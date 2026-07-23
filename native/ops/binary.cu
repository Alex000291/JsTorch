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

// Binary op structs
struct AddOp { __device__ float operator()(float a, float b) const { return a + b; } };
struct SubOp { __device__ float operator()(float a, float b) const { return a - b; } };
struct MulOp { __device__ float operator()(float a, float b) const { return a * b; } };
struct DivOp { __device__ float operator()(float a, float b) const { return a / b; } };
struct MaxOp { __device__ float operator()(float a, float b) const { return fmaxf(a, b); } };
struct MinOp { __device__ float operator()(float a, float b) const { return fminf(a, b); } };
struct PowOp { __device__ float operator()(float a, float b) const {
    if (a >= 0.0f) return powf(a, b);
    float r = powf(-a, b);
    int ib = (int)b;
    if ((float)ib == b) return (ib % 2 != 0) ? -r : r;
    return __int_as_float(0x7FC00000);
} };
struct GtOp  { __device__ float operator()(float a, float b) const { return a > b ? 1.0f : 0.0f; } };
struct LtOp  { __device__ float operator()(float a, float b) const { return a < b ? 1.0f : 0.0f; } };
struct GeOp  { __device__ float operator()(float a, float b) const { return a >= b ? 1.0f : 0.0f; } };
struct LeOp  { __device__ float operator()(float a, float b) const { return a <= b ? 1.0f : 0.0f; } };
struct EqOp  { __device__ float operator()(float a, float b) const { return a == b ? 1.0f : 0.0f; } };
struct NeOp  { __device__ float operator()(float a, float b) const { return a != b ? 1.0f : 0.0f; } };

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

// Macro to reduce boilerplate
#define BROADCAST_EXTERN(name, OpType) \
void launch_broadcast_##name( \
    const float* a, const int* a_shape, const int* a_strides, \
    const float* b, const int* b_shape, const int* b_strides, \
    float* out, const int* out_shape, int ndim, int total_size, cudaStream_t stream \
) { launch_broadcast(a, a_shape, a_strides, b, b_shape, b_strides, out, out_shape, ndim, total_size, OpType(), stream); }

extern "C" {
    BROADCAST_EXTERN(add, AddOp)
    BROADCAST_EXTERN(sub, SubOp)
    BROADCAST_EXTERN(mul, MulOp)
    BROADCAST_EXTERN(div, DivOp)
    BROADCAST_EXTERN(maximum, MaxOp)
    BROADCAST_EXTERN(minimum, MinOp)
    BROADCAST_EXTERN(pow, PowOp)
    BROADCAST_EXTERN(gt, GtOp)
    BROADCAST_EXTERN(lt, LtOp)
    BROADCAST_EXTERN(ge, GeOp)
    BROADCAST_EXTERN(le, LeOp)
    BROADCAST_EXTERN(eq, EqOp)
    BROADCAST_EXTERN(ne, NeOp)
}

}
}
