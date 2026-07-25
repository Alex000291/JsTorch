#include <cuda_runtime.h>

namespace jstorch {
namespace ops {

constexpr int MAX_DIMS = 8;

struct BroadcastArgs {
    int a_shape[MAX_DIMS], a_strides[MAX_DIMS];
    int b_shape[MAX_DIMS], b_strides[MAX_DIMS];
    int out_shape[MAX_DIMS];
    int ndim;
};

// Fast path: contiguous same-shape, vectorized float4
template<typename BinaryOp>
__global__ void binary_vec4(const float4* __restrict__ a, const float4* __restrict__ b,
                            float4* __restrict__ out, int n4, BinaryOp op) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += blockDim.x * gridDim.x) {
        float4 va = a[i], vb = b[i], vo;
        vo.x = op(va.x, vb.x); vo.y = op(va.y, vb.y);
        vo.z = op(va.z, vb.z); vo.w = op(va.w, vb.w);
        out[i] = vo;
    }
}

template<typename BinaryOp>
__global__ void binary_tail(const float* a, const float* b, float* out, int offset, int size, BinaryOp op) {
    int i = offset + blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) out[i] = op(a[i], b[i]);
}

// Broadcast path: index calculation per element
template<typename BinaryOp>
__global__ void broadcast_kernel(
    const float* a, const float* b,
    float* out, BroadcastArgs args, int total_size, BinaryOp op
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;
    
    int indices[MAX_DIMS];
    int tmp = idx;
    for (int i = args.ndim - 1; i >= 0; i--) {
        indices[i] = tmp % args.out_shape[i];
        tmp /= args.out_shape[i];
    }
    
    int a_idx = 0, b_idx = 0;
    for (int i = 0; i < args.ndim; i++) {
        a_idx += (args.a_shape[i] == 1 ? 0 : indices[i]) * args.a_strides[i];
        b_idx += (args.b_shape[i] == 1 ? 0 : indices[i]) * args.b_strides[i];
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
    // Fast path: same shape, both contiguous → vectorized
    bool same_shape = true;
    for (int i = 0; i < ndim; i++) {
        if (a_shape[i] != b_shape[i]) { same_shape = false; break; }
    }
    if (same_shape) {
        constexpr int threads = 128;
        int n4 = total_size / 4;
        if (n4 > 0) {
            int blocks = min((n4 + threads - 1) / threads, 1024);
            binary_vec4<<<blocks, threads, 0, stream>>>(
                reinterpret_cast<const float4*>(a),
                reinterpret_cast<const float4*>(b),
                reinterpret_cast<float4*>(out), n4, op);
        }
        int tail = total_size - n4 * 4;
        if (tail > 0)
            binary_tail<<<1, tail, 0, stream>>>(a, b, out, n4 * 4, total_size, op);
        return;
    }

    // Broadcast path
    BroadcastArgs args;
    args.ndim = ndim;
    for (int i = 0; i < ndim; i++) {
        args.a_shape[i] = a_shape[i];
        args.a_strides[i] = a_strides[i];
        args.b_shape[i] = b_shape[i];
        args.b_strides[i] = b_strides[i];
        args.out_shape[i] = out_shape[i];
    }
    int threads = 128;
    int blocks = min((total_size + threads - 1) / threads, 2048);
    broadcast_kernel<<<blocks, threads, 0, stream>>>(a, b, out, args, total_size, op);
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
