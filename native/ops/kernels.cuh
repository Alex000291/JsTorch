#pragma once
#include <cuda_runtime.h>

namespace jstorch {
namespace ops {

// Vectorized unary kernel: float4 load/store, 128 threads, grid-stride loop
template<typename UnaryOp>
__global__ void unary_kernel_vec4(const float4* __restrict__ in, float4* __restrict__ out, int n4, UnaryOp op) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += blockDim.x * gridDim.x) {
        float4 v = in[i];
        v.x = op(v.x); v.y = op(v.y); v.z = op(v.z); v.w = op(v.w);
        out[i] = v;
    }
}

template<typename UnaryOp>
__global__ void unary_kernel_tail(const float* in, float* out, int offset, int size, UnaryOp op) {
    int i = offset + blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) out[i] = op(in[i]);
}

template<typename UnaryOp>
void launch_unary(const float* input, float* output, int size, UnaryOp op, cudaStream_t stream) {
    constexpr int threads = 128;
    int n4 = size / 4;
    if (n4 > 0) {
        int blocks = min((n4 + threads - 1) / threads, 1024);
        unary_kernel_vec4<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const float4*>(input),
            reinterpret_cast<float4*>(output), n4, op);
    }
    int tail = size - n4 * 4;
    if (tail > 0) {
        unary_kernel_tail<<<1, tail, 0, stream>>>(input, output, n4 * 4, size, op);
    }
}

// 函数对象
struct AbsOp { __device__ float operator()(float x) const { return fabsf(x); } };
struct SqrtOp { __device__ float operator()(float x) const { return sqrtf(x); } };
struct SquareOp { __device__ float operator()(float x) const { return x * x; } };
// __expf / __logf / __sinf / __cosf: fast intrinsics (~4 ULP error, ~2x faster than full-precision)
struct ExpOp     { __device__ float operator()(float x) const { return __expf(x); } };
struct LogOp     { __device__ float operator()(float x) const { return __logf(x); } };
struct SinOp     { __device__ float operator()(float x) const { return __sinf(x); } };
struct CosOp     { __device__ float operator()(float x) const { return __cosf(x); } };
struct SigmoidOp { __device__ float operator()(float x) const { return 1.0f / (1.0f + __expf(-x)); } };
struct TanhOp    { __device__ float operator()(float x) const { return tanhf(x); } };
struct ReluOp    { __device__ float operator()(float x) const { return fmaxf(0.0f, x); } };
struct NegOp     { __device__ float operator()(float x) const { return -x; } };
struct FloorOp   { __device__ float operator()(float x) const { return floorf(x); } };
struct CeilOp    { __device__ float operator()(float x) const { return ceilf(x); } };
struct RoundOp   { __device__ float operator()(float x) const { return roundf(x); } };
struct SiluOp    { __device__ float operator()(float x) const { return x / (1.0f + __expf(-x)); } };
struct GeluOp    { __device__ float operator()(float x) const { return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x))); } };
struct SoftplusOp{ __device__ float operator()(float x) const { return x > 20.0f ? x : __logf(1.0f + __expf(x)); } };

struct Log1pOp { __device__ float operator()(float x) const { return log1pf(x); } };
struct Atan2Op { __device__ float operator()(float x) const { return x; } }; // placeholder, atan2 is binary
struct ReciprocalOp { __device__ float operator()(float x) const { return 1.0f / x; } };
struct SignOp { __device__ float operator()(float x) const { return (x > 0.0f) - (x < 0.0f); } };

// Parameterized unary ops
struct LeakyReluOp { float slope; __device__ float operator()(float x) const { return x > 0.0f ? x : slope * x; } };
struct ClampOp { float lo, hi; __device__ float operator()(float x) const { return fminf(fmaxf(x, lo), hi); } };
struct FmodOp { float d; __device__ float operator()(float x) const { return fmodf(x, d); } };
struct ClampMinOp { float lo; __device__ float operator()(float x) const { return fmaxf(x, lo); } };
struct ClampMaxOp { float hi; __device__ float operator()(float x) const { return fminf(x, hi); } };
struct PowScalarOp { float e; __device__ float operator()(float x) const {
    // powf(negative, e) is NaN for non-integer e; handle sign correctly
    if (x >= 0.0f) return powf(x, e);
    float r = powf(-x, e);
    // Negative base: result is negative if e is odd integer, positive if even, NaN otherwise
    int ie = (int)e;
    if ((float)ie == e) return (ie % 2 != 0) ? -r : r;
    return __int_as_float(0x7FC00000); // NaN for fractional e with negative base
} };
struct MulScalarOp { float s; __device__ float operator()(float x) const { return x * s; } };
struct AddScalarOp { float s; __device__ float operator()(float x) const { return x + s; } };

}
}
