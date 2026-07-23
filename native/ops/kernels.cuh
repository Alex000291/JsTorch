#pragma once
#include <cuda_runtime.h>

namespace jstorch {
namespace ops {

// 通用一元操作kernel
template<typename UnaryOp>
__global__ void unary_kernel(const float* input, float* output, int size, UnaryOp op) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) output[idx] = op(input[idx]);
}

template<typename UnaryOp>
void launch_unary(const float* input, float* output, int size, UnaryOp op, cudaStream_t stream) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    unary_kernel<<<blocks, threads, 0, stream>>>(input, output, size, op);
}

// 函数对象
struct AbsOp { __device__ float operator()(float x) const { return fabsf(x); } };
struct SqrtOp { __device__ float operator()(float x) const { return sqrtf(x); } };
struct SquareOp { __device__ float operator()(float x) const { return x * x; } };
struct ExpOp { __device__ float operator()(float x) const { return expf(x); } };
struct LogOp { __device__ float operator()(float x) const { return logf(x); } };
struct SinOp { __device__ float operator()(float x) const { return sinf(x); } };
struct CosOp { __device__ float operator()(float x) const { return cosf(x); } };
struct SigmoidOp { __device__ float operator()(float x) const { return 1.0f / (1.0f + expf(-x)); } };
struct TanhOp { __device__ float operator()(float x) const { return tanhf(x); } };
struct ReluOp { __device__ float operator()(float x) const { return fmaxf(0.0f, x); } };
struct NegOp { __device__ float operator()(float x) const { return -x; } };
struct FloorOp { __device__ float operator()(float x) const { return floorf(x); } };
struct CeilOp { __device__ float operator()(float x) const { return ceilf(x); } };
struct RoundOp { __device__ float operator()(float x) const { return roundf(x); } };
struct SiluOp { __device__ float operator()(float x) const { return x / (1.0f + expf(-x)); } };
struct GeluOp { __device__ float operator()(float x) const { return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x))); } };
struct SoftplusOp { __device__ float operator()(float x) const { return x > 20.0f ? x : logf(1.0f + expf(x)); } };

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
struct PowScalarOp { float e; __device__ float operator()(float x) const { return powf(x, e); } };
struct MulScalarOp { float s; __device__ float operator()(float x) const { return x * s; } };
struct AddScalarOp { float s; __device__ float operator()(float x) const { return x + s; } };

}
}
