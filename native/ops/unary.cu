#include "kernels.cuh"

namespace jstorch {
namespace ops {

// 导出C接口
extern "C" {

void launch_abs(const float* input, float* output, int size, cudaStream_t stream) {
    launch_unary(input, output, size, AbsOp(), stream);
}

void launch_sqrt(const float* input, float* output, int size, cudaStream_t stream) {
    launch_unary(input, output, size, SqrtOp(), stream);
}

void launch_square(const float* input, float* output, int size, cudaStream_t stream) {
    launch_unary(input, output, size, SquareOp(), stream);
}

void launch_exp(const float* input, float* output, int size, cudaStream_t stream) {
    launch_unary(input, output, size, ExpOp(), stream);
}

void launch_log(const float* input, float* output, int size, cudaStream_t stream) {
    launch_unary(input, output, size, LogOp(), stream);
}

void launch_sin(const float* input, float* output, int size, cudaStream_t stream) {
    launch_unary(input, output, size, SinOp(), stream);
}

void launch_cos(const float* input, float* output, int size, cudaStream_t stream) {
    launch_unary(input, output, size, CosOp(), stream);
}

void launch_sigmoid(const float* input, float* output, int size, cudaStream_t stream) {
    launch_unary(input, output, size, SigmoidOp(), stream);
}

void launch_tanh(const float* input, float* output, int size, cudaStream_t stream) {
    launch_unary(input, output, size, TanhOp(), stream);
}

void launch_relu(const float* input, float* output, int size, cudaStream_t stream) {
    launch_unary(input, output, size, ReluOp(), stream);
}
void launch_neg(const float* input, float* output, int size, cudaStream_t stream) {
    launch_unary(input, output, size, NegOp(), stream);
}
void launch_floor(const float* input, float* output, int size, cudaStream_t stream) {
    launch_unary(input, output, size, FloorOp(), stream);
}
void launch_ceil(const float* input, float* output, int size, cudaStream_t stream) {
    launch_unary(input, output, size, CeilOp(), stream);
}
void launch_round(const float* input, float* output, int size, cudaStream_t stream) {
    launch_unary(input, output, size, RoundOp(), stream);
}
void launch_silu(const float* input, float* output, int size, cudaStream_t stream) {
    launch_unary(input, output, size, SiluOp(), stream);
}
void launch_gelu(const float* input, float* output, int size, cudaStream_t stream) {
    launch_unary(input, output, size, GeluOp(), stream);
}
void launch_softplus(const float* input, float* output, int size, cudaStream_t stream) {
    launch_unary(input, output, size, SoftplusOp(), stream);
}
void launch_leaky_relu(const float* input, float* output, int size, float slope, cudaStream_t stream) {
    launch_unary(input, output, size, LeakyReluOp{slope}, stream);
}
void launch_clamp(const float* input, float* output, int size, float lo, float hi, cudaStream_t stream) {
    launch_unary(input, output, size, ClampOp{lo, hi}, stream);
}
void launch_log1p(const float* input, float* output, int size, cudaStream_t stream) {
    launch_unary(input, output, size, Log1pOp(), stream);
}
void launch_reciprocal(const float* input, float* output, int size, cudaStream_t stream) {
    launch_unary(input, output, size, ReciprocalOp(), stream);
}
void launch_sign(const float* input, float* output, int size, cudaStream_t stream) {
    launch_unary(input, output, size, SignOp(), stream);
}
void launch_fmod(const float* input, float* output, int size, float d, cudaStream_t stream) {
    launch_unary(input, output, size, FmodOp{d}, stream);
}
void launch_clamp_min(const float* input, float* output, int size, float lo, cudaStream_t stream) {
    launch_unary(input, output, size, ClampMinOp{lo}, stream);
}
void launch_clamp_max(const float* input, float* output, int size, float hi, cudaStream_t stream) {
    launch_unary(input, output, size, ClampMaxOp{hi}, stream);
}
void launch_pow_scalar(const float* input, float* output, int size, float e, cudaStream_t stream) {
    launch_unary(input, output, size, PowScalarOp{e}, stream);
}
void launch_mul_scalar(const float* input, float* output, int size, float s, cudaStream_t stream) {
    launch_unary(input, output, size, MulScalarOp{s}, stream);
}
void launch_add_scalar(const float* input, float* output, int size, float s, cudaStream_t stream) {
    launch_unary(input, output, size, AddScalarOp{s}, stream);
}

}

}
}
