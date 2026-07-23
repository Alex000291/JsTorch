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

}

}
}
