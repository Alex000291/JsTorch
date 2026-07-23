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

}

}
}
