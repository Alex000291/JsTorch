#include <cuda_runtime.h>

namespace jstorch {
namespace ops {

__global__ void reduce_sum_kernel(
    const float* input, float* output,
    int outer_size, int reduce_size, int inner_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output = outer_size * inner_size;
    if (idx >= total_output) return;
    
    int outer_idx = idx / inner_size;
    int inner_idx = idx % inner_size;
    
    float sum = 0.0f;
    for (int i = 0; i < reduce_size; i++) {
        int input_idx = (outer_idx * reduce_size + i) * inner_size + inner_idx;
        sum += input[input_idx];
    }
    output[idx] = sum;
}

__global__ void reduce_mean_kernel(
    const float* input, float* output,
    int outer_size, int reduce_size, int inner_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output = outer_size * inner_size;
    if (idx >= total_output) return;
    
    int outer_idx = idx / inner_size;
    int inner_idx = idx % inner_size;
    
    float sum = 0.0f;
    for (int i = 0; i < reduce_size; i++) {
        int input_idx = (outer_idx * reduce_size + i) * inner_size + inner_idx;
        sum += input[input_idx];
    }
    output[idx] = sum / reduce_size;
}

extern "C" {

void launch_reduce_sum(
    const float* input, float* output,
    int outer_size, int reduce_size, int inner_size,
    cudaStream_t stream
) {
    int total_output = outer_size * inner_size;
    int threads = 256;
    int blocks = (total_output + threads - 1) / threads;
    reduce_sum_kernel<<<blocks, threads, 0, stream>>>(input, output, outer_size, reduce_size, inner_size);
}

void launch_reduce_mean(
    const float* input, float* output,
    int outer_size, int reduce_size, int inner_size,
    cudaStream_t stream
) {
    int total_output = outer_size * inner_size;
    int threads = 256;
    int blocks = (total_output + threads - 1) / threads;
    reduce_mean_kernel<<<blocks, threads, 0, stream>>>(input, output, outer_size, reduce_size, inner_size);
}

}

}
}
