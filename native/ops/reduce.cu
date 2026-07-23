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

__global__ void reduce_max_kernel(
    const float* input, float* output, int* indices,
    int outer_size, int reduce_size, int inner_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output = outer_size * inner_size;
    if (idx >= total_output) return;
    int outer_idx = idx / inner_size;
    int inner_idx = idx % inner_size;
    float best = -1e38f;
    int best_i = 0;
    for (int i = 0; i < reduce_size; i++) {
        float v = input[(outer_idx * reduce_size + i) * inner_size + inner_idx];
        if (v > best) { best = v; best_i = i; }
    }
    output[idx] = best;
    if (indices) indices[idx] = best_i;
}

__global__ void reduce_min_kernel(
    const float* input, float* output, int* indices,
    int outer_size, int reduce_size, int inner_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output = outer_size * inner_size;
    if (idx >= total_output) return;
    int outer_idx = idx / inner_size;
    int inner_idx = idx % inner_size;
    float best = 1e38f;
    int best_i = 0;
    for (int i = 0; i < reduce_size; i++) {
        float v = input[(outer_idx * reduce_size + i) * inner_size + inner_idx];
        if (v < best) { best = v; best_i = i; }
    }
    output[idx] = best;
    if (indices) indices[idx] = best_i;
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

void launch_reduce_max(
    const float* input, float* output, int* indices,
    int outer_size, int reduce_size, int inner_size, cudaStream_t stream
) {
    int total = outer_size * inner_size;
    int threads = 256, blocks = (total + threads - 1) / threads;
    reduce_max_kernel<<<blocks, threads, 0, stream>>>(input, output, indices, outer_size, reduce_size, inner_size);
}

void launch_reduce_min(
    const float* input, float* output, int* indices,
    int outer_size, int reduce_size, int inner_size, cudaStream_t stream
) {
    int total = outer_size * inner_size;
    int threads = 256, blocks = (total + threads - 1) / threads;
    reduce_min_kernel<<<blocks, threads, 0, stream>>>(input, output, indices, outer_size, reduce_size, inner_size);
}

}

}
}
