#include <cuda_runtime.h>
#include <cfloat>

namespace jstorch {
namespace ops {

// ==================== Parallel reduction with shared memory ====================
// Strategy:
//   - Each output element = reduction of `reduce_size` values along one axis
//   - Layout: input[outer][reduce][inner], output[outer][inner]
//   - For large reduce_size: launch BLOCK_DIM threads per output element,
//     each thread handles a chunk, then parallel reduce in shared memory
//   - For small reduce_size (<=32): use the old sequential kernel (less overhead)

constexpr int BLOCK_DIM = 256;

// --- Warp-level reduce ---
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val, int& idx, int lane_idx) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(0xFFFFFFFF, val, offset);
        int other_idx = __shfl_down_sync(0xFFFFFFFF, idx, offset);
        if (other > val) { val = other; idx = other_idx; }
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_min(float val, int& idx, int lane_idx) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(0xFFFFFFFF, val, offset);
        int other_idx = __shfl_down_sync(0xFFFFFFFF, idx, offset);
        if (other < val) { val = other; idx = other_idx; }
    }
    return val;
}

// ==================== Sum ====================

// Parallel: 1 block per output element, BLOCK_DIM threads cooperate
__global__ void reduce_sum_parallel(
    const float* __restrict__ input, float* __restrict__ output,
    int outer_size, int reduce_size, int inner_size
) {
    // blockIdx.x = output element index
    int out_idx = blockIdx.x;
    if (out_idx >= outer_size * inner_size) return;

    int outer_idx = out_idx / inner_size;
    int inner_idx = out_idx % inner_size;
    int base = outer_idx * reduce_size * inner_size + inner_idx;

    // Each thread sums a strided chunk
    float sum = 0.0f;
    for (int i = threadIdx.x; i < reduce_size; i += BLOCK_DIM)
        sum += input[base + i * inner_size];

    // Warp reduce
    sum = warp_reduce_sum(sum);

    // Cross-warp reduce via shared memory
    __shared__ float sdata[BLOCK_DIM / 32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) sdata[warp] = sum;
    __syncthreads();

    // First warp reduces the warp results
    if (warp == 0) {
        int nwarps = (BLOCK_DIM + 31) / 32;
        sum = (lane < nwarps) ? sdata[lane] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (lane == 0) output[out_idx] = sum;
    }
}

// Sequential: 1 thread per output element (for small reduce_size)
__global__ void reduce_sum_seq(
    const float* __restrict__ input, float* __restrict__ output,
    int outer_size, int reduce_size, int inner_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outer_size * inner_size;
    if (idx >= total) return;
    int outer_idx = idx / inner_size;
    int inner_idx = idx % inner_size;
    int base = outer_idx * reduce_size * inner_size + inner_idx;
    float sum = 0.0f;
    for (int i = 0; i < reduce_size; i++)
        sum += input[base + i * inner_size];
    output[idx] = sum;
}

// ==================== Mean ====================

__global__ void reduce_mean_parallel(
    const float* __restrict__ input, float* __restrict__ output,
    int outer_size, int reduce_size, int inner_size
) {
    int out_idx = blockIdx.x;
    if (out_idx >= outer_size * inner_size) return;
    int outer_idx = out_idx / inner_size;
    int inner_idx = out_idx % inner_size;
    int base = outer_idx * reduce_size * inner_size + inner_idx;

    float sum = 0.0f;
    for (int i = threadIdx.x; i < reduce_size; i += BLOCK_DIM)
        sum += input[base + i * inner_size];

    sum = warp_reduce_sum(sum);

    __shared__ float sdata[BLOCK_DIM / 32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) sdata[warp] = sum;
    __syncthreads();

    if (warp == 0) {
        int nwarps = (BLOCK_DIM + 31) / 32;
        sum = (lane < nwarps) ? sdata[lane] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (lane == 0) output[out_idx] = sum / reduce_size;
    }
}

__global__ void reduce_mean_seq(
    const float* __restrict__ input, float* __restrict__ output,
    int outer_size, int reduce_size, int inner_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outer_size * inner_size;
    if (idx >= total) return;
    int outer_idx = idx / inner_size;
    int inner_idx = idx % inner_size;
    int base = outer_idx * reduce_size * inner_size + inner_idx;
    float sum = 0.0f;
    for (int i = 0; i < reduce_size; i++)
        sum += input[base + i * inner_size];
    output[idx] = sum / reduce_size;
}

// ==================== Max ====================

__global__ void reduce_max_parallel(
    const float* __restrict__ input, float* __restrict__ output, int* __restrict__ indices,
    int outer_size, int reduce_size, int inner_size
) {
    int out_idx = blockIdx.x;
    if (out_idx >= outer_size * inner_size) return;
    int outer_idx = out_idx / inner_size;
    int inner_idx = out_idx % inner_size;
    int base = outer_idx * reduce_size * inner_size + inner_idx;

    float best = -FLT_MAX;
    int best_i = 0;
    for (int i = threadIdx.x; i < reduce_size; i += BLOCK_DIM) {
        float v = input[base + i * inner_size];
        if (v > best) { best = v; best_i = i; }
    }

    // Warp reduce
    int lane = threadIdx.x & 31;
    best = warp_reduce_max(best, best_i, lane);

    __shared__ float sval[BLOCK_DIM / 32];
    __shared__ int sidx[BLOCK_DIM / 32];
    int warp = threadIdx.x >> 5;
    if (lane == 0) { sval[warp] = best; sidx[warp] = best_i; }
    __syncthreads();

    if (warp == 0) {
        int nwarps = (BLOCK_DIM + 31) / 32;
        best = (lane < nwarps) ? sval[lane] : -FLT_MAX;
        best_i = (lane < nwarps) ? sidx[lane] : 0;
        best = warp_reduce_max(best, best_i, lane);
        if (lane == 0) {
            output[out_idx] = best;
            if (indices) indices[out_idx] = best_i;
        }
    }
}

__global__ void reduce_max_seq(
    const float* __restrict__ input, float* __restrict__ output, int* __restrict__ indices,
    int outer_size, int reduce_size, int inner_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outer_size * inner_size;
    if (idx >= total) return;
    int outer_idx = idx / inner_size;
    int inner_idx = idx % inner_size;
    int base = outer_idx * reduce_size * inner_size + inner_idx;
    float best = -FLT_MAX;
    int best_i = 0;
    for (int i = 0; i < reduce_size; i++) {
        float v = input[base + i * inner_size];
        if (v > best) { best = v; best_i = i; }
    }
    output[idx] = best;
    if (indices) indices[idx] = best_i;
}

// ==================== Min ====================

__global__ void reduce_min_parallel(
    const float* __restrict__ input, float* __restrict__ output, int* __restrict__ indices,
    int outer_size, int reduce_size, int inner_size
) {
    int out_idx = blockIdx.x;
    if (out_idx >= outer_size * inner_size) return;
    int outer_idx = out_idx / inner_size;
    int inner_idx = out_idx % inner_size;
    int base = outer_idx * reduce_size * inner_size + inner_idx;

    float best = FLT_MAX;
    int best_i = 0;
    for (int i = threadIdx.x; i < reduce_size; i += BLOCK_DIM) {
        float v = input[base + i * inner_size];
        if (v < best) { best = v; best_i = i; }
    }

    int lane = threadIdx.x & 31;
    best = warp_reduce_min(best, best_i, lane);

    __shared__ float sval[BLOCK_DIM / 32];
    __shared__ int sidx[BLOCK_DIM / 32];
    int warp = threadIdx.x >> 5;
    if (lane == 0) { sval[warp] = best; sidx[warp] = best_i; }
    __syncthreads();

    if (warp == 0) {
        int nwarps = (BLOCK_DIM + 31) / 32;
        best = (lane < nwarps) ? sval[lane] : FLT_MAX;
        best_i = (lane < nwarps) ? sidx[lane] : 0;
        best = warp_reduce_min(best, best_i, lane);
        if (lane == 0) {
            output[out_idx] = best;
            if (indices) indices[out_idx] = best_i;
        }
    }
}

__global__ void reduce_min_seq(
    const float* __restrict__ input, float* __restrict__ output, int* __restrict__ indices,
    int outer_size, int reduce_size, int inner_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outer_size * inner_size;
    if (idx >= total) return;
    int outer_idx = idx / inner_size;
    int inner_idx = idx % inner_size;
    int base = outer_idx * reduce_size * inner_size + inner_idx;
    float best = FLT_MAX;
    int best_i = 0;
    for (int i = 0; i < reduce_size; i++) {
        float v = input[base + i * inner_size];
        if (v < best) { best = v; best_i = i; }
    }
    output[idx] = best;
    if (indices) indices[idx] = best_i;
}

// ==================== Launchers ====================
// Threshold: use parallel kernel when reduce_size > 32

extern "C" {

void launch_reduce_sum(
    const float* input, float* output,
    int outer_size, int reduce_size, int inner_size,
    cudaStream_t stream
) {
    int total = outer_size * inner_size;
    if (reduce_size > 32) {
        reduce_sum_parallel<<<total, BLOCK_DIM, 0, stream>>>(input, output, outer_size, reduce_size, inner_size);
    } else {
        int threads = 256, blocks = (total + threads - 1) / threads;
        reduce_sum_seq<<<blocks, threads, 0, stream>>>(input, output, outer_size, reduce_size, inner_size);
    }
}

void launch_reduce_mean(
    const float* input, float* output,
    int outer_size, int reduce_size, int inner_size,
    cudaStream_t stream
) {
    int total = outer_size * inner_size;
    if (reduce_size > 32) {
        reduce_mean_parallel<<<total, BLOCK_DIM, 0, stream>>>(input, output, outer_size, reduce_size, inner_size);
    } else {
        int threads = 256, blocks = (total + threads - 1) / threads;
        reduce_mean_seq<<<blocks, threads, 0, stream>>>(input, output, outer_size, reduce_size, inner_size);
    }
}

void launch_reduce_max(
    const float* input, float* output, int* indices,
    int outer_size, int reduce_size, int inner_size, cudaStream_t stream
) {
    int total = outer_size * inner_size;
    if (reduce_size > 32) {
        reduce_max_parallel<<<total, BLOCK_DIM, 0, stream>>>(input, output, indices, outer_size, reduce_size, inner_size);
    } else {
        int threads = 256, blocks = (total + threads - 1) / threads;
        reduce_max_seq<<<blocks, threads, 0, stream>>>(input, output, indices, outer_size, reduce_size, inner_size);
    }
}

void launch_reduce_min(
    const float* input, float* output, int* indices,
    int outer_size, int reduce_size, int inner_size, cudaStream_t stream
) {
    int total = outer_size * inner_size;
    if (reduce_size > 32) {
        reduce_min_parallel<<<total, BLOCK_DIM, 0, stream>>>(input, output, indices, outer_size, reduce_size, inner_size);
    } else {
        int threads = 256, blocks = (total + threads - 1) / threads;
        reduce_min_seq<<<blocks, threads, 0, stream>>>(input, output, indices, outer_size, reduce_size, inner_size);
    }
}

}

}
}
