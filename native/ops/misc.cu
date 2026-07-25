#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace jstorch {
namespace ops {

constexpr int MAX_DIMS = 8;

// === #10 flip ===
__global__ void flip_kernel(const float* input, float* output,
    const int* shape, const int* in_strides, int ndim, int flip_dim, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int tmp = idx, in_idx = 0;
    for (int d = ndim - 1; d >= 0; d--) {
        int coord = tmp % shape[d];
        tmp /= shape[d];
        int actual = (d == flip_dim) ? (shape[d] - 1 - coord) : coord;
        in_idx += actual * in_strides[d];
    }
    output[idx] = input[in_idx];
}

// === #11 pad (constant + reflect) ===
__global__ void pad_kernel(const float* input, float* output,
    const int* in_shape, const int* out_shape, const int* pad_before,
    int ndim, int total_out, float pad_value, int mode) {
    // mode: 0=constant, 1=reflect
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_out) return;
    int tmp = idx;
    int coords[MAX_DIMS];
    for (int d = ndim - 1; d >= 0; d--) {
        coords[d] = tmp % out_shape[d];
        tmp /= out_shape[d];
    }
    // Map output coords to input coords
    int in_idx = 0, in_stride = 1;
    bool outside = false;
    for (int d = ndim - 1; d >= 0; d--) {
        int c = coords[d] - pad_before[d];
        if (c < 0 || c >= in_shape[d]) {
            if (mode == 1) { // reflect
                if (c < 0) c = -c;
                if (c >= in_shape[d]) c = 2 * (in_shape[d] - 1) - c;
                c = max(0, min(c, in_shape[d] - 1));
            } else {
                outside = true;
                break;
            }
        }
        in_idx += c * in_stride;
        if (d > 0) {
            in_stride *= in_shape[d];
        }
    }
    output[idx] = outside ? pad_value : input[in_idx];
}

// === #12 cumsum ===
__global__ void cumsum_kernel(const float* input, float* output,
    int outer, int dim_size, int inner) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= outer * inner) return;
    int o = idx / inner, i = idx % inner;
    float sum = 0.0f;
    for (int d = 0; d < dim_size; d++) {
        sum += input[(o * dim_size + d) * inner + i];
        output[(o * dim_size + d) * inner + i] = sum;
    }
}

// === #13 where ===
__global__ void where_kernel(const float* cond, const float* x, const float* y,
    float* output, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    output[idx] = (cond[idx] > 0.0f) ? x[idx] : y[idx];
}

// === #15 randn ===
__global__ void randn_kernel(float* output, int size, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    curandState state;
    curand_init(seed, idx, 0, &state);
    output[idx] = curand_normal(&state);
}

// === #17 embedding (gather rows) ===
__global__ void embedding_kernel(const float* weight, const int* indices,
    float* output, int num_indices, int embed_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_indices * embed_dim) return;
    int row = idx / embed_dim, col = idx % embed_dim;
    output[idx] = weight[indices[row] * embed_dim + col];
}

// === #20 interpolate 1D (nearest + linear) ===
__global__ void interp1d_nearest_kernel(const float* input, float* output,
    int batch_channels, int in_len, int out_len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_channels * out_len) return;
    int bc = idx / out_len, out_i = idx % out_len;
    float scale = (float)in_len / (float)out_len;
    int in_i = min((int)(out_i * scale), in_len - 1);
    output[idx] = input[bc * in_len + in_i];
}

__global__ void interp1d_linear_kernel(const float* input, float* output,
    int batch_channels, int in_len, int out_len, bool align_corners) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_channels * out_len) return;
    int bc = idx / out_len, out_i = idx % out_len;
    float src;
    if (align_corners && out_len > 1) {
        src = (float)out_i * (float)(in_len - 1) / (float)(out_len - 1);
    } else {
        src = ((float)out_i + 0.5f) * (float)in_len / (float)out_len - 0.5f;
    }
    src = fmaxf(0.0f, fminf(src, (float)(in_len - 1)));
    int lo = (int)src;
    int hi = min(lo + 1, in_len - 1);
    float t = src - (float)lo;
    output[idx] = input[bc * in_len + lo] * (1.0f - t) + input[bc * in_len + hi] * t;
}

// === #9 cat ===
__global__ void cat_kernel(const float* src, float* dst,
    int outer, int src_dim, int dst_dim, int inner, int dst_offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outer * src_dim * inner;
    if (idx >= total) return;
    int o = idx / (src_dim * inner);
    int rem = idx % (src_dim * inner);
    int d = rem / inner;
    int i = rem % inner;
    dst[(o * dst_dim + (d + dst_offset)) * inner + i] = src[idx];
}

// === scatter_add (embedding backward) ===
// grad_output: [num_indices, embed_dim], indices: [num_indices]
// grad_weight: [vocab_size, embed_dim] — atomicAdd because indices can repeat
__global__ void scatter_add_kernel(const float* grad, const int* indices,
    float* grad_weight, int num_indices, int embed_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_indices * embed_dim) return;
    int row = idx / embed_dim, col = idx % embed_dim;
    atomicAdd(&grad_weight[indices[row] * embed_dim + col], grad[idx]);
}

// === interp1d backward ===
// Nearest: each output grad goes to exactly one input position
__global__ void interp1d_backward_nearest_kernel(const float* grad_out, float* grad_in,
    int batch_channels, int in_len, int out_len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_channels * out_len) return;
    int bc = idx / out_len, out_i = idx % out_len;
    float scale = (float)in_len / (float)out_len;
    int in_i = min((int)(out_i * scale), in_len - 1);
    atomicAdd(&grad_in[bc * in_len + in_i], grad_out[idx]);
}

// Linear: distribute grad to two neighboring input positions
__global__ void interp1d_backward_linear_kernel(const float* grad_out, float* grad_in,
    int batch_channels, int in_len, int out_len, bool align_corners) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_channels * out_len) return;
    int bc = idx / out_len, out_i = idx % out_len;
    float src;
    if (align_corners && out_len > 1)
        src = (float)out_i * (float)(in_len - 1) / (float)(out_len - 1);
    else
        src = ((float)out_i + 0.5f) * (float)in_len / (float)out_len - 0.5f;
    src = fmaxf(0.0f, fminf(src, (float)(in_len - 1)));
    int lo = (int)src;
    int hi = min(lo + 1, in_len - 1);
    float t = src - (float)lo;
    float g = grad_out[idx];
    atomicAdd(&grad_in[bc * in_len + lo], g * (1.0f - t));
    atomicAdd(&grad_in[bc * in_len + hi], g * t);
}

// === avgpool2d backward ===
// Distribute grad_output / (kH*kW) back to each contributing input position
__global__ void avgpool2d_backward_kernel(const float* grad_out, float* grad_in,
    int B, int C, int H, int W, int kH, int kW, int sH, int sW,
    int pH, int pW, int Ho, int Wo, bool count_include_pad) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C * Ho * Wo;
    if (idx >= total) return;
    int b = idx / (C*Ho*Wo);
    int r = idx % (C*Ho*Wo);
    int c = r / (Ho*Wo); r %= Ho*Wo;
    int ho = r / Wo, wo = r % Wo;
    int cnt = 0;
    if (!count_include_pad) {
        for (int kh = 0; kh < kH; kh++)
            for (int kw = 0; kw < kW; kw++) {
                int hi = ho*sH - pH + kh, wi = wo*sW - pW + kw;
                if (hi >= 0 && hi < H && wi >= 0 && wi < W) cnt++;
            }
        cnt = max(cnt, 1);
    } else cnt = kH * kW;
    float g = grad_out[idx] / (float)cnt;
    for (int kh = 0; kh < kH; kh++) {
        for (int kw = 0; kw < kW; kw++) {
            int hi = ho*sH - pH + kh, wi = wo*sW - pW + kw;
            if (hi >= 0 && hi < H && wi >= 0 && wi < W)
                atomicAdd(&grad_in[((b*C+c)*H+hi)*W+wi], g);
        }
    }
}

// === strided copy for contiguous() ===
struct StridedCopyArgs {
    int shape[8], src_strides[8], dst_strides[8];
    int ndim;
};

__global__ void strided_copy_kernel(const float* src, float* dst, StridedCopyArgs args, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int tmp = idx, src_offset = 0;
    for (int d = args.ndim - 1; d >= 0; d--) {
        int coord = tmp % args.shape[d];
        tmp /= args.shape[d];
        src_offset += coord * args.src_strides[d];
    }
    dst[idx] = src[src_offset];
}

// === Fused Adam optimizer step ===
__global__ void adam_step_kernel(
    float* param, const float* grad, float* m, float* v,
    float lr, float beta1, float beta2, float eps,
    float bc1, float bc2, float weight_decay, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    float g = grad[i] + weight_decay * param[i];
    float mi = beta1 * m[i] + (1.0f - beta1) * g;
    float vi = beta2 * v[i] + (1.0f - beta2) * g * g;
    m[i] = mi; v[i] = vi;
    float m_hat = mi / bc1;
    float v_hat = vi / bc2;
    param[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
}

// ==================== extern "C" launchers ====================
extern "C" {

void launch_flip(const float* input, float* output,
    const int* d_shape, const int* d_strides, int ndim, int flip_dim, int total, cudaStream_t s) {
    flip_kernel<<<(total+255)/256, 256, 0, s>>>(input, output, d_shape, d_strides, ndim, flip_dim, total);
}

void launch_pad(const float* input, float* output,
    const int* d_in_shape, const int* d_out_shape, const int* d_pad_before,
    int ndim, int total_out, float pad_value, int mode, cudaStream_t s) {
    pad_kernel<<<(total_out+255)/256, 256, 0, s>>>(input, output, d_in_shape, d_out_shape, d_pad_before, ndim, total_out, pad_value, mode);
}

void launch_cumsum(const float* input, float* output,
    int outer, int dim_size, int inner, cudaStream_t s) {
    int total = outer * inner;
    cumsum_kernel<<<(total+255)/256, 256, 0, s>>>(input, output, outer, dim_size, inner);
}

void launch_where(const float* cond, const float* x, const float* y,
    float* output, int total, cudaStream_t s) {
    where_kernel<<<(total+255)/256, 256, 0, s>>>(cond, x, y, output, total);
}

void launch_randn(float* output, int size, unsigned long long seed, cudaStream_t s) {
    randn_kernel<<<(size+255)/256, 256, 0, s>>>(output, size, seed);
}

void launch_embedding(const float* weight, const int* indices,
    float* output, int num_indices, int embed_dim, cudaStream_t s) {
    int total = num_indices * embed_dim;
    embedding_kernel<<<(total+255)/256, 256, 0, s>>>(weight, indices, output, num_indices, embed_dim);
}

void launch_interp1d(const float* input, float* output,
    int batch_channels, int in_len, int out_len, int mode, bool align_corners, cudaStream_t s) {
    int total = batch_channels * out_len;
    if (mode == 0) // nearest
        interp1d_nearest_kernel<<<(total+255)/256, 256, 0, s>>>(input, output, batch_channels, in_len, out_len);
    else // linear
        interp1d_linear_kernel<<<(total+255)/256, 256, 0, s>>>(input, output, batch_channels, in_len, out_len, align_corners);
}

void launch_cat(const float* src, float* dst,
    int outer, int src_dim, int dst_dim, int inner, int dst_offset, cudaStream_t s) {
    int total = outer * src_dim * inner;
    cat_kernel<<<(total+255)/256, 256, 0, s>>>(src, dst, outer, src_dim, dst_dim, inner, dst_offset);
}

void launch_scatter_add(const float* grad, const int* indices,
    float* grad_weight, int num_indices, int embed_dim, int vocab_size, cudaStream_t s) {
    // Zero grad_weight first
    cudaMemsetAsync(grad_weight, 0, vocab_size * embed_dim * sizeof(float), s);
    int total = num_indices * embed_dim;
    scatter_add_kernel<<<(total+255)/256, 256, 0, s>>>(grad, indices, grad_weight, num_indices, embed_dim);
}

void launch_interp1d_backward(const float* grad_out, float* grad_in,
    int batch_channels, int in_len, int out_len, int mode, bool align_corners, cudaStream_t s) {
    cudaMemsetAsync(grad_in, 0, batch_channels * in_len * sizeof(float), s);
    int total = batch_channels * out_len;
    if (mode == 0)
        interp1d_backward_nearest_kernel<<<(total+255)/256, 256, 0, s>>>(grad_out, grad_in, batch_channels, in_len, out_len);
    else
        interp1d_backward_linear_kernel<<<(total+255)/256, 256, 0, s>>>(grad_out, grad_in, batch_channels, in_len, out_len, align_corners);
}

void launch_avgpool2d_backward(const float* grad_out, float* grad_in,
    int B, int C, int H, int W, int kH, int kW, int sH, int sW,
    int pH, int pW, int Ho, int Wo, bool count_include_pad, cudaStream_t s) {
    cudaMemsetAsync(grad_in, 0, B * C * H * W * sizeof(float), s);
    int total = B * C * Ho * Wo;
    avgpool2d_backward_kernel<<<(total+255)/256, 256, 0, s>>>(
        grad_out, grad_in, B, C, H, W, kH, kW, sH, sW, pH, pW, Ho, Wo, count_include_pad);
}

void launch_strided_copy(const float* src, float* dst,
    const int* shape, const int* src_strides, const int* dst_strides,
    int ndim, int total, cudaStream_t s) {
    StridedCopyArgs args;
    args.ndim = ndim;
    for (int i = 0; i < ndim; i++) {
        args.shape[i] = shape[i];
        args.src_strides[i] = src_strides[i];
        args.dst_strides[i] = dst_strides[i];
    }
    strided_copy_kernel<<<(total+255)/256, 256, 0, s>>>(src, dst, args, total);
}

void launch_adam_step(float* param, const float* grad, float* m, float* v,
    float lr, float beta1, float beta2, float eps,
    float bc1, float bc2, float weight_decay, int size, cudaStream_t s) {
    adam_step_kernel<<<(size+255)/256, 256, 0, s>>>(
        param, grad, m, v, lr, beta1, beta2, eps, bc1, bc2, weight_decay, size);
}

}

}
}
