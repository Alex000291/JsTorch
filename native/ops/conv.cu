#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <map>
#include <tuple>

namespace jstorch {
namespace ops {

static cudnnHandle_t get_cudnn_handle() {
    static cudnnHandle_t h = nullptr;
    if (!h) cudnnCreate(&h);
    return h;
}

struct ConvParams {
    int B, Ci, H, W, Co, kH, kW, sH, sW, pH, pW, dH, dW, groups;
    bool operator<(const ConvParams& o) const {
        return std::tie(B,Ci,H,W,Co,kH,kW,sH,sW,pH,pW,dH,dW,groups)
             < std::tie(o.B,o.Ci,o.H,o.W,o.Co,o.kH,o.kW,o.sH,o.sW,o.pH,o.pW,o.dH,o.dW,o.groups);
    }
};

// Shared workspace: all cuDNN ops share one buffer (only one op runs at a time)
void* shared_ws_ptr = nullptr;
size_t shared_ws_cap = 0;

void* get_workspace(size_t needed) {
    if (needed == 0) return nullptr;
    if (needed > shared_ws_cap) {
        if (shared_ws_ptr) cudaFree(shared_ws_ptr);
        cudaMalloc(&shared_ws_ptr, needed);
        shared_ws_cap = needed;
    }
    return shared_ws_ptr;
}

struct ConvPlan {
    cudnnConvolutionFwdAlgo_t algo;
    size_t ws_size;
};

// === Conv1d via im2col + matmul ===

// im2col: [B, C_in, L_in] -> [B, C_in*K, L_out]
__global__ void im2col_1d_kernel(const float* input, float* col,
    int batch, int C_in, int L_in, int K, int stride, int padding, int dilation, int L_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * C_in * K * L_out;
    if (idx >= total) return;
    int b = idx / (C_in * K * L_out);
    int rem = idx % (C_in * K * L_out);
    int c = rem / (K * L_out);
    rem = rem % (K * L_out);
    int k = rem / L_out;
    int l = rem % L_out;
    int in_pos = l * stride - padding + k * dilation;
    float val = 0.0f;
    if (in_pos >= 0 && in_pos < L_in)
        val = input[(b * C_in + c) * L_in + in_pos];
    col[(b * C_in * K + c * K + k) * L_out + l] = val;
}

// col2im for ConvTranspose1d: accumulate from col back to input
__global__ void col2im_1d_kernel(const float* col, float* output,
    int batch, int C_out, int L_out, int K, int stride, int padding, int dilation, int L_in) {
    // col shape: [B, C_out*K, L_in]
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * C_out * L_out;
    if (idx >= total) return;
    int b = idx / (C_out * L_out);
    int rem = idx % (C_out * L_out);
    int c = rem / L_out;
    int l_out = rem % L_out;
    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        // l_out = l_in * stride - padding + k * dilation
        // l_in = (l_out + padding - k * dilation) / stride
        int num = l_out + padding - k * dilation;
        if (num >= 0 && num % stride == 0) {
            int l_in = num / stride;
            if (l_in < L_in)
                sum += col[(b * C_out * K + c * K + k) * L_in + l_in];
        }
    }
    output[(b * C_out + c) * L_out + l_out] = sum;
}

// Add bias: output[b, c, l] += bias[c]
__global__ void add_bias_1d_kernel(float* output, const float* bias,
    int batch, int channels, int length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch * channels * length) return;
    int c = (idx / length) % channels;
    output[idx] += bias[c];
}

static cublasHandle_t get_handle() {
    static cublasHandle_t h = nullptr;
    if (!h) cublasCreate(&h);
    return h;
}

// ==================== 2D Convolution ====================

// im2col 2D: input [B,C,H,W] -> col [B, C*kH*kW, H_out*W_out]
__global__ void im2col_2d_kernel(const float* input, float* col,
    int B, int C, int H, int W, int kH, int kW,
    int sH, int sW, int pH, int pW, int dH, int dW,
    int Ho, int Wo) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C * kH * kW * Ho * Wo;
    if (idx >= total) return;
    int b = idx / (C*kH*kW*Ho*Wo);
    int r = idx % (C*kH*kW*Ho*Wo);
    int c = r / (kH*kW*Ho*Wo); r %= kH*kW*Ho*Wo;
    int kh = r / (kW*Ho*Wo); r %= kW*Ho*Wo;
    int kw = r / (Ho*Wo); r %= Ho*Wo;
    int ho = r / Wo, wo = r % Wo;
    int hi = ho*sH - pH + kh*dH;
    int wi = wo*sW - pW + kw*dW;
    float val = (hi >= 0 && hi < H && wi >= 0 && wi < W)
        ? input[((b*C+c)*H+hi)*W+wi] : 0.0f;
    col[((b*C*kH*kW + c*kH*kW + kh*kW + kw)*Ho + ho)*Wo + wo] = val;
}

// col2im 2D for ConvTranspose2d
__global__ void col2im_2d_kernel(const float* col, float* output,
    int B, int C, int H, int W, int kH, int kW,
    int sH, int sW, int pH, int pW, int dH, int dW,
    int Hi, int Wi) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C * H * W;
    if (idx >= total) return;
    int b = idx / (C*H*W);
    int r = idx % (C*H*W);
    int c = r / (H*W); r %= H*W;
    int h = r / W, w = r % W;
    float sum = 0.0f;
    for (int kh = 0; kh < kH; kh++) {
        for (int kw = 0; kw < kW; kw++) {
            int nh = h + pH - kh * dH;
            int nw = w + pW - kw * dW;
            if (nh >= 0 && nh % sH == 0 && nw >= 0 && nw % sW == 0) {
                int hi = nh / sH, wi = nw / sW;
                if (hi < Hi && wi < Wi)
                    sum += col[((b*C*kH*kW + c*kH*kW + kh*kW + kw)*Hi + hi)*Wi + wi];
            }
        }
    }
    output[idx] = sum;
}

// Add bias 2D: output[b,c,h,w] += bias[c]
__global__ void add_bias_2d_kernel(float* output, const float* bias,
    int B, int C, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C * H * W;
    if (idx >= total) return;
    int c = (idx / (H * W)) % C;
    output[idx] += bias[c];
}

// AvgPool2d
__global__ void avgpool2d_kernel(const float* input, float* output,
    int B, int C, int H, int W, int kH, int kW, int sH, int sW,
    int pH, int pW, int Ho, int Wo, bool count_include_pad) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C * Ho * Wo;
    if (idx >= total) return;
    int b = idx / (C*Ho*Wo);
    int r = idx % (C*Ho*Wo);
    int c = r / (Ho*Wo); r %= Ho*Wo;
    int ho = r / Wo, wo = r % Wo;
    float sum = 0.0f; int cnt = 0;
    for (int kh = 0; kh < kH; kh++) {
        for (int kw = 0; kw < kW; kw++) {
            int hi = ho*sH - pH + kh;
            int wi = wo*sW - pW + kw;
            if (hi >= 0 && hi < H && wi >= 0 && wi < W) {
                sum += input[((b*C+c)*H+hi)*W+wi];
                cnt++;
            }
        }
    }
    int denom = count_include_pad ? (kH * kW) : max(cnt, 1);
    output[idx] = sum / denom;
}

// === MaxPool2d ===

__global__ void maxpool2d_kernel(const float* input, float* output, int* indices,
    int B, int C, int H, int W, int kH, int kW, int sH, int sW,
    int pH, int pW, int Ho, int Wo) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C * Ho * Wo;
    if (idx >= total) return;
    int b = idx / (C*Ho*Wo);
    int r = idx % (C*Ho*Wo);
    int c = r / (Ho*Wo); r %= Ho*Wo;
    int ho = r / Wo, wo = r % Wo;
    float best = -1e38f;
    int best_idx = 0;
    for (int kh = 0; kh < kH; kh++) {
        for (int kw = 0; kw < kW; kw++) {
            int hi = ho*sH - pH + kh;
            int wi = wo*sW - pW + kw;
            if (hi >= 0 && hi < H && wi >= 0 && wi < W) {
                int in_idx = ((b*C+c)*H+hi)*W+wi;
                float v = input[in_idx];
                if (v > best) { best = v; best_idx = in_idx; }
            }
        }
    }
    output[idx] = best;
    indices[idx] = best_idx;
}

__global__ void maxpool2d_backward_kernel(const float* grad_output, const int* indices,
    float* grad_input, int out_total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_total) return;
    atomicAdd(&grad_input[indices[idx]], grad_output[idx]);
}

extern "C" {

// Conv1d: input [B, C_in, L_in], weight [C_out, C_in/groups, K] -> [B, C_out, L_out]
void launch_conv1d(const float* input, const float* weight, const float* bias,
    float* output, float* col_buf,
    int B, int C_in, int L_in, int C_out, int K,
    int stride, int padding, int dilation, int groups, int L_out, cudaStream_t s) {
    
    cublasHandle_t h = get_handle();
    cublasSetStream(h, s);
    float alpha = 1.0f, beta = 0.0f;
    
    int C_in_g = C_in / groups;
    int C_out_g = C_out / groups;
    
    // im2col
    int col_total = B * C_in * K * L_out;
    im2col_1d_kernel<<<(col_total+255)/256, 256, 0, s>>>(
        input, col_buf, B, C_in, L_in, K, stride, padding, dilation, L_out);
    
    // For each batch and group: output = weight * col
    // weight: [C_out_g, C_in_g * K], col: [C_in_g * K, L_out] -> [C_out_g, L_out]
    for (int b = 0; b < B; b++) {
        for (int g = 0; g < groups; g++) {
            const float* w = weight + g * C_out_g * C_in_g * K;
            const float* c = col_buf + (b * C_in * K + g * C_in_g * K) * L_out;
            float* o = output + (b * C_out + g * C_out_g) * L_out;
            int M = C_out_g, KK = C_in_g * K, N = L_out;
            cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, KK,
                &alpha, c, N, w, KK, &beta, o, N);
        }
    }
    
    // Add bias
    if (bias) {
        int total = B * C_out * L_out;
        add_bias_1d_kernel<<<(total+255)/256, 256, 0, s>>>(output, bias, B, C_out, L_out);
    }
}

// ConvTranspose1d: input [B, C_in, L_in], weight [C_in, C_out/groups, K] -> [B, C_out, L_out]
// L_out = (L_in - 1) * stride - 2*padding + dilation*(K-1) + output_padding + 1
void launch_conv_transpose1d(const float* input, const float* weight, const float* bias,
    float* output, float* col_buf,
    int B, int C_in, int L_in, int C_out, int K,
    int stride, int padding, int dilation, int groups, int L_out, cudaStream_t s) {
    
    cublasHandle_t h = get_handle();
    cublasSetStream(h, s);
    float alpha = 1.0f, beta = 0.0f;
    
    int C_in_g = C_in / groups;
    int C_out_g = C_out / groups;
    
    // col = weight^T * input  -> col shape: [B, C_out_g * K, L_in] per group
    // weight: [C_in_g, C_out_g * K] (transposed view)
    for (int b = 0; b < B; b++) {
        for (int g = 0; g < groups; g++) {
            const float* w = weight + g * C_in_g * C_out_g * K;
            const float* inp = input + (b * C_in + g * C_in_g) * L_in;
            float* c = col_buf + (b * C_out * K + g * C_out_g * K) * L_in;
            int M = C_out_g * K, KK = C_in_g, N = L_in;
            // w^T: [C_out_g*K, C_in_g] * inp: [C_in_g, L_in] -> [C_out_g*K, L_in]
            cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_T, N, M, KK,
                &alpha, inp, N, w, M, &beta, c, N);
        }
    }
    
    // col2im
    int out_total = B * C_out * L_out;
    cudaMemsetAsync(output, 0, out_total * sizeof(float), s);
    col2im_1d_kernel<<<(out_total+255)/256, 256, 0, s>>>(
        col_buf, output, B, C_out, L_out, K, stride, padding, dilation, L_in);
    
    // Add bias
    if (bias) {
        add_bias_1d_kernel<<<(out_total+255)/256, 256, 0, s>>>(output, bias, B, C_out, L_out);
    }
}

// Conv2d: input [B,Ci,H,W], weight [Co,Ci/g,kH,kW] -> [B,Co,Ho,Wo]
void launch_conv2d(const float* input, const float* weight, const float* bias,
    float* output, float* col_buf,
    int B, int Ci, int H, int W, int Co, int kH, int kW,
    int sH, int sW, int pH, int pW, int dH, int dW,
    int groups, int Ho, int Wo, cudaStream_t s) {
    
    cublasHandle_t h = get_handle();
    cublasSetStream(h, s);
    float alpha = 1.0f, beta = 0.0f;
    int Ci_g = Ci / groups, Co_g = Co / groups;
    
    // im2col
    int col_total = B * Ci * kH * kW * Ho * Wo;
    im2col_2d_kernel<<<(col_total+255)/256, 256, 0, s>>>(
        input, col_buf, B, Ci, H, W, kH, kW, sH, sW, pH, pW, dH, dW, Ho, Wo);
    
    int N = Ho * Wo;
    for (int b = 0; b < B; b++) {
        for (int g = 0; g < groups; g++) {
            const float* w = weight + g * Co_g * Ci_g * kH * kW;
            const float* c = col_buf + (b * Ci * kH * kW + g * Ci_g * kH * kW) * N;
            float* o = output + (b * Co + g * Co_g) * N;
            int M = Co_g, K = Ci_g * kH * kW;
            cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                &alpha, c, N, w, K, &beta, o, N);
        }
    }
    if (bias) {
        int total = B * Co * Ho * Wo;
        add_bias_2d_kernel<<<(total+255)/256, 256, 0, s>>>(output, bias, B, Co, Ho, Wo);
    }
}

// ConvTranspose2d: input [B,Ci,Hi,Wi], weight [Ci,Co/g,kH,kW] -> [B,Co,Ho,Wo]
void launch_conv_transpose2d(const float* input, const float* weight, const float* bias,
    float* output, float* col_buf,
    int B, int Ci, int Hi, int Wi, int Co, int kH, int kW,
    int sH, int sW, int pH, int pW, int dH, int dW,
    int groups, int Ho, int Wo, cudaStream_t s) {
    
    cublasHandle_t h = get_handle();
    cublasSetStream(h, s);
    float alpha = 1.0f, beta = 0.0f;
    int Ci_g = Ci / groups, Co_g = Co / groups;
    int N_in = Hi * Wi;
    
    for (int b = 0; b < B; b++) {
        for (int g = 0; g < groups; g++) {
            const float* w = weight + g * Ci_g * Co_g * kH * kW;
            const float* inp = input + (b * Ci + g * Ci_g) * N_in;
            float* c = col_buf + (b * Co * kH * kW + g * Co_g * kH * kW) * N_in;
            int M = Co_g * kH * kW, K = Ci_g;
            cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_T, N_in, M, K,
                &alpha, inp, N_in, w, M, &beta, c, N_in);
        }
    }
    
    int out_total = B * Co * Ho * Wo;
    cudaMemsetAsync(output, 0, out_total * sizeof(float), s);
    col2im_2d_kernel<<<(out_total+255)/256, 256, 0, s>>>(
        col_buf, output, B, Co, Ho, Wo, kH, kW, sH, sW, pH, pW, dH, dW, Hi, Wi);
    
    if (bias) {
        add_bias_2d_kernel<<<(out_total+255)/256, 256, 0, s>>>(output, bias, B, Co, Ho, Wo);
    }
}

void launch_avgpool2d(const float* input, float* output,
    int B, int C, int H, int W, int kH, int kW, int sH, int sW,
    int pH, int pW, int Ho, int Wo, bool count_include_pad, cudaStream_t s) {
    int total = B * C * Ho * Wo;
    avgpool2d_kernel<<<(total+255)/256, 256, 0, s>>>(
        input, output, B, C, H, W, kH, kW, sH, sW, pH, pW, Ho, Wo, count_include_pad);
}

void launch_maxpool2d(const float* input, float* output, int* indices,
    int B, int C, int H, int W, int kH, int kW, int sH, int sW,
    int pH, int pW, int Ho, int Wo, cudaStream_t s) {
    int total = B * C * Ho * Wo;
    maxpool2d_kernel<<<(total+255)/256, 256, 0, s>>>(
        input, output, indices, B, C, H, W, kH, kW, sH, sW, pH, pW, Ho, Wo);
}

void launch_maxpool2d_backward(const float* grad_output, const int* indices,
    float* grad_input, int input_total, int out_total, cudaStream_t s) {
    cudaMemsetAsync(grad_input, 0, input_total * sizeof(float), s);
    maxpool2d_backward_kernel<<<(out_total+255)/256, 256, 0, s>>>(
        grad_output, indices, grad_input, out_total);
}

// Conv1d backward weight: im2col(input) then gemm with grad_output
// grad_output: [B, Co, Lo], input: [B, Ci, Li]
// grad_weight: [Co, Ci/g, K]
void launch_conv1d_backward_weight(
    const float* input, const float* grad_output, float* grad_weight, float* col_buf,
    int B, int C_in, int L_in, int C_out, int K,
    int stride, int padding, int dilation, int groups, int L_out, cudaStream_t s) {
    
    cublasHandle_t h = get_handle();
    cublasSetStream(h, s);
    
    int C_in_g = C_in / groups, C_out_g = C_out / groups;
    int wsize = C_out * C_in_g * K;
    cudaMemsetAsync(grad_weight, 0, wsize * sizeof(float), s);
    
    // im2col of input
    int col_total = B * C_in * K * L_out;
    im2col_1d_kernel<<<(col_total+255)/256, 256, 0, s>>>(
        input, col_buf, B, C_in, L_in, K, stride, padding, dilation, L_out);
    
    float alpha = 1.0f, beta = 1.0f; // accumulate across batch
    for (int b = 0; b < B; b++) {
        for (int g = 0; g < groups; g++) {
            // grad_weight_g += grad_output_g @ col_g^T
            // grad_output_g: [C_out_g, L_out], col_g: [C_in_g*K, L_out]
            const float* go = grad_output + (b * C_out + g * C_out_g) * L_out;
            const float* c = col_buf + (b * C_in * K + g * C_in_g * K) * L_out;
            float* gw = grad_weight + g * C_out_g * C_in_g * K;
            int M = C_out_g, N = C_in_g * K, KK = L_out;
            // gw[M,N] += go[M,KK] * col[N,KK]^T
            // cublas col-major: C = alpha * B^T * A + beta * C
            cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, KK,
                &alpha, c, KK, go, KK, &beta, gw, N);
        }
    }
}

// Conv2d backward weight
void launch_conv2d_backward_weight(
    const float* input, const float* grad_output, float* grad_weight, float* col_buf,
    int B, int Ci, int H, int W, int Co, int kH, int kW,
    int sH, int sW, int pH, int pW, int dH, int dW,
    int groups, int Ho, int Wo, cudaStream_t s) {
    
    cublasHandle_t h = get_handle();
    cublasSetStream(h, s);
    
    int Ci_g = Ci / groups, Co_g = Co / groups;
    int wsize = Co * Ci_g * kH * kW;
    cudaMemsetAsync(grad_weight, 0, wsize * sizeof(float), s);
    
    int col_total = B * Ci * kH * kW * Ho * Wo;
    im2col_2d_kernel<<<(col_total+255)/256, 256, 0, s>>>(
        input, col_buf, B, Ci, H, W, kH, kW, sH, sW, pH, pW, dH, dW, Ho, Wo);
    
    float alpha = 1.0f, beta = 1.0f;
    int N_spatial = Ho * Wo;
    for (int b = 0; b < B; b++) {
        for (int g = 0; g < groups; g++) {
            const float* go = grad_output + (b * Co + g * Co_g) * N_spatial;
            const float* c = col_buf + (b * Ci * kH * kW + g * Ci_g * kH * kW) * N_spatial;
            float* gw = grad_weight + g * Co_g * Ci_g * kH * kW;
            int M = Co_g, N = Ci_g * kH * kW, KK = N_spatial;
            cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, KK,
                &alpha, c, KK, go, KK, &beta, gw, N);
        }
    }
}

// ConvTranspose1d backward weight:
// Forward: col = w^T @ x, output = col2im(col)
// Backward: grad_col = im2col(grad_output), dw = x @ grad_col^T
// weight shape: [C_in, C_out/g, K], grad_output: [B, C_out, L_out], input: [B, C_in, L_in]
void launch_conv_transpose1d_backward_weight(
    const float* input, const float* grad_output, float* grad_weight, float* col_buf,
    int B, int C_in, int L_in, int C_out, int K,
    int stride, int padding, int dilation, int groups, int L_out, cudaStream_t s) {
    
    cublasHandle_t h = get_handle();
    cublasSetStream(h, s);
    
    int C_in_g = C_in / groups, C_out_g = C_out / groups;
    int wsize = C_in * C_out_g * K;
    cudaMemsetAsync(grad_weight, 0, wsize * sizeof(float), s);
    
    // im2col of grad_output: [B, C_out, L_out] → col [B, C_out*K, L_in]
    int col_total = B * C_out * K * L_in;
    im2col_1d_kernel<<<(col_total+255)/256, 256, 0, s>>>(
        grad_output, col_buf, B, C_out, L_out, K, stride, padding, dilation, L_in);
    
    float alpha = 1.0f, beta = 1.0f;
    for (int b = 0; b < B; b++) {
        for (int g = 0; g < groups; g++) {
            // dw_g = x_g @ grad_col_g^T
            // x_g: [C_in_g, L_in], grad_col_g: [C_out_g*K, L_in]
            // result: [C_in_g, C_out_g*K]
            const float* x = input + (b * C_in + g * C_in_g) * L_in;
            const float* gc = col_buf + (b * C_out * K + g * C_out_g * K) * L_in;
            float* gw = grad_weight + g * C_in_g * C_out_g * K;
            int M = C_in_g, N = C_out_g * K, KK = L_in;
            // gw[M,N] += x[M,KK] @ gc[N,KK]^T
            cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, KK,
                &alpha, gc, KK, x, KK, &beta, gw, N);
        }
    }
}

// ConvTranspose2d backward weight
void launch_conv_transpose2d_backward_weight(
    const float* input, const float* grad_output, float* grad_weight, float* col_buf,
    int B, int Ci, int Hi, int Wi, int Co, int kH, int kW,
    int sH, int sW, int pH, int pW, int dH, int dW,
    int groups, int Ho, int Wo, cudaStream_t s) {
    
    cublasHandle_t h = get_handle();
    cublasSetStream(h, s);
    
    int Ci_g = Ci / groups, Co_g = Co / groups;
    int wsize = Ci * Co_g * kH * kW;
    cudaMemsetAsync(grad_weight, 0, wsize * sizeof(float), s);
    
    // im2col of grad_output: [B, Co, Ho, Wo] → col [B, Co*kH*kW, Hi*Wi]
    int col_total = B * Co * kH * kW * Hi * Wi;
    im2col_2d_kernel<<<(col_total+255)/256, 256, 0, s>>>(
        grad_output, col_buf, B, Co, Ho, Wo, kH, kW, sH, sW, pH, pW, dH, dW, Hi, Wi);
    
    float alpha = 1.0f, beta = 1.0f;
    int N_in = Hi * Wi;
    for (int b = 0; b < B; b++) {
        for (int g = 0; g < groups; g++) {
            const float* x = input + (b * Ci + g * Ci_g) * N_in;
            const float* gc = col_buf + (b * Co * kH * kW + g * Co_g * kH * kW) * N_in;
            float* gw = grad_weight + g * Ci_g * Co_g * kH * kW;
            int M = Ci_g, N = Co_g * kH * kW, KK = N_in;
            cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, KK,
                &alpha, gc, KK, x, KK, &beta, gw, N);
        }
    }
}

void launch_conv2d_cudnn(const float* input, const float* weight, const float* bias,
    float* output,
    int B, int Ci, int H, int W, int Co, int kH, int kW,
    int sH, int sW, int pH, int pW, int dH, int dW,
    int groups, int Ho, int Wo, cudaStream_t s) {
    
    cudnnHandle_t handle = get_cudnn_handle();
    cudnnSetStream(handle, s);
    
    cudnnTensorDescriptor_t xDesc, yDesc, bDesc;
    cudnnFilterDescriptor_t wDesc;
    cudnnConvolutionDescriptor_t convDesc;
    
    cudnnCreateTensorDescriptor(&xDesc);
    cudnnCreateTensorDescriptor(&yDesc);
    cudnnCreateFilterDescriptor(&wDesc);
    cudnnCreateConvolutionDescriptor(&convDesc);
    
    cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, B, Ci, H, W);
    cudnnSetFilter4dDescriptor(wDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, Co, Ci/groups, kH, kW);
    cudnnSetConvolution2dDescriptor(convDesc, pH, pW, sH, sW, dH, dW, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
    cudnnSetConvolutionGroupCount(convDesc, groups);
    cudnnSetConvolutionMathType(convDesc, CUDNN_TENSOR_OP_MATH);
    cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, B, Co, Ho, Wo);
    
    static std::map<ConvParams, ConvPlan> plan_cache;
    ConvParams params{B,Ci,H,W,Co,kH,kW,sH,sW,pH,pW,dH,dW,groups};
    auto it = plan_cache.find(params);
    if (it == plan_cache.end()) {
        ConvPlan p;
        int count;
        cudnnConvolutionFwdAlgoPerf_t perf;
        cudnnGetConvolutionForwardAlgorithm_v7(handle, xDesc, wDesc, convDesc, yDesc, 1, &count, &perf);
        p.algo = perf.algo; p.ws_size = 0;
        cudnnGetConvolutionForwardWorkspaceSize(handle, xDesc, wDesc, convDesc, yDesc, p.algo, &p.ws_size);
        it = plan_cache.emplace(params, p).first;
    }
    const auto& plan = it->second;
    
    float alpha = 1.0f, beta = 0.0f;
    cudnnConvolutionForward(handle, &alpha, xDesc, input, wDesc, weight, convDesc, plan.algo, get_workspace(plan.ws_size), plan.ws_size, &beta, yDesc, output);
    
    if (bias) {
        cudnnCreateTensorDescriptor(&bDesc);
        cudnnSetTensor4dDescriptor(bDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, Co, 1, 1);
        float one = 1.0f;
        cudnnAddTensor(handle, &one, bDesc, bias, &one, yDesc, output);
        cudnnDestroyTensorDescriptor(bDesc);
    }
    
    cudnnDestroyTensorDescriptor(xDesc);
    cudnnDestroyTensorDescriptor(yDesc);
    cudnnDestroyFilterDescriptor(wDesc);
    cudnnDestroyConvolutionDescriptor(convDesc);
}

// ==================== cuDNN backward data + filter ====================

struct ConvBwdPlan {
    cudnnConvolutionBwdDataAlgo_t data_algo;
    size_t data_ws_size;
    cudnnConvolutionBwdFilterAlgo_t filter_algo;
    size_t filter_ws_size;
};

void launch_conv2d_backward_cudnn(
    const float* input, const float* weight, const float* grad_output,
    float* grad_input, float* grad_weight, float* grad_bias,
    int B, int Ci, int H, int W, int Co, int kH, int kW,
    int sH, int sW, int pH, int pW, int dH, int dW,
    int groups, int Ho, int Wo, cudaStream_t s) {
    
    cudnnHandle_t handle = get_cudnn_handle();
    cudnnSetStream(handle, s);
    
    cudnnTensorDescriptor_t xDesc, dyDesc;
    cudnnFilterDescriptor_t wDesc;
    cudnnConvolutionDescriptor_t convDesc;
    
    cudnnCreateTensorDescriptor(&xDesc);
    cudnnCreateTensorDescriptor(&dyDesc);
    cudnnCreateFilterDescriptor(&wDesc);
    cudnnCreateConvolutionDescriptor(&convDesc);
    
    cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, B, Ci, H, W);
    cudnnSetFilter4dDescriptor(wDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, Co, Ci/groups, kH, kW);
    cudnnSetConvolution2dDescriptor(convDesc, pH, pW, sH, sW, dH, dW, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
    cudnnSetConvolutionGroupCount(convDesc, groups);
    cudnnSetConvolutionMathType(convDesc, CUDNN_TENSOR_OP_MATH);
    cudnnSetTensor4dDescriptor(dyDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, B, Co, Ho, Wo);
    
    static std::map<ConvParams, ConvBwdPlan> bwd_cache;
    ConvParams params{B,Ci,H,W,Co,kH,kW,sH,sW,pH,pW,dH,dW,groups};
    auto it = bwd_cache.find(params);
    if (it == bwd_cache.end()) {
        ConvBwdPlan p{};
        int count;
        cudnnConvolutionBwdDataAlgoPerf_t dperf;
        cudnnGetConvolutionBackwardDataAlgorithm_v7(handle, wDesc, dyDesc, convDesc, xDesc, 1, &count, &dperf);
        p.data_algo = dperf.algo; p.data_ws_size = 0;
        cudnnGetConvolutionBackwardDataWorkspaceSize(handle, wDesc, dyDesc, convDesc, xDesc, p.data_algo, &p.data_ws_size);
        cudnnConvolutionBwdFilterAlgoPerf_t fperf;
        cudnnGetConvolutionBackwardFilterAlgorithm_v7(handle, xDesc, dyDesc, convDesc, wDesc, 1, &count, &fperf);
        p.filter_algo = fperf.algo; p.filter_ws_size = 0;
        cudnnGetConvolutionBackwardFilterWorkspaceSize(handle, xDesc, dyDesc, convDesc, wDesc, p.filter_algo, &p.filter_ws_size);
        it = bwd_cache.emplace(params, p).first;
    }
    const auto& plan = it->second;
    float alpha = 1.0f, beta = 0.0f;
    
    if (grad_input)
        cudnnConvolutionBackwardData(handle, &alpha, wDesc, weight, dyDesc, grad_output,
            convDesc, plan.data_algo, get_workspace(plan.data_ws_size), plan.data_ws_size, &beta, xDesc, grad_input);
    if (grad_weight)
        cudnnConvolutionBackwardFilter(handle, &alpha, xDesc, input, dyDesc, grad_output,
            convDesc, plan.filter_algo, get_workspace(plan.filter_ws_size), plan.filter_ws_size, &beta, wDesc, grad_weight);
    if (grad_bias) {
        cudnnTensorDescriptor_t dbDesc;
        cudnnCreateTensorDescriptor(&dbDesc);
        cudnnSetTensor4dDescriptor(dbDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, Co, 1, 1);
        cudnnConvolutionBackwardBias(handle, &alpha, dyDesc, grad_output, &beta, dbDesc, grad_bias);
        cudnnDestroyTensorDescriptor(dbDesc);
    }
    
    cudnnDestroyTensorDescriptor(xDesc);
    cudnnDestroyTensorDescriptor(dyDesc);
    cudnnDestroyFilterDescriptor(wDesc);
    cudnnDestroyConvolutionDescriptor(convDesc);
}

}

}
}
