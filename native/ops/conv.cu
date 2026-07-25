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
    
    // Cache: algo + workspace (persistent allocation)
    struct CachedPlan {
        cudnnConvolutionFwdAlgo_t algo;
        void* ws; size_t ws_size;
    };
    using CacheKey = std::tuple<int,int,int,int,int,int,int,int,int,int,int,int,int,int>;
    static std::map<CacheKey, CachedPlan> plan_cache;
    
    auto key = std::make_tuple(B,Ci,H,W,Co,kH,kW,sH,sW,pH,pW,dH,dW,groups);
    auto it = plan_cache.find(key);
    CachedPlan plan;
    if (it != plan_cache.end()) {
        plan = it->second;
    } else {
        int returnedCount;
        cudnnConvolutionFwdAlgoPerf_t perfResults;
        cudnnGetConvolutionForwardAlgorithm_v7(handle, xDesc, wDesc, convDesc, yDesc, 1, &returnedCount, &perfResults);
        plan.algo = perfResults.algo;
        plan.ws = nullptr; plan.ws_size = 0;
        cudnnGetConvolutionForwardWorkspaceSize(handle, xDesc, wDesc, convDesc, yDesc, plan.algo, &plan.ws_size);
        if (plan.ws_size > 0) cudaMalloc(&plan.ws, plan.ws_size);
        plan_cache[key] = plan;
    }
    
    float alpha = 1.0f, beta = 0.0f;
    cudnnConvolutionForward(handle, &alpha, xDesc, input, wDesc, weight, convDesc, plan.algo, plan.ws, plan.ws_size, &beta, yDesc, output);
    
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

}

}
}
