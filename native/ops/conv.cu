#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace jstorch {
namespace ops {

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

}

}
}
