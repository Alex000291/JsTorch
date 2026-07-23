// conv2d.cu - 2D Convolution kernel
#include <cuda_runtime.h>

// Input: [batch, in_channels, height, width]
// Weight: [out_channels, in_channels, kernel_h, kernel_w]
// Output: [batch, out_channels, out_h, out_w]

__global__ void conv2d_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch,
    int in_channels,
    int out_channels,
    int input_h,
    int input_w,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int output_h,
    int output_w
) {
    // Each thread computes one output element
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    int total_output = batch * out_channels * output_h * output_w;
    if (idx >= total_output) return;
    
    // Decode output position
    int w_out = idx % output_w;
    int h_out = (idx / output_w) % output_h;
    int c_out = (idx / (output_w * output_h)) % out_channels;
    int b = idx / (output_w * output_h * out_channels);
    
    float sum = 0.0f;
    
    // Convolve over input channels and kernel
    for (int c_in = 0; c_in < in_channels; c_in++) {
        for (int kh = 0; kh < kernel_h; kh++) {
            for (int kw = 0; kw < kernel_w; kw++) {
                int h_in = h_out * stride_h + kh - padding_h;
                int w_in = w_out * stride_w + kw - padding_w;
                
                // Check bounds (zero padding)
                if (h_in >= 0 && h_in < input_h && w_in >= 0 && w_in < input_w) {
                    int input_idx = ((b * in_channels + c_in) * input_h + h_in) * input_w + w_in;
                    int weight_idx = ((c_out * in_channels + c_in) * kernel_h + kh) * kernel_w + kw;
                    
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    // Add bias if provided
    if (bias != nullptr) {
        sum += bias[c_out];
    }
    
    output[idx] = sum;
}

extern "C" void launch_conv2d(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch,
    int in_channels,
    int out_channels,
    int input_h,
    int input_w,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int output_h,
    int output_w,
    cudaStream_t stream
) {
    int total_output = batch * out_channels * output_h * output_w;
    int threads = 256;
    int blocks = (total_output + threads - 1) / threads;
    
    conv2d_kernel<<<blocks, threads, 0, stream>>>(
        input, weight, bias, output,
        batch, in_channels, out_channels,
        input_h, input_w,
        kernel_h, kernel_w,
        stride_h, stride_w,
        padding_h, padding_w,
        output_h, output_w
    );
}

// ==================== MaxPool2D ====================

__global__ void maxpool2d_kernel(
    const float* input,
    float* output,
    int batch,
    int channels,
    int input_h,
    int input_w,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int output_h,
    int output_w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    int total_output = batch * channels * output_h * output_w;
    if (idx >= total_output) return;
    
    // Decode output position
    int w_out = idx % output_w;
    int h_out = (idx / output_w) % output_h;
    int c = (idx / (output_w * output_h)) % channels;
    int b = idx / (output_w * output_h * channels);
    
    float max_val = -INFINITY;
    
    for (int kh = 0; kh < kernel_h; kh++) {
        for (int kw = 0; kw < kernel_w; kw++) {
            int h_in = h_out * stride_h + kh - padding_h;
            int w_in = w_out * stride_w + kw - padding_w;
            
            if (h_in >= 0 && h_in < input_h && w_in >= 0 && w_in < input_w) {
                int input_idx = ((b * channels + c) * input_h + h_in) * input_w + w_in;
                max_val = fmaxf(max_val, input[input_idx]);
            }
        }
    }
    
    output[idx] = max_val;
}

extern "C" void launch_maxpool2d(
    const float* input,
    float* output,
    int batch,
    int channels,
    int input_h,
    int input_w,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int output_h,
    int output_w,
    cudaStream_t stream
) {
    int total_output = batch * channels * output_h * output_w;
    int threads = 256;
    int blocks = (total_output + threads - 1) / threads;
    
    maxpool2d_kernel<<<blocks, threads, 0, stream>>>(
        input, output,
        batch, channels,
        input_h, input_w,
        kernel_h, kernel_w,
        stride_h, stride_w,
        padding_h, padding_w,
        output_h, output_w
    );
}
