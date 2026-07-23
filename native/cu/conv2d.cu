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
    int max_blocks = 65535;
    int blocks = min((total_output + threads - 1) / threads, max_blocks);
    
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

// ==================== Conv2D Backward ====================

// Backward pass for input
__global__ void conv2d_backward_input_kernel(
    const float* grad_output,
    const float* weight,
    float* grad_input,
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
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_input = batch * in_channels * input_h * input_w;
    
    if (idx >= total_input) return;
    
    // Decode input position
    int w_in = idx % input_w;
    int h_in = (idx / input_w) % input_h;
    int c_in = (idx / (input_w * input_h)) % in_channels;
    int b = idx / (input_w * input_h * in_channels);
    
    float sum = 0.0f;
    
    // For each output channel
    for (int c_out = 0; c_out < out_channels; c_out++) {
        // For each position in output that this input contributes to
        for (int kh = 0; kh < kernel_h; kh++) {
            for (int kw = 0; kw < kernel_w; kw++) {
                // Calculate output position
                int h_out_temp = h_in + padding_h - kh;
                int w_out_temp = w_in + padding_w - kw;
                
                // Check if this maps to valid output position
                if (h_out_temp % stride_h == 0 && w_out_temp % stride_w == 0) {
                    int h_out = h_out_temp / stride_h;
                    int w_out = w_out_temp / stride_w;
                    
                    if (h_out >= 0 && h_out < output_h && w_out >= 0 && w_out < output_w) {
                        int grad_output_idx = ((b * out_channels + c_out) * output_h + h_out) * output_w + w_out;
                        int weight_idx = ((c_out * in_channels + c_in) * kernel_h + kh) * kernel_w + kw;
                        
                        sum += grad_output[grad_output_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    
    grad_input[idx] = sum;
}

// Backward pass for weight
__global__ void conv2d_backward_weight_kernel(
    const float* input,
    const float* grad_output,
    float* grad_weight,
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
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_weight = out_channels * in_channels * kernel_h * kernel_w;
    
    if (idx >= total_weight) return;
    
    // Decode weight position
    int kw = idx % kernel_w;
    int kh = (idx / kernel_w) % kernel_h;
    int c_in = (idx / (kernel_w * kernel_h)) % in_channels;
    int c_out = idx / (kernel_w * kernel_h * in_channels);
    
    float sum = 0.0f;
    
    // Sum over batch and output positions
    for (int b = 0; b < batch; b++) {
        for (int h_out = 0; h_out < output_h; h_out++) {
            for (int w_out = 0; w_out < output_w; w_out++) {
                int h_in = h_out * stride_h + kh - padding_h;
                int w_in = w_out * stride_w + kw - padding_w;
                
                if (h_in >= 0 && h_in < input_h && w_in >= 0 && w_in < input_w) {
                    int input_idx = ((b * in_channels + c_in) * input_h + h_in) * input_w + w_in;
                    int grad_output_idx = ((b * out_channels + c_out) * output_h + h_out) * output_w + w_out;
                    
                    sum += input[input_idx] * grad_output[grad_output_idx];
                }
            }
        }
    }
    
    grad_weight[idx] = sum;
}

// Backward pass for bias
__global__ void conv2d_backward_bias_kernel(
    const float* grad_output,
    float* grad_bias,
    int batch,
    int out_channels,
    int output_h,
    int output_w
) {
    int c_out = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (c_out >= out_channels) return;
    
    float sum = 0.0f;
    
    // Sum over batch and spatial dimensions
    for (int b = 0; b < batch; b++) {
        for (int h = 0; h < output_h; h++) {
            for (int w = 0; w < output_w; w++) {
                int idx = ((b * out_channels + c_out) * output_h + h) * output_w + w;
                sum += grad_output[idx];
            }
        }
    }
    
    grad_bias[c_out] = sum;
}

extern "C" void launch_conv2d_backward_input(
    const float* grad_output,
    const float* weight,
    float* grad_input,
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
    int total_input = batch * in_channels * input_h * input_w;
    int threads = 256;
    int max_blocks = 65535;
    int blocks = min((total_input + threads - 1) / threads, max_blocks);
    
    conv2d_backward_input_kernel<<<blocks, threads, 0, stream>>>(
        grad_output, weight, grad_input,
        batch, in_channels, out_channels,
        input_h, input_w, kernel_h, kernel_w,
        stride_h, stride_w, padding_h, padding_w,
        output_h, output_w
    );
}

extern "C" void launch_conv2d_backward_weight(
    const float* input,
    const float* grad_output,
    float* grad_weight,
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
    int total_weight = out_channels * in_channels * kernel_h * kernel_w;
    int threads = 256;
    int blocks = (total_weight + threads - 1) / threads;
    
    conv2d_backward_weight_kernel<<<blocks, threads, 0, stream>>>(
        input, grad_output, grad_weight,
        batch, in_channels, out_channels,
        input_h, input_w, kernel_h, kernel_w,
        stride_h, stride_w, padding_h, padding_w,
        output_h, output_w
    );
}

extern "C" void launch_conv2d_backward_bias(
    const float* grad_output,
    float* grad_bias,
    int batch,
    int out_channels,
    int output_h,
    int output_w,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (out_channels + threads - 1) / threads;
    
    conv2d_backward_bias_kernel<<<blocks, threads, 0, stream>>>(
        grad_output, grad_bias,
        batch, out_channels, output_h, output_w
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
