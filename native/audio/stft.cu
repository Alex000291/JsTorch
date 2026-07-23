#include <cufft.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace jstorch {
namespace audio {

__global__ void hann_window_kernel(float* window, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        window[i] = 0.5f * (1.0f - cosf(2.0f * float(M_PI) * i / (size - 1)));
    }
}

__global__ void stft_windowing_kernel(
    const float* input, const float* window, float* frames,
    int batch, int input_length, int n_fft, int hop_length, int win_length, int n_frames
) {
    int batch_idx = blockIdx.x;
    int frame_idx = blockIdx.y;
    int i = threadIdx.x;
    
    if (batch_idx >= batch || frame_idx >= n_frames) return;
    
    const float* x = input + batch_idx * input_length;
    int offset = frame_idx * hop_length;
    float* frame = frames + (batch_idx * n_frames + frame_idx) * n_fft;
    
    for (int idx = i; idx < n_fft; idx += blockDim.x) {
        if (idx < win_length && offset + idx < input_length) {
            frame[idx] = x[offset + idx] * window[idx];
        } else {
            frame[idx] = 0.0f;
        }
    }
}

__global__ void istft_overlap_add_kernel(
    const float* frames, const float* window, float* output,
    int batch, int output_length, int n_fft, int hop_length, int win_length, int n_frames
) {
    int batch_idx = blockIdx.x;
    int frame_idx = blockIdx.y;
    int i = threadIdx.x;
    
    if (batch_idx >= batch || frame_idx >= n_frames) return;
    
    const float* frame = frames + (batch_idx * n_frames + frame_idx) * n_fft;
    float* y = output + batch_idx * output_length;
    int offset = frame_idx * hop_length;
    
    for (int idx = i; idx < win_length; idx += blockDim.x) {
        if (offset + idx < output_length) {
            atomicAdd(&y[offset + idx], frame[idx] * window[idx] / n_fft);
        }
    }
}

extern "C" {

void* stft_create_window(int win_length, cudaStream_t stream) {
    float* d_window;
    cudaMalloc(&d_window, win_length * sizeof(float));
    int threads = 256;
    int blocks = (win_length + threads - 1) / threads;
    hann_window_kernel<<<blocks, threads, 0, stream>>>(d_window, win_length);
    cudaStreamSynchronize(stream);
    return d_window;
}

void stft_destroy_window(void* window) {
    cudaFree(window);
}

void stft_forward(
    const float* input, cuComplex* output, const float* window,
    int batch, int input_length, int n_fft, int hop_length, int win_length,
    cudaStream_t stream
) {
    int n_frames = (input_length - win_length) / hop_length + 1;
    int n_freq = n_fft / 2 + 1;
    
    float* d_frames;
    cudaMalloc(&d_frames, batch * n_frames * n_fft * sizeof(float));
    
    dim3 grid(batch, n_frames);
    stft_windowing_kernel<<<grid, 256, 0, stream>>>(
        input, window, d_frames, batch, input_length, n_fft, hop_length, win_length, n_frames
    );
    
    cufftHandle plan;
    int n[1] = {n_fft};
    cufftPlanMany(&plan, 1, n, nullptr, 1, n_fft, nullptr, 1, n_freq, CUFFT_R2C, batch * n_frames);
    cufftSetStream(plan, stream);
    cufftExecR2C(plan, d_frames, output);
    
    cufftDestroy(plan);
    cudaFree(d_frames);
}

void stft_inverse(
    const cuComplex* input, float* output, const float* window,
    int batch, int output_length, int n_fft, int hop_length, int win_length,
    cudaStream_t stream
) {
    int n_frames = (output_length - win_length) / hop_length + 1;
    int n_freq = n_fft / 2 + 1;
    
    cudaMemset(output, 0, batch * output_length * sizeof(float));
    
    float* d_frames;
    cudaMalloc(&d_frames, batch * n_frames * n_fft * sizeof(float));
    
    cufftHandle plan;
    int n[1] = {n_fft};
    cufftPlanMany(&plan, 1, n, nullptr, 1, n_freq, nullptr, 1, n_fft, CUFFT_C2R, batch * n_frames);
    cufftSetStream(plan, stream);
    cufftExecC2R(plan, const_cast<cuComplex*>(input), d_frames);
    
    dim3 grid(batch, n_frames);
    istft_overlap_add_kernel<<<grid, 256, 0, stream>>>(
        d_frames, window, output, batch, output_length, n_fft, hop_length, win_length, n_frames
    );
    
    cufftDestroy(plan);
    cudaFree(d_frames);
}

}

}
}
