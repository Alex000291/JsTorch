#pragma once
#include "dtype.hpp"
#include "shape.hpp"
#include <memory>
#include <string>
#include <cuda_runtime.h>

namespace jstorch {

class Tensor {
private:
    std::shared_ptr<void> data_;
    DType dtype_;
    Shape shape_;
    Strides strides_;
    cudaStream_t stream_;
    bool owns_stream_;

public:
    // Constructors
    Tensor() : dtype_(DType::Float32), stream_(0), owns_stream_(false) {}
    explicit Tensor(const Shape& shape, DType dtype = DType::Float32);
    Tensor(std::shared_ptr<void> data, const Shape& shape, const Strides& strides,
           DType dtype, cudaStream_t stream);
    ~Tensor();
    
    // Getters
    const Shape& shape() const { return shape_; }
    const Strides& strides() const { return strides_; }
    DType dtype() const { return dtype_; }
    int ndim() const { return (int)shape_.size(); }
    int size() const { return total_size(shape_); }
    cudaStream_t stream() const { return stream_; }
    
    template<typename T> T* data() { return static_cast<T*>(data_.get()); }
    template<typename T> const T* data() const { return static_cast<const T*>(data_.get()); }
    
    bool is_contiguous() const { return jstorch::is_contiguous(shape_, strides_); }
    
    // Copy
    Tensor clone() const;
    Tensor contiguous() const;
    
    // View ops
    Tensor view(const Shape& new_shape) const;
    Tensor reshape(const Shape& new_shape) const;
    Tensor squeeze(int axis = -1) const;
    Tensor unsqueeze(int axis) const;
    Tensor transpose() const;
    Tensor transpose(int dim0, int dim1) const;
    Tensor slice(int dim, int start, int end) const;
    Tensor flatten(int start_dim, int end_dim) const;
    
    // Binary ops (broadcast)
    Tensor add(const Tensor& other) const;
    Tensor sub(const Tensor& other) const;
    Tensor mul(const Tensor& other) const;
    Tensor div(const Tensor& other) const;
    Tensor maximum(const Tensor& other) const;
    Tensor minimum(const Tensor& other) const;
    Tensor pow(const Tensor& other) const;
    Tensor gt(const Tensor& other) const;
    Tensor lt(const Tensor& other) const;
    Tensor ge(const Tensor& other) const;
    Tensor le(const Tensor& other) const;
    Tensor eq(const Tensor& other) const;
    Tensor ne(const Tensor& other) const;
    
    // Unary ops
    Tensor abs() const;
    Tensor sqrt() const;
    Tensor square() const;
    Tensor exp() const;
    Tensor log() const;
    Tensor log1p() const;
    Tensor sin() const;
    Tensor cos() const;
    Tensor neg() const;
    Tensor floor() const;
    Tensor ceil() const;
    Tensor round() const;
    Tensor sigmoid() const;
    Tensor tanh() const;
    Tensor relu() const;
    Tensor silu() const;
    Tensor gelu() const;
    Tensor softplus() const;
    Tensor reciprocal() const;
    Tensor sign() const;
    Tensor leaky_relu(float slope = 0.01f) const;
    Tensor clamp(float lo, float hi) const;
    Tensor clamp_min(float lo) const;
    Tensor clamp_max(float hi) const;
    Tensor fmod(float d) const;
    Tensor pow_scalar(float e) const;
    Tensor mul_scalar(float s) const;
    Tensor add_scalar(float s) const;
    
    // Reduce
    Tensor sum(int dim = -1, bool keepdim = false) const;
    Tensor mean(int dim = -1, bool keepdim = false) const;
    Tensor max(int dim, bool keepdim = false) const;
    Tensor min(int dim, bool keepdim = false) const;
    Tensor argmax(int dim) const;
    Tensor argmin(int dim) const;
    
    // Tensor ops
    static Tensor cat(const std::vector<Tensor>& tensors, int dim);
    Tensor flip(int dim) const;
    Tensor pad(const std::vector<int>& padding, int mode = 0, float value = 0.0f) const;
    Tensor cumsum(int dim) const;
    static Tensor where(const Tensor& cond, const Tensor& x, const Tensor& y);
    
    Tensor matmul(const Tensor& other) const;
    Tensor embedding(const Tensor& indices) const;
    
    // Conv1d
    Tensor conv1d(const Tensor& weight, const Tensor* bias,
                  int stride, int padding, int dilation, int groups) const;
    Tensor conv_transpose1d(const Tensor& weight, const Tensor* bias,
                            int stride, int padding, int output_padding, int dilation, int groups) const;
    
    // Conv2d
    Tensor conv2d(const Tensor& weight, const Tensor* bias,
                  int sH, int sW, int pH, int pW, int dH, int dW, int groups) const;
    Tensor conv_transpose2d(const Tensor& weight, const Tensor* bias,
                            int sH, int sW, int pH, int pW, int opH, int opW,
                            int dH, int dW, int groups) const;
    Tensor avg_pool2d(int kH, int kW, int sH, int sW, int pH, int pW, bool count_include_pad = true) const;
    std::pair<Tensor, Tensor> max_pool2d(int kH, int kW, int sH, int sW, int pH, int pW) const;
    static Tensor maxpool2d_backward(const Tensor& grad, const Tensor& indices, int B, int C, int H, int W);
    static void conv2d_backward_cudnn(const Tensor& input, const Tensor& weight, const Tensor& grad_output,
                                       Tensor& grad_input, Tensor& grad_weight, Tensor* grad_bias,
                                       int sH, int sW, int pH, int pW, int dH, int dW, int groups);

    // cuDNN RNN forward/backward
    // mode: 0=RNN_RELU, 1=RNN_TANH, 2=LSTM, 3=GRU
    // x: [B, seq_len, input_size], hx: [num_layers*num_dir, B, hidden], cx: same (LSTM only)
    // Returns: {y, hy} or {y, hy, cy} for LSTM
    static std::vector<Tensor> rnn_forward(const Tensor& x, const Tensor* hx, const Tensor* cx,
                                            const std::vector<Tensor>& weights_ih, const std::vector<Tensor>& weights_hh,
                                            const std::vector<Tensor>& biases_ih, const std::vector<Tensor>& biases_hh,
                                            int mode, int hidden_size, int num_layers, int bidirectional);

    // Returns: {dx, dhx, [dcx], dw_ih[0], dw_hh[0], db_ih[0], db_hh[0], ...}
    static std::vector<Tensor> rnn_backward(const Tensor& x, const Tensor* hx, const Tensor* cx,
                                             const Tensor& y, const Tensor& dy, const Tensor* dhy, const Tensor* dcy,
                                             const std::vector<Tensor>& weights_ih, const std::vector<Tensor>& weights_hh,
                                             const std::vector<Tensor>& biases_ih, const std::vector<Tensor>& biases_hh,
                                             int mode, int hidden_size, int num_layers, int bidirectional);

    Tensor interpolate(int target_size, int mode, bool align_corners = false) const;
    
    // Backward ops
    Tensor scatter_add(const Tensor& grad, const Tensor& indices, int vocab_size) const;
    Tensor interp1d_backward(int in_len, int mode, bool align_corners) const;
    Tensor avgpool2d_backward(int H, int W, int kH, int kW, int sH, int sW, int pH, int pW, bool count_include_pad) const;
    static Tensor conv1d_backward_weight(const Tensor& input, const Tensor& grad_output,
        int C_in_g, int K, int stride, int padding, int dilation, int groups);
    static Tensor conv2d_backward_weight(const Tensor& input, const Tensor& grad_output,
        int Ci_g, int kH, int kW, int sH, int sW, int pH, int pW, int dH, int dW, int groups);
    static Tensor conv_transpose1d_backward_weight(const Tensor& input, const Tensor& grad_output,
        int C_out_g, int K, int stride, int padding, int dilation, int groups);
    static Tensor conv_transpose2d_backward_weight(const Tensor& input, const Tensor& grad_output,
        int Co_g, int kH, int kW, int sH, int sW, int pH, int pW, int dH, int dW, int groups);
    
    // Factory
    static Tensor from_array(const float* data, const Shape& shape);
    static Tensor from_buffer(const float* data, int count, const Shape& shape);
    static Tensor randn(const Shape& shape);
    static Tensor from_int_array(const int* data, const Shape& shape);
    static Tensor full(const Shape& shape, float value);
    static Tensor arange(float start, float end, float step = 1.0f);
    Tensor randn_like() const;
    std::vector<float> to_array() const;
    
    // Fused Adam optimizer step
    static void adam_step(Tensor& param, const Tensor& grad, Tensor& m, Tensor& v,
        float lr, float beta1, float beta2, float eps, float bc1, float bc2, float weight_decay);

    // Release all cached GPU memory back to CUDA.
    // Call between benchmark batch sizes to prevent VRAM exhaustion.
    static void clear_cache();
};

}
