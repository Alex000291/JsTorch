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
    bool is_complex() const { return dtype_ == DType::Complex64; }
    
    // Copy
    Tensor clone() const;
    Tensor contiguous() const;
    
    // View ops (#6, #7, #8)
    Tensor view(const Shape& new_shape) const;
    Tensor reshape(const Shape& new_shape) const;
    Tensor squeeze(int axis = -1) const;
    Tensor unsqueeze(int axis) const;
    Tensor transpose() const;                    // 2D only (backward compat)
    Tensor transpose(int dim0, int dim1) const;  // #6: generalized
    Tensor slice(int dim, int start, int end) const;  // #7
    std::vector<Tensor> split(int dim, int chunk_size) const;  // #8
    
    // Complex
    Tensor real() const;
    Tensor imag() const;
    static Tensor from_real_imag(const Tensor& real, const Tensor& imag);
    
    // Binary ops (broadcast) — #4, #5
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
    
    // Unary ops — #1, #2, #3
    Tensor abs() const;
    Tensor sqrt() const;
    Tensor square() const;
    Tensor exp() const;
    Tensor log() const;
    Tensor sin() const;
    Tensor cos() const;
    Tensor neg() const;         // #1
    Tensor floor() const;
    Tensor ceil() const;
    Tensor round() const;
    Tensor sigmoid() const;
    Tensor tanh() const;
    Tensor relu() const;
    Tensor silu() const;
    Tensor gelu() const;
    Tensor softplus() const;
    Tensor leaky_relu(float slope = 0.01f) const;  // #3
    Tensor clamp(float lo, float hi) const;         // #2
    
    // Reduce
    Tensor sum(int dim = -1, bool keepdim = false) const;
    Tensor mean(int dim = -1, bool keepdim = false) const;
    
    // Tensor ops — #9-#13, #16-#20
    static Tensor cat(const std::vector<Tensor>& tensors, int dim);  // #9
    Tensor flip(int dim) const;                                       // #10
    Tensor pad(const std::vector<int>& padding, int mode = 0, float value = 0.0f) const;  // #11
    Tensor cumsum(int dim) const;                                     // #12
    static Tensor where(const Tensor& cond, const Tensor& x, const Tensor& y);  // #13
    
    Tensor matmul(const Tensor& other) const;  // #16
    Tensor embedding(const Tensor& indices) const;  // #17 (self=weight, indices=int tensor)
    Tensor conv1d(const Tensor& weight, const Tensor* bias,
                  int stride, int padding, int dilation, int groups) const;  // #18
    Tensor conv_transpose1d(const Tensor& weight, const Tensor* bias,
                            int stride, int padding, int output_padding, int dilation, int groups) const;  // #19
    Tensor interpolate(int target_size, int mode, bool align_corners = false) const;  // #20
    
    // Factory — #14, #15
    static Tensor from_array(const float* data, const Shape& shape);
    static Tensor from_buffer(const float* data, int count, const Shape& shape);  // #14
    static Tensor randn(const Shape& shape);  // #15
    Tensor randn_like() const;                // #15
    std::vector<float> to_array() const;
    
    // Int tensor support (for embedding indices)
    static Tensor from_int_array(const int* data, const Shape& shape);
};

}
