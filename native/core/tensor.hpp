#pragma once
#include "dtype.hpp"
#include "shape.hpp"
#include <memory>
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
    // 构造
    explicit Tensor(const Shape& shape, DType dtype = DType::Float32);
    
    // 从现有数据构造(view)
    Tensor(std::shared_ptr<void> data, const Shape& shape, const Strides& strides,
           DType dtype, cudaStream_t stream);
    
    ~Tensor();
    
    // Getters
    const Shape& shape() const { return shape_; }
    const Strides& strides() const { return strides_; }
    DType dtype() const { return dtype_; }
    int ndim() const { return shape_.size(); }
    int size() const { return total_size(shape_); }
    cudaStream_t stream() const { return stream_; }
    
    template<typename T>
    T* data() { return static_cast<T*>(data_.get()); }
    
    template<typename T>
    const T* data() const { return static_cast<const T*>(data_.get()); }
    
    bool is_contiguous() const { return jstorch::is_contiguous(shape_, strides_); }
    bool is_complex() const { return dtype_ == DType::Complex64; }
    
    // 深拷贝
    Tensor clone() const;
    Tensor contiguous() const;
    
    // View操作
    Tensor view(const Shape& new_shape) const;
    Tensor reshape(const Shape& new_shape) const;
    Tensor squeeze(int axis = -1) const;
    Tensor unsqueeze(int axis) const;
    Tensor transpose() const;  // 2D only
    
    // 复数
    Tensor real() const;
    Tensor imag() const;
    static Tensor from_real_imag(const Tensor& real, const Tensor& imag);
    
    // 二元操作(自动broadcast)
    Tensor add(const Tensor& other) const;
    Tensor sub(const Tensor& other) const;
    Tensor mul(const Tensor& other) const;
    Tensor div(const Tensor& other) const;
    
    // 一元操作(通过ops/unary.cu实现)
    Tensor abs() const;
    Tensor sqrt() const;
    Tensor square() const;
    Tensor exp() const;
    Tensor log() const;
    Tensor sin() const;
    Tensor cos() const;
    Tensor sigmoid() const;
    Tensor tanh() const;
    Tensor relu() const;
    
    // 归约
    Tensor sum(int dim = -1, bool keepdim = false) const;
    Tensor mean(int dim = -1, bool keepdim = false) const;
    
    // Host交互
    static Tensor from_array(const float* data, const Shape& shape);
    std::vector<float> to_array() const;
};

}
