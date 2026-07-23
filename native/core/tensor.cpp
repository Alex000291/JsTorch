#include "tensor.hpp"
#include <stdexcept>

namespace jstorch {

// 外部kernel声明
extern "C" {
    void launch_abs(const float*, float*, int, cudaStream_t);
    void launch_sqrt(const float*, float*, int, cudaStream_t);
    void launch_square(const float*, float*, int, cudaStream_t);
    void launch_exp(const float*, float*, int, cudaStream_t);
    void launch_log(const float*, float*, int, cudaStream_t);
    void launch_sin(const float*, float*, int, cudaStream_t);
    void launch_cos(const float*, float*, int, cudaStream_t);
    void launch_sigmoid(const float*, float*, int, cudaStream_t);
    void launch_tanh(const float*, float*, int, cudaStream_t);
    void launch_relu(const float*, float*, int, cudaStream_t);
    
    void launch_broadcast_add(const float*, const int*, const int*,
                             const float*, const int*, const int*,
                             float*, const int*, int, int, cudaStream_t);
    void launch_broadcast_sub(const float*, const int*, const int*,
                             const float*, const int*, const int*,
                             float*, const int*, int, int, cudaStream_t);
    void launch_broadcast_mul(const float*, const int*, const int*,
                             const float*, const int*, const int*,
                             float*, const int*, int, int, cudaStream_t);
    void launch_broadcast_div(const float*, const int*, const int*,
                             const float*, const int*, const int*,
                             float*, const int*, int, int, cudaStream_t);
    
    void launch_reduce_sum(const float*, float*, int, int, int, cudaStream_t);
    void launch_reduce_mean(const float*, float*, int, int, int, cudaStream_t);
}

// Helper: host vector -> device array
static int* to_device(const std::vector<int>& vec) {
    int* d_ptr;
    cudaMalloc(&d_ptr, vec.size() * sizeof(int));
    cudaMemcpy(d_ptr, vec.data(), vec.size() * sizeof(int), cudaMemcpyHostToDevice);
    return d_ptr;
}

// 构造 - 分配GPU内存，零初始化
Tensor::Tensor(const Shape& shape, DType dtype)
    : dtype_(dtype), shape_(shape), strides_(compute_strides(shape)),
      stream_(0), owns_stream_(false) {
    size_t bytes = size() * dtype_size(dtype_);
    void* ptr;
    cudaMalloc(&ptr, bytes);
    cudaMemset(ptr, 0, bytes);
    data_ = std::shared_ptr<void>(ptr, [](void* p) { cudaFree(p); });
}

// 共享数据构造（view用）
Tensor::Tensor(std::shared_ptr<void> data, const Shape& shape, const Strides& strides,
               DType dtype, cudaStream_t stream)
    : data_(data), dtype_(dtype), shape_(shape), strides_(strides), 
      stream_(0), owns_stream_(false) {}

Tensor::~Tensor() {
    if (owns_stream_ && stream_) {
        cudaStreamDestroy(stream_);
    }
}

// 深拷贝
Tensor Tensor::clone() const {
    Tensor result(shape_, dtype_);
    cudaMemcpy(result.data_.get(), data_.get(), size() * dtype_size(dtype_), cudaMemcpyDeviceToDevice);
    return result;
}

// Contiguous
Tensor Tensor::contiguous() const {
    if (is_contiguous()) return *this;
    return clone();
}

// View / Reshape
Tensor Tensor::view(const Shape& new_shape) const {
    if (total_size(new_shape) != size()) throw std::runtime_error("View size mismatch");
    if (!is_contiguous()) throw std::runtime_error("View requires contiguous");
    return Tensor(data_, new_shape, compute_strides(new_shape), dtype_, stream_);
}

Tensor Tensor::reshape(const Shape& new_shape) const {
    return is_contiguous() ? view(new_shape) : contiguous().view(new_shape);
}

Tensor Tensor::squeeze(int axis) const {
    axis = axis == -1 ? -1 : normalize_axis(axis, ndim());
    Shape new_shape;
    if (axis == -1) {
        for (int d : shape_) if (d != 1) new_shape.push_back(d);
    } else {
        if (shape_[axis] != 1) throw std::runtime_error("Can only squeeze size-1");
        for (int i = 0; i < ndim(); i++) if (i != axis) new_shape.push_back(shape_[i]);
    }
    return view(new_shape);
}

Tensor Tensor::unsqueeze(int axis) const {
    axis = normalize_axis(axis, ndim() + 1);
    Shape new_shape = shape_;
    new_shape.insert(new_shape.begin() + axis, 1);
    return view(new_shape);
}

Tensor Tensor::transpose() const {
    if (ndim() != 2) throw std::runtime_error("transpose() requires 2D");
    Shape new_shape = {shape_[1], shape_[0]};
    Strides new_strides = {strides_[1], strides_[0]};
    return Tensor(data_, new_shape, new_strides, dtype_, stream_);
}

// 一元操作 - 全部用默认stream(0)，同步执行
#define UNARY(name, k) \
Tensor Tensor::name() const { \
    Tensor r(shape_, dtype_); \
    k(data<float>(), r.data<float>(), size(), 0); \
    cudaDeviceSynchronize(); \
    return r; \
}

UNARY(abs, launch_abs)
UNARY(sqrt, launch_sqrt)
UNARY(square, launch_square)
UNARY(exp, launch_exp)
UNARY(log, launch_log)
UNARY(sin, launch_sin)
UNARY(cos, launch_cos)
UNARY(sigmoid, launch_sigmoid)
UNARY(tanh, launch_tanh)
UNARY(relu, launch_relu)

// 二元操作(broadcast) - 默认stream(0)
#define BINARY(name, k) \
Tensor Tensor::name(const Tensor& o) const { \
    Shape s = broadcast_shape(shape_, o.shape_); \
    Tensor r(s, dtype_); \
    int *as=to_device(shape_), *ast=to_device(strides_), *bs=to_device(o.shape_), *bst=to_device(o.strides_), *os=to_device(s); \
    k(data<float>(), as, ast, o.data<float>(), bs, bst, r.data<float>(), os, s.size(), r.size(), 0); \
    cudaDeviceSynchronize(); \
    cudaFree(as); cudaFree(ast); cudaFree(bs); cudaFree(bst); cudaFree(os); \
    return r; \
}

BINARY(add, launch_broadcast_add)
BINARY(sub, launch_broadcast_sub)
BINARY(mul, launch_broadcast_mul)
BINARY(div, launch_broadcast_div)

// 归约
Tensor Tensor::sum(int dim, bool keepdim) const {
    if (dim == -1) dim = ndim() - 1;
    dim = normalize_axis(dim, ndim());
    
    int outer = 1, inner = 1;
    for (int i = 0; i < dim; i++) outer *= shape_[i];
    for (int i = dim + 1; i < ndim(); i++) inner *= shape_[i];
    int reduce_size = shape_[dim];
    
    Shape out_shape;
    for (int i = 0; i < ndim(); i++) {
        if (i == dim) { if (keepdim) out_shape.push_back(1); }
        else out_shape.push_back(shape_[i]);
    }
    
    Tensor result(out_shape, dtype_);
    launch_reduce_sum(data<float>(), result.data<float>(), outer, reduce_size, inner, 0);
    cudaDeviceSynchronize();
    return result;
}

Tensor Tensor::mean(int dim, bool keepdim) const {
    if (dim == -1) dim = ndim() - 1;
    dim = normalize_axis(dim, ndim());
    
    int outer = 1, inner = 1;
    for (int i = 0; i < dim; i++) outer *= shape_[i];
    for (int i = dim + 1; i < ndim(); i++) inner *= shape_[i];
    int reduce_size = shape_[dim];
    
    Shape out_shape;
    for (int i = 0; i < ndim(); i++) {
        if (i == dim) { if (keepdim) out_shape.push_back(1); }
        else out_shape.push_back(shape_[i]);
    }
    
    Tensor result(out_shape, dtype_);
    launch_reduce_mean(data<float>(), result.data<float>(), outer, reduce_size, inner, 0);
    cudaDeviceSynchronize();
    return result;
}

// Host交互
Tensor Tensor::from_array(const float* data, const Shape& shape) {
    Tensor result(shape);
    cudaMemcpy(result.data<float>(), data, result.size() * sizeof(float), cudaMemcpyHostToDevice);
    return result;
}

std::vector<float> Tensor::to_array() const {
    if (is_contiguous()) {
        std::vector<float> result(size());
        cudaMemcpy(result.data(), data<float>(), size() * sizeof(float), cudaMemcpyDeviceToHost);
        return result;
    }
    
    // 非连续: 拷贝底层数据到host，按stride逐元素读取
    // 计算底层数据的最大偏移来确定需要拷贝多少
    int max_offset = 0;
    for (int i = 0; i < ndim(); i++) {
        max_offset += (shape_[i] - 1) * strides_[i];
    }
    int raw_size = max_offset + 1;
    
    std::vector<float> raw(raw_size);
    cudaMemcpy(raw.data(), data<float>(), raw_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    // 按stride逐元素提取
    std::vector<float> result(size());
    for (int i = 0; i < size(); i++) {
        int remaining = i;
        int offset = 0;
        for (int d = 0; d < ndim(); d++) {
            int stride_count = 1;
            for (int dd = d + 1; dd < ndim(); dd++) stride_count *= shape_[dd];
            int idx = remaining / stride_count;
            remaining %= stride_count;
            offset += idx * strides_[d];
        }
        result[i] = raw[offset];
    }
    return result;
}

// 复数操作 - 暂不实现
Tensor Tensor::real() const { throw std::runtime_error("Not implemented"); }
Tensor Tensor::imag() const { throw std::runtime_error("Not implemented"); }
Tensor Tensor::from_real_imag(const Tensor&, const Tensor&) { throw std::runtime_error("Not implemented"); }

}
