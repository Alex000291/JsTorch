#include "tensor.hpp"
#include <stdexcept>
#include <cstdlib>
#include <ctime>

namespace jstorch {

// ==================== extern "C" kernel declarations ====================
extern "C" {
    // Unary (simple)
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
    void launch_neg(const float*, float*, int, cudaStream_t);
    void launch_floor(const float*, float*, int, cudaStream_t);
    void launch_ceil(const float*, float*, int, cudaStream_t);
    void launch_round(const float*, float*, int, cudaStream_t);
    void launch_silu(const float*, float*, int, cudaStream_t);
    void launch_gelu(const float*, float*, int, cudaStream_t);
    void launch_softplus(const float*, float*, int, cudaStream_t);
    // Unary (parameterized)
    void launch_leaky_relu(const float*, float*, int, float, cudaStream_t);
    void launch_clamp(const float*, float*, int, float, float, cudaStream_t);
    
    // Binary (broadcast) — macro-generated in binary.cu
    #define DECL_BROADCAST(name) \
    void launch_broadcast_##name(const float*, const int*, const int*, \
        const float*, const int*, const int*, float*, const int*, int, int, cudaStream_t);
    DECL_BROADCAST(add) DECL_BROADCAST(sub) DECL_BROADCAST(mul) DECL_BROADCAST(div)
    DECL_BROADCAST(maximum) DECL_BROADCAST(minimum) DECL_BROADCAST(pow)
    DECL_BROADCAST(gt) DECL_BROADCAST(lt) DECL_BROADCAST(ge)
    DECL_BROADCAST(le) DECL_BROADCAST(eq) DECL_BROADCAST(ne)
    
    // Reduce
    void launch_reduce_sum(const float*, float*, int, int, int, cudaStream_t);
    void launch_reduce_mean(const float*, float*, int, int, int, cudaStream_t);
    
    // Misc
    void launch_flip(const float*, float*, const int*, const int*, int, int, int, cudaStream_t);
    void launch_pad(const float*, float*, const int*, const int*, const int*, int, int, float, int, cudaStream_t);
    void launch_cumsum(const float*, float*, int, int, int, cudaStream_t);
    void launch_where(const float*, const float*, const float*, float*, int, cudaStream_t);
    void launch_randn(float*, int, unsigned long long, cudaStream_t);
    void launch_embedding(const float*, const int*, float*, int, int, cudaStream_t);
    void launch_interp1d(const float*, float*, int, int, int, int, bool, cudaStream_t);
    void launch_cat(const float*, float*, int, int, int, int, int, cudaStream_t);
    
    // Matmul
    void launch_matmul(const float*, const float*, float*, int, int, int, cudaStream_t);
    void launch_bmm(const float*, const float*, float*, int, int, int, int, cudaStream_t);
    
    // Conv
    void launch_conv1d(const float*, const float*, const float*, float*, float*,
        int, int, int, int, int, int, int, int, int, int, cudaStream_t);
    void launch_conv_transpose1d(const float*, const float*, const float*, float*, float*,
        int, int, int, int, int, int, int, int, int, int, cudaStream_t);
}

// ==================== Helpers ====================
static int* to_device(const std::vector<int>& vec) {
    int* d; cudaMalloc(&d, vec.size() * sizeof(int));
    cudaMemcpy(d, vec.data(), vec.size() * sizeof(int), cudaMemcpyHostToDevice);
    return d;
}

// Ensure contiguous strides for binary ops
static std::vector<int> contiguous_strides_for_broadcast(const Shape& shape, const Strides& strides, int target_ndim) {
    std::vector<int> result(target_ndim, 0);
    int offset = target_ndim - (int)shape.size();
    for (int i = 0; i < (int)shape.size(); i++)
        result[i + offset] = strides[i];
    return result;
}

static Shape pad_shape(const Shape& s, int ndim) {
    Shape r(ndim, 1);
    int offset = ndim - (int)s.size();
    for (int i = 0; i < (int)s.size(); i++) r[i + offset] = s[i];
    return r;
}

// ==================== Constructors ====================
Tensor::Tensor(const Shape& shape, DType dtype)
    : dtype_(dtype), shape_(shape), strides_(compute_strides(shape)),
      stream_(0), owns_stream_(false) {
    size_t bytes = size() * dtype_size(dtype_);
    void* ptr; cudaMalloc(&ptr, bytes); cudaMemset(ptr, 0, bytes);
    data_ = std::shared_ptr<void>(ptr, [](void* p) { cudaFree(p); });
}

Tensor::Tensor(std::shared_ptr<void> data, const Shape& shape, const Strides& strides,
               DType dtype, cudaStream_t stream)
    : data_(data), dtype_(dtype), shape_(shape), strides_(strides),
      stream_(0), owns_stream_(false) {}

Tensor::~Tensor() { if (owns_stream_ && stream_) cudaStreamDestroy(stream_); }

// ==================== Copy ====================
Tensor Tensor::clone() const {
    Tensor r(shape_, dtype_);
    cudaMemcpy(r.data_.get(), data_.get(), size() * dtype_size(dtype_), cudaMemcpyDeviceToDevice);
    return r;
}

Tensor Tensor::contiguous() const {
    if (is_contiguous()) return *this;
    // Need to actually copy with stride support
    Tensor r(shape_, dtype_);
    auto src = to_array();
    cudaMemcpy(r.data<float>(), src.data(), src.size() * sizeof(float), cudaMemcpyHostToDevice);
    return r;
}

// ==================== View ops ====================
Tensor Tensor::view(const Shape& new_shape) const {
    if (total_size(new_shape) != size()) throw std::runtime_error("View size mismatch");
    if (!is_contiguous()) throw std::runtime_error("View requires contiguous");
    return Tensor(data_, new_shape, compute_strides(new_shape), dtype_, stream_);
}

Tensor Tensor::reshape(const Shape& new_shape) const {
    // Handle -1 in shape
    Shape resolved = new_shape;
    int neg_idx = -1, known = 1;
    for (int i = 0; i < (int)resolved.size(); i++) {
        if (resolved[i] == -1) { neg_idx = i; }
        else { known *= resolved[i]; }
    }
    if (neg_idx >= 0) resolved[neg_idx] = size() / known;
    return is_contiguous() ? view(resolved) : contiguous().view(resolved);
}

Tensor Tensor::squeeze(int axis) const {
    axis = axis == -1 ? -1 : normalize_axis(axis, ndim());
    Shape ns;
    if (axis == -1) { for (int d : shape_) if (d != 1) ns.push_back(d); }
    else {
        if (shape_[axis] != 1) throw std::runtime_error("Can only squeeze size-1");
        for (int i = 0; i < ndim(); i++) if (i != axis) ns.push_back(shape_[i]);
    }
    if (ns.empty()) ns.push_back(1);
    return view(ns);
}

Tensor Tensor::unsqueeze(int axis) const {
    axis = normalize_axis(axis, ndim() + 1);
    Shape ns = shape_; ns.insert(ns.begin() + axis, 1);
    return view(ns);
}

// #6: transpose — 2D backward compat
Tensor Tensor::transpose() const {
    if (ndim() != 2) throw std::runtime_error("transpose() requires 2D");
    return transpose(0, 1);
}

// #6: transpose(dim0, dim1) — generalized
Tensor Tensor::transpose(int dim0, int dim1) const {
    dim0 = normalize_axis(dim0, ndim());
    dim1 = normalize_axis(dim1, ndim());
    Shape ns = shape_; Strides nst = strides_;
    std::swap(ns[dim0], ns[dim1]);
    std::swap(nst[dim0], nst[dim1]);
    return Tensor(data_, ns, nst, dtype_, stream_);
}

// #7: slice
Tensor Tensor::slice(int dim, int start, int end) const {
    dim = normalize_axis(dim, ndim());
    if (start < 0) start += shape_[dim];
    if (end < 0) end += shape_[dim];
    if (end > shape_[dim]) end = shape_[dim];
    if (start >= end) throw std::runtime_error("slice: start >= end");
    
    // Compute byte offset
    int offset = start * strides_[dim];
    float* new_ptr = const_cast<float*>(data<float>()) + offset;
    // Create shared_ptr that shares ownership with original
    std::shared_ptr<void> shared(data_, new_ptr);
    
    Shape ns = shape_; ns[dim] = end - start;
    return Tensor(shared, ns, strides_, dtype_, stream_);
}

// #8: split
std::vector<Tensor> Tensor::split(int dim, int chunk_size) const {
    dim = normalize_axis(dim, ndim());
    std::vector<Tensor> result;
    int total = shape_[dim];
    for (int s = 0; s < total; s += chunk_size) {
        int e = std::min(s + chunk_size, total);
        result.push_back(slice(dim, s, e));
    }
    return result;
}

// ==================== Unary ops ====================
#define UNARY(name, launcher) \
Tensor Tensor::name() const { \
    Tensor c = contiguous(); \
    Tensor r(shape_, dtype_); \
    launcher(c.data<float>(), r.data<float>(), size(), 0); \
    cudaDeviceSynchronize(); return r; \
}

UNARY(abs, launch_abs)   UNARY(sqrt, launch_sqrt)   UNARY(square, launch_square)
UNARY(exp, launch_exp)   UNARY(log, launch_log)     UNARY(sin, launch_sin)
UNARY(cos, launch_cos)   UNARY(neg, launch_neg)     UNARY(floor, launch_floor)
UNARY(ceil, launch_ceil) UNARY(round, launch_round) UNARY(sigmoid, launch_sigmoid)
UNARY(tanh, launch_tanh) UNARY(relu, launch_relu)   UNARY(silu, launch_silu)
UNARY(gelu, launch_gelu) UNARY(softplus, launch_softplus)

Tensor Tensor::leaky_relu(float slope) const {
    Tensor c = contiguous(); Tensor r(shape_, dtype_);
    launch_leaky_relu(c.data<float>(), r.data<float>(), size(), slope, 0);
    cudaDeviceSynchronize(); return r;
}

Tensor Tensor::clamp(float lo, float hi) const {
    Tensor c = contiguous(); Tensor r(shape_, dtype_);
    launch_clamp(c.data<float>(), r.data<float>(), size(), lo, hi, 0);
    cudaDeviceSynchronize(); return r;
}

// ==================== Binary ops ====================
#define BINARY(name, launcher) \
Tensor Tensor::name(const Tensor& o) const { \
    Shape s = broadcast_shape(shape_, o.shape_); \
    int nd = (int)s.size(); \
    auto as = pad_shape(shape_, nd), bs = pad_shape(o.shape_, nd); \
    auto ast = contiguous_strides_for_broadcast(shape_, is_contiguous() ? compute_strides(shape_) : strides_, nd); \
    auto bst = contiguous_strides_for_broadcast(o.shape_, o.is_contiguous() ? compute_strides(o.shape_) : o.strides_, nd); \
    Tensor ca = contiguous(), cb = o.contiguous(); \
    Tensor r(s, dtype_); \
    int *das=to_device(as), *dast=to_device(ast), *dbs=to_device(bs), *dbst=to_device(bst), *dos=to_device(s); \
    launcher(ca.data<float>(), das, dast, cb.data<float>(), dbs, dbst, r.data<float>(), dos, nd, r.size(), 0); \
    cudaDeviceSynchronize(); \
    cudaFree(das); cudaFree(dast); cudaFree(dbs); cudaFree(dbst); cudaFree(dos); \
    return r; \
}

BINARY(add, launch_broadcast_add) BINARY(sub, launch_broadcast_sub)
BINARY(mul, launch_broadcast_mul) BINARY(div, launch_broadcast_div)
BINARY(maximum, launch_broadcast_maximum) BINARY(minimum, launch_broadcast_minimum)
BINARY(pow, launch_broadcast_pow)
BINARY(gt, launch_broadcast_gt) BINARY(lt, launch_broadcast_lt)
BINARY(ge, launch_broadcast_ge) BINARY(le, launch_broadcast_le)
BINARY(eq, launch_broadcast_eq) BINARY(ne, launch_broadcast_ne)

// ==================== Reduce ====================
Tensor Tensor::sum(int dim, bool keepdim) const {
    Tensor c = contiguous();
    if (dim == -1) dim = ndim() - 1;
    dim = normalize_axis(dim, ndim());
    int outer = 1, inner = 1;
    for (int i = 0; i < dim; i++) outer *= shape_[i];
    for (int i = dim + 1; i < ndim(); i++) inner *= shape_[i];
    Shape os;
    for (int i = 0; i < ndim(); i++) {
        if (i == dim) { if (keepdim) os.push_back(1); }
        else os.push_back(shape_[i]);
    }
    if (os.empty()) os.push_back(1);
    Tensor r(os, dtype_);
    launch_reduce_sum(c.data<float>(), r.data<float>(), outer, shape_[dim], inner, 0);
    cudaDeviceSynchronize(); return r;
}

Tensor Tensor::mean(int dim, bool keepdim) const {
    Tensor c = contiguous();
    if (dim == -1) dim = ndim() - 1;
    dim = normalize_axis(dim, ndim());
    int outer = 1, inner = 1;
    for (int i = 0; i < dim; i++) outer *= shape_[i];
    for (int i = dim + 1; i < ndim(); i++) inner *= shape_[i];
    Shape os;
    for (int i = 0; i < ndim(); i++) {
        if (i == dim) { if (keepdim) os.push_back(1); }
        else os.push_back(shape_[i]);
    }
    if (os.empty()) os.push_back(1);
    Tensor r(os, dtype_);
    launch_reduce_mean(c.data<float>(), r.data<float>(), outer, shape_[dim], inner, 0);
    cudaDeviceSynchronize(); return r;
}

// ==================== #9 cat ====================
Tensor Tensor::cat(const std::vector<Tensor>& tensors, int dim) {
    if (tensors.empty()) throw std::runtime_error("cat: empty list");
    const auto& first = tensors[0];
    dim = normalize_axis(dim, first.ndim());
    
    // Compute output shape
    Shape os = first.shape();
    int total_dim = first.shape()[dim];
    for (size_t i = 1; i < tensors.size(); i++) {
        total_dim += tensors[i].shape()[dim];
    }
    os[dim] = total_dim;
    
    int outer = 1, inner = 1;
    for (int i = 0; i < dim; i++) outer *= os[i];
    for (int i = dim + 1; i < (int)os.size(); i++) inner *= os[i];
    
    Tensor r(os, first.dtype());
    int offset = 0;
    for (auto& t : tensors) {
        Tensor tc = t.contiguous();
        launch_cat(tc.data<float>(), r.data<float>(), outer, tc.shape()[dim], total_dim, inner, offset, 0);
        offset += tc.shape()[dim];
    }
    cudaDeviceSynchronize();
    return r;
}

// ==================== #10 flip ====================
Tensor Tensor::flip(int dim) const {
    dim = normalize_axis(dim, ndim());
    Tensor c = contiguous();
    Tensor r(shape_, dtype_);
    int* ds = to_device(shape_);
    int* dst = to_device(compute_strides(shape_));
    launch_flip(c.data<float>(), r.data<float>(), ds, dst, ndim(), dim, size(), 0);
    cudaDeviceSynchronize();
    cudaFree(ds); cudaFree(dst);
    return r;
}

// ==================== #11 pad ====================
Tensor Tensor::pad(const std::vector<int>& padding, int mode, float value) const {
    // padding format: [last_dim_left, last_dim_right, second_last_left, ...]
    Tensor c = contiguous();
    int nd = ndim();
    std::vector<int> pad_before(nd, 0), pad_after(nd, 0);
    for (int i = 0; i < (int)padding.size() / 2; i++) {
        int d = nd - 1 - i;
        pad_before[d] = padding[i * 2];
        pad_after[d] = padding[i * 2 + 1];
    }
    Shape os = shape_;
    for (int d = 0; d < nd; d++) os[d] += pad_before[d] + pad_after[d];
    
    Tensor r(os, dtype_);
    int* dis = to_device(shape_);
    int* dos = to_device(os);
    int* dpb = to_device(pad_before);
    launch_pad(c.data<float>(), r.data<float>(), dis, dos, dpb, nd, r.size(), value, mode, 0);
    cudaDeviceSynchronize();
    cudaFree(dis); cudaFree(dos); cudaFree(dpb);
    return r;
}

// ==================== #12 cumsum ====================
Tensor Tensor::cumsum(int dim) const {
    Tensor c = contiguous();
    dim = normalize_axis(dim, ndim());
    int outer = 1, inner = 1;
    for (int i = 0; i < dim; i++) outer *= shape_[i];
    for (int i = dim + 1; i < ndim(); i++) inner *= shape_[i];
    Tensor r(shape_, dtype_);
    launch_cumsum(c.data<float>(), r.data<float>(), outer, shape_[dim], inner, 0);
    cudaDeviceSynchronize();
    return r;
}

// ==================== #13 where ====================
Tensor Tensor::where(const Tensor& cond, const Tensor& x, const Tensor& y) {
    // All must be same shape (caller broadcasts beforehand)
    Tensor cc = cond.contiguous(), cx = x.contiguous(), cy = y.contiguous();
    Tensor r(cc.shape(), cc.dtype());
    launch_where(cc.data<float>(), cx.data<float>(), cy.data<float>(), r.data<float>(), r.size(), 0);
    cudaDeviceSynchronize();
    return r;
}

// ==================== #16 matmul ====================
Tensor Tensor::matmul(const Tensor& other) const {
    Tensor ca = contiguous(), cb = other.contiguous();
    if (ndim() == 2 && other.ndim() == 2) {
        int M = shape_[0], K = shape_[1], N = other.shape_[1];
        if (other.shape_[0] != K) throw std::runtime_error("matmul shape mismatch");
        Tensor r({M, N}, dtype_);
        launch_matmul(ca.data<float>(), cb.data<float>(), r.data<float>(), M, K, N, 0);
        cudaDeviceSynchronize(); return r;
    }
    if (ndim() == 3 && other.ndim() == 3) {
        int B = shape_[0], M = shape_[1], K = shape_[2], N = other.shape_[2];
        Tensor r({B, M, N}, dtype_);
        launch_bmm(ca.data<float>(), cb.data<float>(), r.data<float>(), B, M, K, N, 0);
        cudaDeviceSynchronize(); return r;
    }
    // 2D x 1D -> 1D (matrix-vector)
    if (ndim() == 2 && other.ndim() == 1) {
        int M = shape_[0], K = shape_[1];
        Tensor ob = cb.unsqueeze(1); // [K,1]
        Tensor r({M, 1}, dtype_);
        launch_matmul(ca.data<float>(), ob.data<float>(), r.data<float>(), M, K, 1, 0);
        cudaDeviceSynchronize();
        return r.squeeze(1);
    }
    throw std::runtime_error("matmul: unsupported dims");
}

// ==================== #17 embedding ====================
Tensor Tensor::embedding(const Tensor& indices) const {
    // self = weight [num_embed, embed_dim], indices = [...]
    int embed_dim = shape_[1];
    int num_indices = indices.size();
    Shape os = indices.shape();
    os.push_back(embed_dim);
    Tensor r(os, dtype_);
    Tensor cw = contiguous();
    launch_embedding(cw.data<float>(), indices.data<int>(), r.data<float>(), num_indices, embed_dim, 0);
    cudaDeviceSynchronize();
    return r;
}

// ==================== #18 Conv1d ====================
Tensor Tensor::conv1d(const Tensor& weight, const Tensor* bias,
    int stride, int padding, int dilation, int groups) const {
    // input: [B, C_in, L_in], weight: [C_out, C_in/groups, K]
    Tensor ci = contiguous(), cw = weight.contiguous();
    int B = shape_[0], C_in = shape_[1], L_in = shape_[2];
    int C_out = weight.shape()[0], K = weight.shape()[2];
    int L_out = (L_in + 2 * padding - dilation * (K - 1) - 1) / stride + 1;
    
    // Allocate col buffer
    int col_size = B * C_in * K * L_out;
    float* col_buf; cudaMalloc(&col_buf, col_size * sizeof(float));
    
    Tensor r({B, C_out, L_out}, dtype_);
    const float* b_ptr = bias ? bias->data<float>() : nullptr;
    launch_conv1d(ci.data<float>(), cw.data<float>(), b_ptr, r.data<float>(), col_buf,
        B, C_in, L_in, C_out, K, stride, padding, dilation, groups, L_out, 0);
    cudaDeviceSynchronize();
    cudaFree(col_buf);
    return r;
}

// ==================== #19 ConvTranspose1d ====================
Tensor Tensor::conv_transpose1d(const Tensor& weight, const Tensor* bias,
    int stride, int padding, int output_padding, int dilation, int groups) const {
    Tensor ci = contiguous(), cw = weight.contiguous();
    int B = shape_[0], C_in = shape_[1], L_in = shape_[2];
    int C_out = weight.shape()[1] * groups, K = weight.shape()[2];
    int L_out = (L_in - 1) * stride - 2 * padding + dilation * (K - 1) + output_padding + 1;
    
    int col_size = B * C_out * K * L_in;
    float* col_buf; cudaMalloc(&col_buf, col_size * sizeof(float));
    
    Tensor r({B, C_out, L_out}, dtype_);
    const float* b_ptr = bias ? bias->data<float>() : nullptr;
    launch_conv_transpose1d(ci.data<float>(), cw.data<float>(), b_ptr, r.data<float>(), col_buf,
        B, C_in, L_in, C_out, K, stride, padding, dilation, groups, L_out, 0);
    cudaDeviceSynchronize();
    cudaFree(col_buf);
    return r;
}

// ==================== #20 interpolate ====================
Tensor Tensor::interpolate(int target_size, int mode, bool align_corners) const {
    // input: [B, C, L] -> [B, C, target_size]
    Tensor c = contiguous();
    int bc = 1;
    for (int i = 0; i < ndim() - 1; i++) bc *= shape_[i];
    int in_len = shape_[ndim() - 1];
    Shape os = shape_; os[ndim() - 1] = target_size;
    Tensor r(os, dtype_);
    launch_interp1d(c.data<float>(), r.data<float>(), bc, in_len, target_size, mode, align_corners, 0);
    cudaDeviceSynchronize();
    return r;
}

// ==================== Factory ====================
Tensor Tensor::from_array(const float* data, const Shape& shape) {
    Tensor r(shape);
    cudaMemcpy(r.data<float>(), data, r.size() * sizeof(float), cudaMemcpyHostToDevice);
    return r;
}

// #14: from_buffer — same as from_array but explicit count param
Tensor Tensor::from_buffer(const float* data, int count, const Shape& shape) {
    if (total_size(shape) != count) throw std::runtime_error("from_buffer: size mismatch");
    return from_array(data, shape);
}

// #15: randn
Tensor Tensor::randn(const Shape& shape) {
    Tensor r(shape);
    static unsigned long long seed_counter = 42;
    launch_randn(r.data<float>(), r.size(), seed_counter++, 0);
    cudaDeviceSynchronize();
    return r;
}

Tensor Tensor::randn_like() const {
    return Tensor::randn(shape_);
}

// Int tensor for embedding indices
Tensor Tensor::from_int_array(const int* data, const Shape& shape) {
    int count = total_size(shape);
    Tensor r(shape, DType::Float32); // Reuse allocation, store ints
    // Allocate as int
    void* ptr; cudaMalloc(&ptr, count * sizeof(int));
    cudaMemcpy(ptr, data, count * sizeof(int), cudaMemcpyHostToDevice);
    r.data_ = std::shared_ptr<void>(ptr, [](void* p) { cudaFree(p); });
    return r;
}

std::vector<float> Tensor::to_array() const {
    if (is_contiguous()) {
        std::vector<float> result(size());
        cudaMemcpy(result.data(), data<float>(), size() * sizeof(float), cudaMemcpyDeviceToHost);
        return result;
    }
    int max_offset = 0;
    for (int i = 0; i < ndim(); i++)
        max_offset += (shape_[i] - 1) * strides_[i];
    int raw_size = max_offset + 1;
    std::vector<float> raw(raw_size);
    cudaMemcpy(raw.data(), data<float>(), raw_size * sizeof(float), cudaMemcpyDeviceToHost);
    std::vector<float> result(size());
    for (int i = 0; i < size(); i++) {
        int remaining = i, offset = 0;
        for (int d = 0; d < ndim(); d++) {
            int sc = 1;
            for (int dd = d + 1; dd < ndim(); dd++) sc *= shape_[dd];
            int idx = remaining / sc; remaining %= sc;
            offset += idx * strides_[d];
        }
        result[i] = raw[offset];
    }
    return result;
}

// Complex — not implemented for RVC
Tensor Tensor::real() const { throw std::runtime_error("Not implemented"); }
Tensor Tensor::imag() const { throw std::runtime_error("Not implemented"); }
Tensor Tensor::from_real_imag(const Tensor&, const Tensor&) { throw std::runtime_error("Not implemented"); }

}
