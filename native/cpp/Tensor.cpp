// Tensor.cpp - Host-side Tensor with GPU data pointer
#include <napi.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <sstream>

// CUDA kernel declarations (implemented in .cu files)
extern "C" {
    void init_cublas();
    void cleanup_cublas();
    void launch_matmul(const float* A, const float* B, float* C, 
                      int M, int N, int K, cudaStream_t stream);
    void launch_add(const float* A, const float* B, float* C, 
                   int size, cudaStream_t stream);
    void launch_sub(const float* A, const float* B, float* C,
                   int size, cudaStream_t stream);
    void launch_mul(const float* A, const float* B, float* C,
                   int size, cudaStream_t stream);
    void launch_div(const float* A, const float* B, float* C,
                   int size, cudaStream_t stream);
    void launch_mul_scalar(const float* input, float scalar, float* output,
                          int size, cudaStream_t stream);
    void launch_relu(const float* input, float* output, int size, cudaStream_t stream);
    void launch_exp(const float* input, float* output, int size, cudaStream_t stream);
    void launch_log(const float* input, float* output, int size, cudaStream_t stream);
    void launch_neg(const float* input, float* output, int size, cudaStream_t stream);
    void launch_sum(const float* input, float* output, int size, cudaStream_t stream);
    void launch_conv2d(const float* input, const float* weight, const float* bias, float* output,
                      int batch, int in_channels, int out_channels,
                      int input_h, int input_w, int kernel_h, int kernel_w,
                      int stride_h, int stride_w, int padding_h, int padding_w,
                      int output_h, int output_w, cudaStream_t stream);
    void launch_conv2d_backward_input(const float* grad_output, const float* weight, float* grad_input,
                                     int batch, int in_channels, int out_channels,
                                     int input_h, int input_w, int kernel_h, int kernel_w,
                                     int stride_h, int stride_w, int padding_h, int padding_w,
                                     int output_h, int output_w, cudaStream_t stream);
    void launch_conv2d_backward_weight(const float* input, const float* grad_output, float* grad_weight,
                                      int batch, int in_channels, int out_channels,
                                      int input_h, int input_w, int kernel_h, int kernel_w,
                                      int stride_h, int stride_w, int padding_h, int padding_w,
                                      int output_h, int output_w, cudaStream_t stream);
    void launch_conv2d_backward_bias(const float* grad_output, float* grad_bias,
                                    int batch, int out_channels, int output_h, int output_w,
                                    cudaStream_t stream);
    void launch_maxpool2d(const float* input, float* output,
                         int batch, int channels, int input_h, int input_w,
                         int kernel_h, int kernel_w, int stride_h, int stride_w,
                         int padding_h, int padding_w, int output_h, int output_w,
                         cudaStream_t stream);
}

// ==================== Tensor Class ====================

class Tensor : public Napi::ObjectWrap<Tensor> {
private:
    // Host-side metadata
    std::vector<int> shape_;
    int size_;           // Total number of elements
    
    // GPU pointer
    float* d_data_;      // Device data pointer
    
    // Device management
    int device_;
    cudaStream_t stream_;

public:
    // Constructor
    Tensor(const Napi::CallbackInfo& info) : Napi::ObjectWrap<Tensor>(info) {
        Napi::Env env = info.Env();
        
        // Parse shape and data from JavaScript array
        auto data = parseArray(info[0]);
        shape_ = inferShape(info[0]);
        size_ = computeSize(shape_);
        device_ = 0;
        
        // Allocate GPU memory
        cudaMalloc(&d_data_, size_ * sizeof(float));
        cudaStreamCreate(&stream_);
        
        // Copy data to GPU
        cudaMemcpy(d_data_, data.data(), size_ * sizeof(float), cudaMemcpyHostToDevice);
    }
    
    ~Tensor() {
        if (d_data_) cudaFree(d_data_);
        if (stream_) cudaStreamDestroy(stream_);
    }
    
    // Static factory method
    static Napi::Object Init(Napi::Env env, Napi::Object exports) {
        Napi::Function func = DefineClass(env, "Tensor", {
            InstanceMethod("add", &Tensor::Add),
            InstanceMethod("sub", &Tensor::Sub),
            InstanceMethod("mul", &Tensor::Mul),
            InstanceMethod("div", &Tensor::Div),
            InstanceMethod("mulScalar", &Tensor::MulScalar),
            InstanceMethod("matmul", &Tensor::MatMul),
            InstanceMethod("relu", &Tensor::ReLU),
            InstanceMethod("exp", &Tensor::Exp),
            InstanceMethod("log", &Tensor::Log),
            InstanceMethod("neg", &Tensor::Neg),
            InstanceMethod("sum", &Tensor::Sum),
            InstanceMethod("mean", &Tensor::Mean),
            InstanceMethod("item", &Tensor::Item),
            InstanceMethod("toArray", &Tensor::ToArray),
            InstanceMethod("conv2d", &Tensor::Conv2D),
            InstanceMethod("conv2dBackwardInput", &Tensor::Conv2DBackwardInput),
            InstanceMethod("conv2dBackwardWeight", &Tensor::Conv2DBackwardWeight),
            InstanceMethod("conv2dBackwardBias", &Tensor::Conv2DBackwardBias),
            InstanceMethod("maxpool2d", &Tensor::MaxPool2D),
            InstanceAccessor("shape", &Tensor::GetShape, nullptr),
            InstanceAccessor("size", &Tensor::GetSize, nullptr),
        });
        
        Napi::FunctionReference* constructor = new Napi::FunctionReference();
        *constructor = Napi::Persistent(func);
        env.SetInstanceData(constructor);
        
        exports.Set("Tensor", func);
        return exports;
    }
    
    // ==================== Operations ====================
    
    // Element-wise add
    Napi::Value Add(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        
        if (info.Length() < 1 || !info[0].IsObject()) {
            Napi::TypeError::New(env, "Expected tensor argument").ThrowAsJavaScriptException();
            return env.Null();
        }
        
        Tensor* other = Napi::ObjectWrap<Tensor>::Unwrap(info[0].As<Napi::Object>());
        
        if (!other) {
            Napi::TypeError::New(env, "Invalid tensor argument").ThrowAsJavaScriptException();
            return env.Null();
        }
        
        // Check shape compatibility
        if (shape_ != other->shape_) {
            Napi::TypeError::New(env, "Shape mismatch").ThrowAsJavaScriptException();
            return env.Null();
        }
        
        // Create result - copy this tensor's data to host, add, then create new tensor
        std::vector<float> h_a(size_);
        std::vector<float> h_b(size_);
        std::vector<float> h_c(size_);
        
        cudaMemcpy(h_a.data(), d_data_, size_ * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_b.data(), other->d_data_, size_ * sizeof(float), cudaMemcpyDeviceToHost);
        
        // CPU addition (temporary - will move back to GPU later)
        for (int i = 0; i < size_; i++) {
            h_c[i] = h_a[i] + h_b[i];
        }
        
        // Convert back to nested array
        Napi::Value resultData = buildNestedArray(env, h_c, shape_, 0, 0);
        
        // Create new Tensor using constructor
        Napi::FunctionReference* constructor = env.GetInstanceData<Napi::FunctionReference>();
        return constructor->New({ resultData });
    }
    
    // Matrix multiplication
    Napi::Value MatMul(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        
        if (info.Length() < 1 || !info[0].IsObject()) {
            Napi::TypeError::New(env, "Expected tensor argument").ThrowAsJavaScriptException();
            return env.Null();
        }
        
        Tensor* other = Napi::ObjectWrap<Tensor>::Unwrap(info[0].As<Napi::Object>());
        
        if (!other || shape_.size() != 2 || other->shape_.size() != 2) {
            Napi::TypeError::New(env, "Both tensors must be 2D").ThrowAsJavaScriptException();
            return env.Null();
        }
        
        int M = shape_[0];
        int K = shape_[1];
        int K2 = other->shape_[0];
        int N = other->shape_[1];
        
        if (K != K2) {
            Napi::TypeError::New(env, "Inner dimensions must match").ThrowAsJavaScriptException();
            return env.Null();
        }
        
        // Allocate GPU memory for result
        float* d_result;
        cudaMalloc(&d_result, M * N * sizeof(float));
        
        // Perform matmul on GPU using cuBLAS
        launch_matmul(d_data_, other->d_data_, d_result, M, N, K, stream_);
        cudaStreamSynchronize(stream_);
        
        // Copy result back to host
        std::vector<float> h_result(M * N);
        cudaMemcpy(h_result.data(), d_result, M * N * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_result);
        
        // Convert to nested array
        std::vector<int> result_shape = {M, N};
        Napi::Value resultData = buildNestedArray(env, h_result, result_shape, 0, 0);
        
        // Create new Tensor using constructor
        Napi::FunctionReference* constructor = env.GetInstanceData<Napi::FunctionReference>();
        return constructor->New({ resultData });
    }
    
    // Copy data from GPU to CPU and return as JavaScript array
    Napi::Value ToArray(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        
        // Allocate host memory
        std::vector<float> h_data(size_);
        
        // Copy from GPU to CPU
        cudaMemcpy(h_data.data(), d_data_, size_ * sizeof(float), cudaMemcpyDeviceToHost);
        
        // Convert to nested JavaScript array based on shape
        return buildNestedArray(env, h_data, shape_, 0, 0);
    }
    
    // ==================== Accessors ====================
    
    Napi::Value GetShape(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        Napi::Array arr = Napi::Array::New(env, shape_.size());
        for (size_t i = 0; i < shape_.size(); i++) {
            arr[i] = Napi::Number::New(env, shape_[i]);
        }
        return arr;
    }
    
    Napi::Value GetSize(const Napi::CallbackInfo& info) {
        return Napi::Number::New(info.Env(), size_);
    }
    
    // ReLU activation
    Napi::Value ReLU(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        
        // Allocate GPU memory for result
        float* d_result;
        cudaMalloc(&d_result, size_ * sizeof(float));
        
        // Launch ReLU kernel
        launch_relu(d_data_, d_result, size_, stream_);
        cudaStreamSynchronize(stream_);
        
        // Copy result back to host
        std::vector<float> h_result(size_);
        cudaMemcpy(h_result.data(), d_result, size_ * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_result);
        
        // Convert to nested array and create new Tensor
        Napi::Value resultData = buildNestedArray(env, h_result, shape_, 0, 0);
        Napi::FunctionReference* constructor = env.GetInstanceData<Napi::FunctionReference>();
        return constructor->New({ resultData });
    }
    
    // Get scalar value (for loss.item())
    Napi::Value Item(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        
        if (size_ != 1) {
            Napi::TypeError::New(env, "item() only works on scalar tensors").ThrowAsJavaScriptException();
            return env.Null();
        }
        
        float value;
        cudaMemcpy(&value, d_data_, sizeof(float), cudaMemcpyDeviceToHost);
        
        return Napi::Number::New(env, value);
    }
    
    // Subtraction
    Napi::Value Sub(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        if (info.Length() < 1 || !info[0].IsObject()) {
            Napi::TypeError::New(env, "Expected tensor argument").ThrowAsJavaScriptException();
            return env.Null();
        }
        
        Tensor* other = Napi::ObjectWrap<Tensor>::Unwrap(info[0].As<Napi::Object>());
        if (!other || shape_ != other->shape_) {
            Napi::TypeError::New(env, "Shape mismatch").ThrowAsJavaScriptException();
            return env.Null();
        }
        
        std::vector<float> h_a(size_), h_b(size_), h_c(size_);
        cudaMemcpy(h_a.data(), d_data_, size_ * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_b.data(), other->d_data_, size_ * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < size_; i++) h_c[i] = h_a[i] - h_b[i];
        
        Napi::Value resultData = buildNestedArray(env, h_c, shape_, 0, 0);
        Napi::FunctionReference* constructor = env.GetInstanceData<Napi::FunctionReference>();
        return constructor->New({ resultData });
    }
    
    // Multiplication
    Napi::Value Mul(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        if (info.Length() < 1 || !info[0].IsObject()) {
            Napi::TypeError::New(env, "Expected tensor argument").ThrowAsJavaScriptException();
            return env.Null();
        }
        
        Tensor* other = Napi::ObjectWrap<Tensor>::Unwrap(info[0].As<Napi::Object>());
        if (!other || shape_ != other->shape_) {
            Napi::TypeError::New(env, "Shape mismatch").ThrowAsJavaScriptException();
            return env.Null();
        }
        
        std::vector<float> h_a(size_), h_b(size_), h_c(size_);
        cudaMemcpy(h_a.data(), d_data_, size_ * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_b.data(), other->d_data_, size_ * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < size_; i++) h_c[i] = h_a[i] * h_b[i];
        
        Napi::Value resultData = buildNestedArray(env, h_c, shape_, 0, 0);
        Napi::FunctionReference* constructor = env.GetInstanceData<Napi::FunctionReference>();
        return constructor->New({ resultData });
    }
    
    // Division
    Napi::Value Div(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        if (info.Length() < 1 || !info[0].IsObject()) {
            Napi::TypeError::New(env, "Expected tensor argument").ThrowAsJavaScriptException();
            return env.Null();
        }
        
        Tensor* other = Napi::ObjectWrap<Tensor>::Unwrap(info[0].As<Napi::Object>());
        if (!other || shape_ != other->shape_) {
            Napi::TypeError::New(env, "Shape mismatch").ThrowAsJavaScriptException();
            return env.Null();
        }
        
        std::vector<float> h_a(size_), h_b(size_), h_c(size_);
        cudaMemcpy(h_a.data(), d_data_, size_ * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_b.data(), other->d_data_, size_ * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < size_; i++) h_c[i] = h_a[i] / h_b[i];
        
        Napi::Value resultData = buildNestedArray(env, h_c, shape_, 0, 0);
        Napi::FunctionReference* constructor = env.GetInstanceData<Napi::FunctionReference>();
        return constructor->New({ resultData });
    }
    
    // Scalar multiplication
    Napi::Value MulScalar(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        if (info.Length() < 1 || !info[0].IsNumber()) {
            Napi::TypeError::New(env, "Expected scalar number").ThrowAsJavaScriptException();
            return env.Null();
        }
        
        float scalar = info[0].As<Napi::Number>().FloatValue();
        std::vector<float> h_data(size_);
        cudaMemcpy(h_data.data(), d_data_, size_ * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < size_; i++) h_data[i] *= scalar;
        
        Napi::Value resultData = buildNestedArray(env, h_data, shape_, 0, 0);
        Napi::FunctionReference* constructor = env.GetInstanceData<Napi::FunctionReference>();
        return constructor->New({ resultData });
    }
    
    // Exp
    Napi::Value Exp(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        
        float* d_result;
        cudaMalloc(&d_result, size_ * sizeof(float));
        launch_exp(d_data_, d_result, size_, stream_);
        cudaStreamSynchronize(stream_);
        
        std::vector<float> h_result(size_);
        cudaMemcpy(h_result.data(), d_result, size_ * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_result);
        
        Napi::Value resultData = buildNestedArray(env, h_result, shape_, 0, 0);
        Napi::FunctionReference* constructor = env.GetInstanceData<Napi::FunctionReference>();
        return constructor->New({ resultData });
    }
    
    // Log
    Napi::Value Log(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        
        float* d_result;
        cudaMalloc(&d_result, size_ * sizeof(float));
        launch_log(d_data_, d_result, size_, stream_);
        cudaStreamSynchronize(stream_);
        
        std::vector<float> h_result(size_);
        cudaMemcpy(h_result.data(), d_result, size_ * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_result);
        
        Napi::Value resultData = buildNestedArray(env, h_result, shape_, 0, 0);
        Napi::FunctionReference* constructor = env.GetInstanceData<Napi::FunctionReference>();
        return constructor->New({ resultData });
    }
    
    // Negate
    Napi::Value Neg(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        
        float* d_result;
        cudaMalloc(&d_result, size_ * sizeof(float));
        launch_neg(d_data_, d_result, size_, stream_);
        cudaStreamSynchronize(stream_);
        
        std::vector<float> h_result(size_);
        cudaMemcpy(h_result.data(), d_result, size_ * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_result);
        
        Napi::Value resultData = buildNestedArray(env, h_result, shape_, 0, 0);
        Napi::FunctionReference* constructor = env.GetInstanceData<Napi::FunctionReference>();
        return constructor->New({ resultData });
    }
    
    // Sum
    Napi::Value Sum(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        
        float* d_result;
        cudaMalloc(&d_result, sizeof(float));
        launch_sum(d_data_, d_result, size_, stream_);
        cudaStreamSynchronize(stream_);
        
        float result;
        cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_result);
        
        // Return as scalar tensor
        Napi::Value resultData = Napi::Array::New(env, 1);
        resultData.As<Napi::Array>()[uint32_t(0)] = Napi::Number::New(env, result);
        
        Napi::FunctionReference* constructor = env.GetInstanceData<Napi::FunctionReference>();
        return constructor->New({ resultData });
    }
    
    // Mean
    Napi::Value Mean(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        
        float* d_result;
        cudaMalloc(&d_result, sizeof(float));
        launch_sum(d_data_, d_result, size_, stream_);
        cudaStreamSynchronize(stream_);
        
        float sum_val;
        cudaMemcpy(&sum_val, d_result, sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_result);
        
        float mean_val = sum_val / size_;
        
        // Return as scalar tensor
        Napi::Value resultData = Napi::Array::New(env, 1);
        resultData.As<Napi::Array>()[uint32_t(0)] = Napi::Number::New(env, mean_val);
        
        Napi::FunctionReference* constructor = env.GetInstanceData<Napi::FunctionReference>();
        return constructor->New({ resultData });
    }
    
    // ==================== Helpers ====================
    
private:
    // Default constructor for internal use
    Tensor(Napi::Env env) : Napi::ObjectWrap<Tensor>(Napi::CallbackInfo(env, nullptr)) {
        d_data_ = nullptr;
        stream_ = nullptr;
        size_ = 0;
        device_ = 0;
    }
    
    // Parse nested JavaScript array to flat vector
    std::vector<float> parseArray(Napi::Value val) {
        std::vector<float> result;
        flattenArray(val, result);
        return result;
    }
    
    void flattenArray(Napi::Value val, std::vector<float>& out) {
        if (val.IsArray()) {
            Napi::Array arr = val.As<Napi::Array>();
            for (uint32_t i = 0; i < arr.Length(); i++) {
                flattenArray(arr[i], out);
            }
        } else if (val.IsNumber()) {
            out.push_back(val.As<Napi::Number>().FloatValue());
        }
    }
    
    // Infer shape from nested array
    std::vector<int> inferShape(Napi::Value val) {
        std::vector<int> shape;
        Napi::Value current = val;
        
        while (current.IsArray()) {
            Napi::Array arr = current.As<Napi::Array>();
            shape.push_back(arr.Length());
            if (arr.Length() > 0) {
                current = arr[uint32_t(0)];
            } else {
                break;
            }
        }
        
        return shape;
    }
    
    // Compute total size from shape
    int computeSize(const std::vector<int>& shape) {
        int size = 1;
        for (int dim : shape) size *= dim;
        return size;
    }
    
    // Build nested JavaScript array from flat data
    Napi::Value buildNestedArray(Napi::Env env, const std::vector<float>& data,
                                 const std::vector<int>& shape, int dim, int offset) {
        if (dim == shape.size() - 1) {
            // Last dimension: create flat array
            Napi::Array arr = Napi::Array::New(env, shape[dim]);
            for (int i = 0; i < shape[dim]; i++) {
                arr[i] = Napi::Number::New(env, data[offset + i]);
            }
            return arr;
        } else {
            // Recursive case
            Napi::Array arr = Napi::Array::New(env, shape[dim]);
            int stride = 1;
            for (size_t i = dim + 1; i < shape.size(); i++) {
                stride *= shape[i];
            }
            for (int i = 0; i < shape[dim]; i++) {
                arr[i] = buildNestedArray(env, data, shape, dim + 1, offset + i * stride);
            }
            return arr;
        }
    }
    
public:
    // Accessors for internal use
    float* data() { return d_data_; }
    const float* data() const { return d_data_; }
    const std::vector<int>& shape() const { return shape_; }
    int size() const { return size_; }
    cudaStream_t stream() const { return stream_; }
    
    // ==================== Conv2D Forward ====================
    
    // Conv2D Forward
    // input: this tensor [batch, in_channels, height, width]
    // weight: [out_channels, in_channels, kernel_h, kernel_w]
    // bias: [out_channels] or null
    // Returns: [batch, out_channels, out_h, out_w]
    Napi::Value Conv2D(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        
        // Parse arguments: weight, bias (optional), stride, padding
        Tensor* weight = Tensor::Unwrap(info[0].As<Napi::Object>());
        Tensor* bias = info[1].IsNull() ? nullptr : Tensor::Unwrap(info[1].As<Napi::Object>());
        
        int stride_h = info[2].As<Napi::Number>().Int32Value();
        int stride_w = info[3].As<Napi::Number>().Int32Value();
        int padding_h = info[4].As<Napi::Number>().Int32Value();
        int padding_w = info[5].As<Napi::Number>().Int32Value();
        
        // Input shape: [batch, in_channels, height, width]
        if (shape_.size() != 4) {
            Napi::TypeError::New(env, "Conv2D input must be 4D [batch, channels, height, width]").ThrowAsJavaScriptException();
            return env.Null();
        }
        
        int batch = shape_[0];
        int in_channels = shape_[1];
        int input_h = shape_[2];
        int input_w = shape_[3];
        
        // Weight shape: [out_channels, in_channels, kernel_h, kernel_w]
        int out_channels = weight->shape_[0];
        int kernel_h = weight->shape_[2];
        int kernel_w = weight->shape_[3];
        
        // Calculate output dimensions
        int output_h = (input_h + 2 * padding_h - kernel_h) / stride_h + 1;
        int output_w = (input_w + 2 * padding_w - kernel_w) / stride_w + 1;
        
        // Create output tensor
        std::vector<int> out_shape = {batch, out_channels, output_h, output_w};
        int out_size = batch * out_channels * output_h * output_w;
        
        // Allocate output GPU memory
        float* d_out;
        cudaMalloc(&d_out, out_size * sizeof(float));
        
        // Launch kernel
        launch_conv2d(
            d_data_, weight->d_data_, bias ? bias->d_data_ : nullptr, d_out,
            batch, in_channels, out_channels,
            input_h, input_w, kernel_h, kernel_w,
            stride_h, stride_w, padding_h, padding_w,
            output_h, output_w, stream_
        );
        
        cudaStreamSynchronize(stream_);
        
        // Create result tensor from GPU data
        std::vector<float> host_data(out_size);
        cudaMemcpy(host_data.data(), d_out, out_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_out);
        
        // Convert to nested JS array
        Napi::Value resultData = buildNestedArray(env, host_data, out_shape, 0, 0);
        
        // Create Tensor
        Napi::FunctionReference* constructor = env.GetInstanceData<Napi::FunctionReference>();
        return constructor->New({ resultData });
    }
    
    // ==================== Conv2D Backward ====================
    
    // Conv2D Backward Input
    // grad_output: this tensor [batch, out_channels, out_h, out_w]
    // weight: [out_channels, in_channels, kernel_h, kernel_w]
    // input_shape: [batch, in_channels, input_h, input_w]
    // Returns: grad_input [batch, in_channels, input_h, input_w]
    Napi::Value Conv2DBackwardInput(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        
        // Parse arguments: weight, input_shape, stride, padding
        Tensor* weight = Tensor::Unwrap(info[0].As<Napi::Object>());
        Napi::Array input_shape_arr = info[1].As<Napi::Array>();
        int stride_h = info[2].As<Napi::Number>().Int32Value();
        int stride_w = info[3].As<Napi::Number>().Int32Value();
        int padding_h = info[4].As<Napi::Number>().Int32Value();
        int padding_w = info[5].As<Napi::Number>().Int32Value();
        
        // Parse input shape
        int batch = input_shape_arr.Get(uint32_t(0)).As<Napi::Number>().Int32Value();
        int in_channels = input_shape_arr.Get(uint32_t(1)).As<Napi::Number>().Int32Value();
        int input_h = input_shape_arr.Get(uint32_t(2)).As<Napi::Number>().Int32Value();
        int input_w = input_shape_arr.Get(uint32_t(3)).As<Napi::Number>().Int32Value();
        
        // grad_output shape: this tensor
        int out_channels = shape_[1];
        int output_h = shape_[2];
        int output_w = shape_[3];
        
        // Weight shape
        int kernel_h = weight->shape_[2];
        int kernel_w = weight->shape_[3];
        
        // Allocate grad_input
        std::vector<int> grad_input_shape = {batch, in_channels, input_h, input_w};
        int grad_input_size = batch * in_channels * input_h * input_w;
        
        float* d_grad_input;
        cudaMalloc(&d_grad_input, grad_input_size * sizeof(float));
        
        // Launch backward input kernel
        launch_conv2d_backward_input(
            d_data_, weight->d_data_, d_grad_input,
            batch, in_channels, out_channels,
            input_h, input_w, kernel_h, kernel_w,
            stride_h, stride_w, padding_h, padding_w,
            output_h, output_w, stream_
        );
        
        cudaStreamSynchronize(stream_);
        
        // Copy to host
        std::vector<float> host_data(grad_input_size);
        cudaMemcpy(host_data.data(), d_grad_input, grad_input_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_grad_input);
        
        // Convert to nested array
        Napi::Value resultData = buildNestedArray(env, host_data, grad_input_shape, 0, 0);
        
        // Create Tensor
        Napi::FunctionReference* constructor = env.GetInstanceData<Napi::FunctionReference>();
        return constructor->New({ resultData });
    }
    
    // Conv2D Backward Weight
    // grad_output: this tensor [batch, out_channels, out_h, out_w]
    // input: [batch, in_channels, input_h, input_w]
    // weight_shape: [out_channels, in_channels, kernel_h, kernel_w]
    // Returns: grad_weight [out_channels, in_channels, kernel_h, kernel_w]
    Napi::Value Conv2DBackwardWeight(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        
        // Parse arguments: input, weight_shape, stride, padding
        Tensor* input = Tensor::Unwrap(info[0].As<Napi::Object>());
        Napi::Array weight_shape_arr = info[1].As<Napi::Array>();
        int stride_h = info[2].As<Napi::Number>().Int32Value();
        int stride_w = info[3].As<Napi::Number>().Int32Value();
        int padding_h = info[4].As<Napi::Number>().Int32Value();
        int padding_w = info[5].As<Napi::Number>().Int32Value();
        
        // Parse weight shape
        int out_channels = weight_shape_arr.Get(uint32_t(0)).As<Napi::Number>().Int32Value();
        int in_channels = weight_shape_arr.Get(uint32_t(1)).As<Napi::Number>().Int32Value();
        int kernel_h = weight_shape_arr.Get(uint32_t(2)).As<Napi::Number>().Int32Value();
        int kernel_w = weight_shape_arr.Get(uint32_t(3)).As<Napi::Number>().Int32Value();
        
        // Input shape
        int batch = input->shape_[0];
        int input_h = input->shape_[2];
        int input_w = input->shape_[3];
        
        // grad_output shape: this tensor
        int output_h = shape_[2];
        int output_w = shape_[3];
        
        // Allocate grad_weight
        std::vector<int> grad_weight_shape = {out_channels, in_channels, kernel_h, kernel_w};
        int grad_weight_size = out_channels * in_channels * kernel_h * kernel_w;
        
        float* d_grad_weight;
        cudaMalloc(&d_grad_weight, grad_weight_size * sizeof(float));
        
        // Launch backward weight kernel
        launch_conv2d_backward_weight(
            input->d_data_, d_data_, d_grad_weight,
            batch, in_channels, out_channels,
            input_h, input_w, kernel_h, kernel_w,
            stride_h, stride_w, padding_h, padding_w,
            output_h, output_w, stream_
        );
        
        cudaStreamSynchronize(stream_);
        
        // Copy to host
        std::vector<float> host_data(grad_weight_size);
        cudaMemcpy(host_data.data(), d_grad_weight, grad_weight_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_grad_weight);
        
        // Convert to nested array
        Napi::Value resultData = buildNestedArray(env, host_data, grad_weight_shape, 0, 0);
        
        // Create Tensor
        Napi::FunctionReference* constructor = env.GetInstanceData<Napi::FunctionReference>();
        return constructor->New({ resultData });
    }
    
    // Conv2D Backward Bias
    // grad_output: this tensor [batch, out_channels, out_h, out_w]
    // Returns: grad_bias [out_channels]
    Napi::Value Conv2DBackwardBias(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        
        // grad_output shape: this tensor
        int batch = shape_[0];
        int out_channels = shape_[1];
        int output_h = shape_[2];
        int output_w = shape_[3];
        
        // Allocate grad_bias
        std::vector<int> grad_bias_shape = {out_channels};
        
        float* d_grad_bias;
        cudaMalloc(&d_grad_bias, out_channels * sizeof(float));
        
        // Launch backward bias kernel
        launch_conv2d_backward_bias(
            d_data_, d_grad_bias,
            batch, out_channels, output_h, output_w, stream_
        );
        
        cudaStreamSynchronize(stream_);
        
        // Copy to host
        std::vector<float> host_data(out_channels);
        cudaMemcpy(host_data.data(), d_grad_bias, out_channels * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_grad_bias);
        
        // Convert to nested array (1D)
        Napi::Array result = Napi::Array::New(env, out_channels);
        for (int i = 0; i < out_channels; i++) {
            result[i] = Napi::Number::New(env, host_data[i]);
        }
        
        // Create Tensor
        Napi::FunctionReference* constructor = env.GetInstanceData<Napi::FunctionReference>();
        return constructor->New({ result });
    }
    
    // ==================== MaxPool2D ====================
    
    // MaxPool2D
    // input: this tensor [batch, channels, height, width]
    // Returns: [batch, channels, out_h, out_w]
    Napi::Value MaxPool2D(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        
        // Parse arguments: kernel_size, stride, padding
        int kernel_h = info[0].As<Napi::Number>().Int32Value();
        int kernel_w = info[1].As<Napi::Number>().Int32Value();
        int stride_h = info[2].As<Napi::Number>().Int32Value();
        int stride_w = info[3].As<Napi::Number>().Int32Value();
        int padding_h = info[4].As<Napi::Number>().Int32Value();
        int padding_w = info[5].As<Napi::Number>().Int32Value();
        
        // Input shape: [batch, channels, height, width]
        if (shape_.size() != 4) {
            Napi::TypeError::New(env, "MaxPool2D input must be 4D [batch, channels, height, width]").ThrowAsJavaScriptException();
            return env.Null();
        }
        
        int batch = shape_[0];
        int channels = shape_[1];
        int input_h = shape_[2];
        int input_w = shape_[3];
        
        // Calculate output dimensions
        int output_h = (input_h + 2 * padding_h - kernel_h) / stride_h + 1;
        int output_w = (input_w + 2 * padding_w - kernel_w) / stride_w + 1;
        
        // Create output tensor
        std::vector<int> out_shape = {batch, channels, output_h, output_w};
        int out_size = batch * channels * output_h * output_w;
        
        // Allocate output GPU memory
        float* d_out;
        cudaMalloc(&d_out, out_size * sizeof(float));
        
        // Launch kernel
        launch_maxpool2d(
            d_data_, d_out,
            batch, channels,
            input_h, input_w, kernel_h, kernel_w,
            stride_h, stride_w, padding_h, padding_w,
            output_h, output_w, stream_
        );
        
        cudaStreamSynchronize(stream_);
        
        // Create result tensor from GPU data
        std::vector<float> host_data(out_size);
        cudaMemcpy(host_data.data(), d_out, out_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_out);
        
        // Convert to nested JS array
        Napi::Value resultData = buildNestedArray(env, host_data, out_shape, 0, 0);
        
        // Create Tensor
        Napi::FunctionReference* constructor = env.GetInstanceData<Napi::FunctionReference>();
        return constructor->New({ resultData });
    }
};

// ==================== Tensor Factory Functions ====================

// zeros factory
Napi::Value Zeros(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    
    if (info.Length() < 1 || !info[0].IsArray()) {
        Napi::TypeError::New(env, "Expected shape array").ThrowAsJavaScriptException();
        return env.Null();
    }
    
    Napi::Array shapeArr = info[0].As<Napi::Array>();
    std::vector<int> shape;
    int total = 1;
    
    for (uint32_t i = 0; i < shapeArr.Length(); i++) {
        int dim = shapeArr.Get(i).As<Napi::Number>().Int32Value();
        shape.push_back(dim);
        total *= dim;
    }
    
    // Create flat array of zeros
    std::vector<float> data(total, 0.0f);
    
    // Build nested array matching shape
    std::function<Napi::Value(int, int)> build;
    build = [&](int dim, int offset) -> Napi::Value {
        if (dim == shape.size() - 1) {
            Napi::Array arr = Napi::Array::New(env, shape[dim]);
            for (int i = 0; i < shape[dim]; i++) {
                arr[i] = Napi::Number::New(env, data[offset + i]);
            }
            return arr;
        } else {
            Napi::Array arr = Napi::Array::New(env, shape[dim]);
            int stride = 1;
            for (size_t i = dim + 1; i < shape.size(); i++) {
                stride *= shape[i];
            }
            for (int i = 0; i < shape[dim]; i++) {
                arr[i] = build(dim + 1, offset + i * stride);
            }
            return arr;
        }
    };
    
    Napi::Value dataArr = build(0, 0);
    
    // Create Tensor
    Napi::FunctionReference* constructor = env.GetInstanceData<Napi::FunctionReference>();
    return constructor->New({ dataArr });
}

// ones factory
Napi::Value Ones(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    
    if (info.Length() < 1 || !info[0].IsArray()) {
        Napi::TypeError::New(env, "Expected shape array").ThrowAsJavaScriptException();
        return env.Null();
    }
    
    Napi::Array shapeArr = info[0].As<Napi::Array>();
    std::vector<int> shape;
    int total = 1;
    
    for (uint32_t i = 0; i < shapeArr.Length(); i++) {
        int dim = shapeArr.Get(i).As<Napi::Number>().Int32Value();
        shape.push_back(dim);
        total *= dim;
    }
    
    // Create flat array of ones
    std::vector<float> data(total, 1.0f);
    
    // Build nested array matching shape
    std::function<Napi::Value(int, int)> build;
    build = [&](int dim, int offset) -> Napi::Value {
        if (dim == shape.size() - 1) {
            Napi::Array arr = Napi::Array::New(env, shape[dim]);
            for (int i = 0; i < shape[dim]; i++) {
                arr[i] = Napi::Number::New(env, data[offset + i]);
            }
            return arr;
        } else {
            Napi::Array arr = Napi::Array::New(env, shape[dim]);
            int stride = 1;
            for (size_t i = dim + 1; i < shape.size(); i++) {
                stride *= shape[i];
            }
            for (int i = 0; i < shape[dim]; i++) {
                arr[i] = build(dim + 1, offset + i * stride);
            }
            return arr;
        }
    };
    
    Napi::Value dataArr = build(0, 0);
    
    // Create Tensor
    Napi::FunctionReference* constructor = env.GetInstanceData<Napi::FunctionReference>();
    return constructor->New({ dataArr });
}

// ==================== Module Initialization ====================

// Cleanup function for module unload
static void CleanupCublas() {
    cleanup_cublas();
}

Napi::Object InitAll(Napi::Env env, Napi::Object exports) {
    // Initialize cuBLAS
    init_cublas();
    
    // Register cleanup on module unload
    env.AddCleanupHook(CleanupCublas);
    
    // Export Tensor class
    Tensor::Init(env, exports);
    
    // Export factory functions
    exports.Set("zeros", Napi::Function::New(env, Zeros));
    exports.Set("ones", Napi::Function::New(env, Ones));
    
    return exports;
}

NODE_API_MODULE(jstorch, InitAll)
