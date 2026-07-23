#include <napi.h>
#include "../core/tensor.hpp"
#include <vector>

using namespace jstorch;

// ==================== Helpers ====================
Napi::Value buildNestedArray(Napi::Env env, const std::vector<float>& data, const Shape& shape, int dim, int offset) {
    if (dim == (int)shape.size() - 1) {
        Napi::Array arr = Napi::Array::New(env, shape[dim]);
        for (int i = 0; i < shape[dim]; i++)
            arr[i] = Napi::Number::New(env, data[offset + i]);
        return arr;
    }
    int stride = 1;
    for (int i = dim + 1; i < (int)shape.size(); i++) stride *= shape[i];
    Napi::Array arr = Napi::Array::New(env, shape[dim]);
    for (int i = 0; i < shape[dim]; i++)
        arr[i] = buildNestedArray(env, data, shape, dim + 1, offset + i * stride);
    return arr;
}

void parseArray(Napi::Value val, std::vector<float>& data) {
    if (val.IsNumber()) data.push_back(val.As<Napi::Number>().FloatValue());
    else if (val.IsArray()) {
        Napi::Array arr = val.As<Napi::Array>();
        for (uint32_t i = 0; i < arr.Length(); i++) parseArray(arr[i], data);
    }
}

Shape inferShape(Napi::Value val) {
    if (val.IsNumber()) return {};
    Shape shape;
    Napi::Value cur = val;
    while (cur.IsArray()) {
        Napi::Array arr = cur.As<Napi::Array>();
        shape.push_back(arr.Length());
        cur = arr.Get(uint32_t(0));
    }
    return shape;
}

Shape parseShape(const Napi::CallbackInfo& info, int idx) {
    Napi::Array arr = info[idx].As<Napi::Array>();
    Shape s;
    for (uint32_t i = 0; i < arr.Length(); i++)
        s.push_back(arr.Get(i).As<Napi::Number>().Int32Value());
    return s;
}

// ==================== TensorWrap ====================
class TensorWrap : public Napi::ObjectWrap<TensorWrap> {
public: // data_ needs friend access from static methods
    Tensor tensor_;

    // Helper: wrap a C++ Tensor into a JS TensorWrap
    static Napi::Object Wrap(Napi::Env env, Tensor t) {
        auto* ctor = env.GetInstanceData<Napi::FunctionReference>();
        Napi::Object obj = ctor->New({});
        Napi::ObjectWrap<TensorWrap>::Unwrap(obj)->tensor_ = std::move(t);
        return obj;
    }

    static Napi::Object Init(Napi::Env env, Napi::Object exports) {
        Napi::Function func = DefineClass(env, "Tensor", {
            // Accessors
            InstanceAccessor("shape", &TensorWrap::GetShape, nullptr),
            InstanceAccessor("ndim", &TensorWrap::GetNdim, nullptr),
            InstanceMethod("toArray", &TensorWrap::ToArray),
            
            // View ops
            InstanceMethod("reshape", &TensorWrap::Reshape),
            InstanceMethod("squeeze", &TensorWrap::Squeeze),
            InstanceMethod("unsqueeze", &TensorWrap::Unsqueeze),
            InstanceMethod("transpose", &TensorWrap::Transpose),
            InstanceMethod("slice", &TensorWrap::Slice),
            InstanceMethod("contiguous", &TensorWrap::Contiguous),
            InstanceMethod("clone", &TensorWrap::Clone),
            InstanceMethod("flatten", &TensorWrap::Flatten),
            
            // Unary ops
            InstanceMethod("abs", &TensorWrap::abs),
            InstanceMethod("sqrt", &TensorWrap::sqrt_),
            InstanceMethod("square", &TensorWrap::square),
            InstanceMethod("exp", &TensorWrap::exp_),
            InstanceMethod("log", &TensorWrap::log_),
            InstanceMethod("sin", &TensorWrap::sin_),
            InstanceMethod("cos", &TensorWrap::cos_),
            InstanceMethod("neg", &TensorWrap::neg),
            InstanceMethod("floor", &TensorWrap::floor_),
            InstanceMethod("ceil", &TensorWrap::ceil_),
            InstanceMethod("round", &TensorWrap::round_),
            InstanceMethod("sigmoid", &TensorWrap::sigmoid),
            InstanceMethod("tanh", &TensorWrap::tanh_),
            InstanceMethod("relu", &TensorWrap::relu),
            InstanceMethod("silu", &TensorWrap::silu),
            InstanceMethod("gelu", &TensorWrap::gelu),
            InstanceMethod("softplus", &TensorWrap::softplus),
            InstanceMethod("leaky_relu", &TensorWrap::leaky_relu),
            InstanceMethod("clamp", &TensorWrap::Clamp),
            InstanceMethod("clamp_min", &TensorWrap::ClampMin),
            InstanceMethod("clamp_max", &TensorWrap::ClampMax),
            InstanceMethod("fmod", &TensorWrap::Fmod),
            InstanceMethod("log1p", &TensorWrap::log1p_),
            InstanceMethod("reciprocal", &TensorWrap::reciprocal_),
            InstanceMethod("sign", &TensorWrap::sign_),
            InstanceMethod("pow_scalar", &TensorWrap::PowScalar),
            InstanceMethod("mul_scalar", &TensorWrap::MulScalar),
            InstanceMethod("add_scalar", &TensorWrap::AddScalar),
            
            // Binary ops
            InstanceMethod("add", &TensorWrap::add),
            InstanceMethod("sub", &TensorWrap::sub),
            InstanceMethod("mul", &TensorWrap::mul),
            InstanceMethod("div", &TensorWrap::div_),
            InstanceMethod("maximum", &TensorWrap::maximum),
            InstanceMethod("minimum", &TensorWrap::minimum),
            InstanceMethod("pow", &TensorWrap::pow_),
            InstanceMethod("gt", &TensorWrap::gt),
            InstanceMethod("lt", &TensorWrap::lt),
            InstanceMethod("ge", &TensorWrap::ge),
            InstanceMethod("le", &TensorWrap::le),
            InstanceMethod("eq", &TensorWrap::eq_),
            InstanceMethod("ne", &TensorWrap::ne_),
            InstanceMethod("matmul", &TensorWrap::Matmul),
            
            // Reduce
            InstanceMethod("sum", &TensorWrap::Sum),
            InstanceMethod("mean", &TensorWrap::Mean),
            InstanceMethod("max", &TensorWrap::Max),
            InstanceMethod("min", &TensorWrap::Min),
            InstanceMethod("argmax", &TensorWrap::Argmax),
            InstanceMethod("argmin", &TensorWrap::Argmin),
            
            // Tensor ops
            InstanceMethod("flip", &TensorWrap::Flip),
            InstanceMethod("pad", &TensorWrap::Pad),
            InstanceMethod("cumsum", &TensorWrap::Cumsum),
            InstanceMethod("embedding", &TensorWrap::Embedding),
            InstanceMethod("conv1d", &TensorWrap::Conv1d),
            InstanceMethod("conv_transpose1d", &TensorWrap::ConvTranspose1d),
            InstanceMethod("interpolate", &TensorWrap::Interpolate),
            InstanceMethod("randn_like", &TensorWrap::RandnLike),
            InstanceMethod("conv2d", &TensorWrap::Conv2d_),
            InstanceMethod("conv_transpose2d", &TensorWrap::ConvTranspose2d_),
            InstanceMethod("avg_pool2d", &TensorWrap::AvgPool2d),
            
            // Backward ops
            InstanceMethod("scatter_add", &TensorWrap::ScatterAdd),
            InstanceMethod("interp1d_backward", &TensorWrap::Interp1dBackward),
            InstanceMethod("avgpool2d_backward", &TensorWrap::AvgPool2dBackward),
            StaticMethod("conv1dBackwardWeight", &TensorWrap::Conv1dBackwardWeight),
            StaticMethod("conv2dBackwardWeight", &TensorWrap::Conv2dBackwardWeight),
            StaticMethod("convTranspose1dBackwardWeight", &TensorWrap::ConvTranspose1dBackwardWeight),
            StaticMethod("convTranspose2dBackwardWeight", &TensorWrap::ConvTranspose2dBackwardWeight),
            
            // Static methods
            StaticMethod("cat", &TensorWrap::Cat),
            StaticMethod("where", &TensorWrap::Where),
            StaticMethod("fromBuffer", &TensorWrap::FromBuffer),
            StaticMethod("randn", &TensorWrap::Randn),
            StaticMethod("fromIntArray", &TensorWrap::FromIntArray),
            StaticMethod("full", &TensorWrap::Full),
            StaticMethod("arange", &TensorWrap::Arange),
        });
        
        auto* ctor = new Napi::FunctionReference();
        *ctor = Napi::Persistent(func);
        env.SetInstanceData(ctor);
        exports.Set("Tensor", func);
        return exports;
    }
    
    TensorWrap(const Napi::CallbackInfo& info)
        : Napi::ObjectWrap<TensorWrap>(info), tensor_(Shape{1}) {
        if (info.Length() > 0 && info[0].IsArray()) {
            std::vector<float> data;
            parseArray(info[0], data);
            Shape shape = inferShape(info[0]);
            tensor_ = Tensor::from_array(data.data(), shape);
        }
    }
    
    Tensor& tensor() { return tensor_; }

    // ==================== Accessors ====================
    Napi::Value GetShape(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        Napi::Array arr = Napi::Array::New(env, tensor_.shape().size());
        for (size_t i = 0; i < tensor_.shape().size(); i++)
            arr[i] = Napi::Number::New(env, tensor_.shape()[i]);
        return arr;
    }
    
    Napi::Value GetNdim(const Napi::CallbackInfo& info) {
        return Napi::Number::New(info.Env(), tensor_.ndim());
    }
    
    Napi::Value ToArray(const Napi::CallbackInfo& info) {
        auto data = tensor_.to_array();
        return buildNestedArray(info.Env(), data, tensor_.shape(), 0, 0);
    }

    // ==================== View ops ====================
    Napi::Value Reshape(const Napi::CallbackInfo& info) {
        return Wrap(info.Env(), tensor_.reshape(parseShape(info, 0)));
    }
    Napi::Value Squeeze(const Napi::CallbackInfo& info) {
        int axis = info.Length() > 0 ? info[0].As<Napi::Number>().Int32Value() : -1;
        return Wrap(info.Env(), tensor_.squeeze(axis));
    }
    Napi::Value Unsqueeze(const Napi::CallbackInfo& info) {
        return Wrap(info.Env(), tensor_.unsqueeze(info[0].As<Napi::Number>().Int32Value()));
    }
    Napi::Value Transpose(const Napi::CallbackInfo& info) {
        if (info.Length() >= 2)
            return Wrap(info.Env(), tensor_.transpose(
                info[0].As<Napi::Number>().Int32Value(),
                info[1].As<Napi::Number>().Int32Value()));
        return Wrap(info.Env(), tensor_.transpose());
    }
    Napi::Value Slice(const Napi::CallbackInfo& info) {
        int dim = info[0].As<Napi::Number>().Int32Value();
        int start = info[1].As<Napi::Number>().Int32Value();
        int end = info[2].As<Napi::Number>().Int32Value();
        return Wrap(info.Env(), tensor_.slice(dim, start, end));
    }
    Napi::Value Contiguous(const Napi::CallbackInfo& info) {
        return Wrap(info.Env(), tensor_.contiguous());
    }
    Napi::Value Clone(const Napi::CallbackInfo& info) {
        return Wrap(info.Env(), tensor_.clone());
    }

    // ==================== Unary ops (macro) ====================
    #define DEF_UNARY(jsName, cppName) \
    Napi::Value jsName(const Napi::CallbackInfo& info) { return Wrap(info.Env(), tensor_.cppName()); }
    
    DEF_UNARY(abs, abs)       DEF_UNARY(sqrt_, sqrt)     DEF_UNARY(square, square)
    DEF_UNARY(exp_, exp)      DEF_UNARY(log_, log)       DEF_UNARY(sin_, sin)
    DEF_UNARY(cos_, cos)      DEF_UNARY(neg, neg)        DEF_UNARY(floor_, floor)
    DEF_UNARY(ceil_, ceil)    DEF_UNARY(round_, round)   DEF_UNARY(sigmoid, sigmoid)
    DEF_UNARY(tanh_, tanh)    DEF_UNARY(relu, relu)      DEF_UNARY(silu, silu)
    DEF_UNARY(gelu, gelu)     DEF_UNARY(softplus, softplus)
    DEF_UNARY(log1p_, log1p)  DEF_UNARY(reciprocal_, reciprocal) DEF_UNARY(sign_, sign)
    DEF_UNARY(RandnLike, randn_like)
    
    Napi::Value leaky_relu(const Napi::CallbackInfo& info) {
        float slope = info.Length() > 0 ? info[0].As<Napi::Number>().FloatValue() : 0.01f;
        return Wrap(info.Env(), tensor_.leaky_relu(slope));
    }
    Napi::Value Clamp(const Napi::CallbackInfo& info) {
        float lo = info[0].As<Napi::Number>().FloatValue();
        float hi = info[1].As<Napi::Number>().FloatValue();
        return Wrap(info.Env(), tensor_.clamp(lo, hi));
    }
    Napi::Value ClampMin(const Napi::CallbackInfo& info) {
        return Wrap(info.Env(), tensor_.clamp_min(info[0].As<Napi::Number>().FloatValue()));
    }
    Napi::Value ClampMax(const Napi::CallbackInfo& info) {
        return Wrap(info.Env(), tensor_.clamp_max(info[0].As<Napi::Number>().FloatValue()));
    }
    Napi::Value Fmod(const Napi::CallbackInfo& info) {
        return Wrap(info.Env(), tensor_.fmod(info[0].As<Napi::Number>().FloatValue()));
    }
    Napi::Value PowScalar(const Napi::CallbackInfo& info) {
        return Wrap(info.Env(), tensor_.pow_scalar(info[0].As<Napi::Number>().FloatValue()));
    }
    Napi::Value MulScalar(const Napi::CallbackInfo& info) {
        return Wrap(info.Env(), tensor_.mul_scalar(info[0].As<Napi::Number>().FloatValue()));
    }
    Napi::Value AddScalar(const Napi::CallbackInfo& info) {
        return Wrap(info.Env(), tensor_.add_scalar(info[0].As<Napi::Number>().FloatValue()));
    }

    // ==================== Binary ops (macro) ====================
    #define DEF_BINARY(jsName, cppName) \
    Napi::Value jsName(const Napi::CallbackInfo& info) { \
        if (info[0].IsNumber()) { \
            float s = info[0].As<Napi::Number>().FloatValue(); \
            auto scalar = Tensor::from_array(&s, {1}); \
            return Wrap(info.Env(), tensor_.cppName(scalar)); \
        } \
        auto* other = Napi::ObjectWrap<TensorWrap>::Unwrap(info[0].As<Napi::Object>()); \
        return Wrap(info.Env(), tensor_.cppName(other->tensor())); \
    }
    
    // Optimized scalar ops: mul/add/pow use dedicated kernels
    Napi::Value mul(const Napi::CallbackInfo& info) {
        if (info[0].IsNumber())
            return Wrap(info.Env(), tensor_.mul_scalar(info[0].As<Napi::Number>().FloatValue()));
        auto* o = Napi::ObjectWrap<TensorWrap>::Unwrap(info[0].As<Napi::Object>());
        return Wrap(info.Env(), tensor_.mul(o->tensor()));
    }
    Napi::Value add(const Napi::CallbackInfo& info) {
        if (info[0].IsNumber())
            return Wrap(info.Env(), tensor_.add_scalar(info[0].As<Napi::Number>().FloatValue()));
        auto* o = Napi::ObjectWrap<TensorWrap>::Unwrap(info[0].As<Napi::Object>());
        return Wrap(info.Env(), tensor_.add(o->tensor()));
    }
    Napi::Value sub(const Napi::CallbackInfo& info) {
        if (info[0].IsNumber())
            return Wrap(info.Env(), tensor_.add_scalar(-info[0].As<Napi::Number>().FloatValue()));
        auto* o = Napi::ObjectWrap<TensorWrap>::Unwrap(info[0].As<Napi::Object>());
        return Wrap(info.Env(), tensor_.sub(o->tensor()));
    }
    Napi::Value div_(const Napi::CallbackInfo& info) {
        if (info[0].IsNumber())
            return Wrap(info.Env(), tensor_.mul_scalar(1.0f / info[0].As<Napi::Number>().FloatValue()));
        auto* o = Napi::ObjectWrap<TensorWrap>::Unwrap(info[0].As<Napi::Object>());
        return Wrap(info.Env(), tensor_.div(o->tensor()));
    }
    Napi::Value pow_(const Napi::CallbackInfo& info) {
        if (info[0].IsNumber())
            return Wrap(info.Env(), tensor_.pow_scalar(info[0].As<Napi::Number>().FloatValue()));
        auto* o = Napi::ObjectWrap<TensorWrap>::Unwrap(info[0].As<Napi::Object>());
        return Wrap(info.Env(), tensor_.pow(o->tensor()));
    }
    
    DEF_BINARY(maximum, maximum) DEF_BINARY(minimum, minimum)
    DEF_BINARY(gt, gt)       DEF_BINARY(lt, lt)
    DEF_BINARY(ge, ge)       DEF_BINARY(le, le)       DEF_BINARY(eq_, eq)
    DEF_BINARY(ne_, ne)      DEF_BINARY(Matmul, matmul)

    // ==================== Reduce ====================
    Napi::Value Sum(const Napi::CallbackInfo& info) {
        int dim = info.Length() > 0 ? info[0].As<Napi::Number>().Int32Value() : -1;
        bool keepdim = info.Length() > 1 ? info[1].As<Napi::Boolean>().Value() : false;
        return Wrap(info.Env(), tensor_.sum(dim, keepdim));
    }
    Napi::Value Mean(const Napi::CallbackInfo& info) {
        int dim = info.Length() > 0 ? info[0].As<Napi::Number>().Int32Value() : -1;
        bool keepdim = info.Length() > 1 ? info[1].As<Napi::Boolean>().Value() : false;
        return Wrap(info.Env(), tensor_.mean(dim, keepdim));
    }

    Napi::Value Max(const Napi::CallbackInfo& info) {
        int dim = info[0].As<Napi::Number>().Int32Value();
        bool keepdim = info.Length() > 1 ? info[1].As<Napi::Boolean>().Value() : false;
        return Wrap(info.Env(), tensor_.max(dim, keepdim));
    }
    Napi::Value Min(const Napi::CallbackInfo& info) {
        int dim = info[0].As<Napi::Number>().Int32Value();
        bool keepdim = info.Length() > 1 ? info[1].As<Napi::Boolean>().Value() : false;
        return Wrap(info.Env(), tensor_.min(dim, keepdim));
    }
    Napi::Value Argmax(const Napi::CallbackInfo& info) {
        return Wrap(info.Env(), tensor_.argmax(info[0].As<Napi::Number>().Int32Value()));
    }
    Napi::Value Argmin(const Napi::CallbackInfo& info) {
        return Wrap(info.Env(), tensor_.argmin(info[0].As<Napi::Number>().Int32Value()));
    }
    Napi::Value Flatten(const Napi::CallbackInfo& info) {
        int start = info[0].As<Napi::Number>().Int32Value();
        int end = info.Length() > 1 ? info[1].As<Napi::Number>().Int32Value() : -1;
        return Wrap(info.Env(), tensor_.flatten(start, end));
    }

    // ==================== Tensor ops ====================
    Napi::Value Flip(const Napi::CallbackInfo& info) {
        return Wrap(info.Env(), tensor_.flip(info[0].As<Napi::Number>().Int32Value()));
    }
    
    Napi::Value Pad(const Napi::CallbackInfo& info) {
        Napi::Array arr = info[0].As<Napi::Array>();
        std::vector<int> padding;
        for (uint32_t i = 0; i < arr.Length(); i++)
            padding.push_back(arr.Get(i).As<Napi::Number>().Int32Value());
        int mode = info.Length() > 1 ? info[1].As<Napi::Number>().Int32Value() : 0;
        float value = info.Length() > 2 ? info[2].As<Napi::Number>().FloatValue() : 0.0f;
        return Wrap(info.Env(), tensor_.pad(padding, mode, value));
    }
    
    Napi::Value Cumsum(const Napi::CallbackInfo& info) {
        return Wrap(info.Env(), tensor_.cumsum(info[0].As<Napi::Number>().Int32Value()));
    }
    
    Napi::Value Embedding(const Napi::CallbackInfo& info) {
        auto* idx = Napi::ObjectWrap<TensorWrap>::Unwrap(info[0].As<Napi::Object>());
        return Wrap(info.Env(), tensor_.embedding(idx->tensor()));
    }
    
    Napi::Value Conv1d(const Napi::CallbackInfo& info) {
        auto* w = Napi::ObjectWrap<TensorWrap>::Unwrap(info[0].As<Napi::Object>());
        const Tensor* bias_ptr = nullptr;
        Tensor bias_tensor(Shape{1});
        if (!info[1].IsNull() && !info[1].IsUndefined()) {
            auto* bw = Napi::ObjectWrap<TensorWrap>::Unwrap(info[1].As<Napi::Object>());
            bias_tensor = bw->tensor();
            bias_ptr = &bias_tensor;
        }
        int stride = info[2].As<Napi::Number>().Int32Value();
        int padding = info[3].As<Napi::Number>().Int32Value();
        int dilation = info[4].As<Napi::Number>().Int32Value();
        int groups = info[5].As<Napi::Number>().Int32Value();
        return Wrap(info.Env(), tensor_.conv1d(w->tensor(), bias_ptr, stride, padding, dilation, groups));
    }
    
    Napi::Value ConvTranspose1d(const Napi::CallbackInfo& info) {
        auto* w = Napi::ObjectWrap<TensorWrap>::Unwrap(info[0].As<Napi::Object>());
        const Tensor* bias_ptr = nullptr;
        Tensor bias_tensor(Shape{1});
        if (!info[1].IsNull() && !info[1].IsUndefined()) {
            auto* bw = Napi::ObjectWrap<TensorWrap>::Unwrap(info[1].As<Napi::Object>());
            bias_tensor = bw->tensor();
            bias_ptr = &bias_tensor;
        }
        int stride = info[2].As<Napi::Number>().Int32Value();
        int padding = info[3].As<Napi::Number>().Int32Value();
        int output_padding = info[4].As<Napi::Number>().Int32Value();
        int dilation = info[5].As<Napi::Number>().Int32Value();
        int groups = info[6].As<Napi::Number>().Int32Value();
        return Wrap(info.Env(), tensor_.conv_transpose1d(w->tensor(), bias_ptr, stride, padding, output_padding, dilation, groups));
    }
    
    Napi::Value Conv2d_(const Napi::CallbackInfo& info) {
        auto* w = Napi::ObjectWrap<TensorWrap>::Unwrap(info[0].As<Napi::Object>());
        const Tensor* bias_ptr = nullptr; Tensor bias_t(Shape{1});
        if (!info[1].IsNull() && !info[1].IsUndefined()) {
            bias_t = Napi::ObjectWrap<TensorWrap>::Unwrap(info[1].As<Napi::Object>())->tensor();
            bias_ptr = &bias_t;
        }
        int sH = info[2].As<Napi::Number>().Int32Value();
        int sW = info[3].As<Napi::Number>().Int32Value();
        int pH = info[4].As<Napi::Number>().Int32Value();
        int pW = info[5].As<Napi::Number>().Int32Value();
        int dH = info[6].As<Napi::Number>().Int32Value();
        int dW = info[7].As<Napi::Number>().Int32Value();
        int groups = info[8].As<Napi::Number>().Int32Value();
        return Wrap(info.Env(), tensor_.conv2d(w->tensor(), bias_ptr, sH, sW, pH, pW, dH, dW, groups));
    }
    
    Napi::Value ConvTranspose2d_(const Napi::CallbackInfo& info) {
        auto* w = Napi::ObjectWrap<TensorWrap>::Unwrap(info[0].As<Napi::Object>());
        const Tensor* bias_ptr = nullptr; Tensor bias_t(Shape{1});
        if (!info[1].IsNull() && !info[1].IsUndefined()) {
            bias_t = Napi::ObjectWrap<TensorWrap>::Unwrap(info[1].As<Napi::Object>())->tensor();
            bias_ptr = &bias_t;
        }
        int sH = info[2].As<Napi::Number>().Int32Value();
        int sW = info[3].As<Napi::Number>().Int32Value();
        int pH = info[4].As<Napi::Number>().Int32Value();
        int pW = info[5].As<Napi::Number>().Int32Value();
        int opH = info[6].As<Napi::Number>().Int32Value();
        int opW = info[7].As<Napi::Number>().Int32Value();
        int dH = info[8].As<Napi::Number>().Int32Value();
        int dW = info[9].As<Napi::Number>().Int32Value();
        int groups = info[10].As<Napi::Number>().Int32Value();
        return Wrap(info.Env(), tensor_.conv_transpose2d(w->tensor(), bias_ptr, sH, sW, pH, pW, opH, opW, dH, dW, groups));
    }
    
    Napi::Value AvgPool2d(const Napi::CallbackInfo& info) {
        int kH = info[0].As<Napi::Number>().Int32Value();
        int kW = info[1].As<Napi::Number>().Int32Value();
        int sH = info[2].As<Napi::Number>().Int32Value();
        int sW = info[3].As<Napi::Number>().Int32Value();
        int pH = info[4].As<Napi::Number>().Int32Value();
        int pW = info[5].As<Napi::Number>().Int32Value();
        bool cip = info.Length() > 6 ? info[6].As<Napi::Boolean>().Value() : true;
        return Wrap(info.Env(), tensor_.avg_pool2d(kH, kW, sH, sW, pH, pW, cip));
    }
    
    Napi::Value Interpolate(const Napi::CallbackInfo& info) {
        int target = info[0].As<Napi::Number>().Int32Value();
        int mode = info.Length() > 1 ? info[1].As<Napi::Number>().Int32Value() : 0;
        bool align = info.Length() > 2 ? info[2].As<Napi::Boolean>().Value() : false;
        return Wrap(info.Env(), tensor_.interpolate(target, mode, align));
    }

    // ==================== Backward ops ====================
    Napi::Value ScatterAdd(const Napi::CallbackInfo& info) {
        auto* grad = Napi::ObjectWrap<TensorWrap>::Unwrap(info[0].As<Napi::Object>());
        auto* indices = Napi::ObjectWrap<TensorWrap>::Unwrap(info[1].As<Napi::Object>());
        int vocab_size = info[2].As<Napi::Number>().Int32Value();
        return Wrap(info.Env(), tensor_.scatter_add(grad->tensor(), indices->tensor(), vocab_size));
    }
    Napi::Value Interp1dBackward(const Napi::CallbackInfo& info) {
        int in_len = info[0].As<Napi::Number>().Int32Value();
        int mode = info[1].As<Napi::Number>().Int32Value();
        bool align = info.Length() > 2 ? info[2].As<Napi::Boolean>().Value() : false;
        return Wrap(info.Env(), tensor_.interp1d_backward(in_len, mode, align));
    }
    Napi::Value AvgPool2dBackward(const Napi::CallbackInfo& info) {
        int H = info[0].As<Napi::Number>().Int32Value();
        int W = info[1].As<Napi::Number>().Int32Value();
        int kH = info[2].As<Napi::Number>().Int32Value();
        int kW = info[3].As<Napi::Number>().Int32Value();
        int sH = info[4].As<Napi::Number>().Int32Value();
        int sW = info[5].As<Napi::Number>().Int32Value();
        int pH = info[6].As<Napi::Number>().Int32Value();
        int pW = info[7].As<Napi::Number>().Int32Value();
        bool cip = info.Length() > 8 ? info[8].As<Napi::Boolean>().Value() : true;
        return Wrap(info.Env(), tensor_.avgpool2d_backward(H, W, kH, kW, sH, sW, pH, pW, cip));
    }
    static Napi::Value Conv1dBackwardWeight(const Napi::CallbackInfo& info) {
        auto* input = Napi::ObjectWrap<TensorWrap>::Unwrap(info[0].As<Napi::Object>());
        auto* grad_out = Napi::ObjectWrap<TensorWrap>::Unwrap(info[1].As<Napi::Object>());
        int C_in_g = info[2].As<Napi::Number>().Int32Value();
        int K = info[3].As<Napi::Number>().Int32Value();
        int stride = info[4].As<Napi::Number>().Int32Value();
        int padding = info[5].As<Napi::Number>().Int32Value();
        int dilation = info[6].As<Napi::Number>().Int32Value();
        int groups = info[7].As<Napi::Number>().Int32Value();
        return Wrap(info.Env(), Tensor::conv1d_backward_weight(
            input->tensor(), grad_out->tensor(), C_in_g, K, stride, padding, dilation, groups));
    }
    static Napi::Value ConvTranspose1dBackwardWeight(const Napi::CallbackInfo& info) {
        auto* input = Napi::ObjectWrap<TensorWrap>::Unwrap(info[0].As<Napi::Object>());
        auto* grad_out = Napi::ObjectWrap<TensorWrap>::Unwrap(info[1].As<Napi::Object>());
        int C_out_g = info[2].As<Napi::Number>().Int32Value();
        int K = info[3].As<Napi::Number>().Int32Value();
        int stride = info[4].As<Napi::Number>().Int32Value();
        int padding = info[5].As<Napi::Number>().Int32Value();
        int dilation = info[6].As<Napi::Number>().Int32Value();
        int groups = info[7].As<Napi::Number>().Int32Value();
        return Wrap(info.Env(), Tensor::conv_transpose1d_backward_weight(
            input->tensor(), grad_out->tensor(), C_out_g, K, stride, padding, dilation, groups));
    }
    static Napi::Value ConvTranspose2dBackwardWeight(const Napi::CallbackInfo& info) {
        auto* input = Napi::ObjectWrap<TensorWrap>::Unwrap(info[0].As<Napi::Object>());
        auto* grad_out = Napi::ObjectWrap<TensorWrap>::Unwrap(info[1].As<Napi::Object>());
        int Co_g = info[2].As<Napi::Number>().Int32Value();
        int kH = info[3].As<Napi::Number>().Int32Value();
        int kW = info[4].As<Napi::Number>().Int32Value();
        int sH = info[5].As<Napi::Number>().Int32Value();
        int sW = info[6].As<Napi::Number>().Int32Value();
        int pH = info[7].As<Napi::Number>().Int32Value();
        int pW = info[8].As<Napi::Number>().Int32Value();
        int dH = info[9].As<Napi::Number>().Int32Value();
        int dW = info[10].As<Napi::Number>().Int32Value();
        int groups = info[11].As<Napi::Number>().Int32Value();
        return Wrap(info.Env(), Tensor::conv_transpose2d_backward_weight(
            input->tensor(), grad_out->tensor(), Co_g, kH, kW, sH, sW, pH, pW, dH, dW, groups));
    }
    static Napi::Value Conv2dBackwardWeight(const Napi::CallbackInfo& info) {
        auto* input = Napi::ObjectWrap<TensorWrap>::Unwrap(info[0].As<Napi::Object>());
        auto* grad_out = Napi::ObjectWrap<TensorWrap>::Unwrap(info[1].As<Napi::Object>());
        int Ci_g = info[2].As<Napi::Number>().Int32Value();
        int kH = info[3].As<Napi::Number>().Int32Value();
        int kW = info[4].As<Napi::Number>().Int32Value();
        int sH = info[5].As<Napi::Number>().Int32Value();
        int sW = info[6].As<Napi::Number>().Int32Value();
        int pH = info[7].As<Napi::Number>().Int32Value();
        int pW = info[8].As<Napi::Number>().Int32Value();
        int dH = info[9].As<Napi::Number>().Int32Value();
        int dW = info[10].As<Napi::Number>().Int32Value();
        int groups = info[11].As<Napi::Number>().Int32Value();
        return Wrap(info.Env(), Tensor::conv2d_backward_weight(
            input->tensor(), grad_out->tensor(), Ci_g, kH, kW, sH, sW, pH, pW, dH, dW, groups));
    }

    // ==================== Static methods ====================
    static Napi::Value Cat(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        Napi::Array arr = info[0].As<Napi::Array>();
        std::vector<Tensor> tensors;
        for (uint32_t i = 0; i < arr.Length(); i++)
            tensors.push_back(Napi::ObjectWrap<TensorWrap>::Unwrap(arr.Get(i).As<Napi::Object>())->tensor());
        int dim = info.Length() > 1 ? info[1].As<Napi::Number>().Int32Value() : 0;
        return Wrap(env, Tensor::cat(tensors, dim));
    }
    
    static Napi::Value Where(const Napi::CallbackInfo& info) {
        auto* c = Napi::ObjectWrap<TensorWrap>::Unwrap(info[0].As<Napi::Object>());
        auto* x = Napi::ObjectWrap<TensorWrap>::Unwrap(info[1].As<Napi::Object>());
        auto* y = Napi::ObjectWrap<TensorWrap>::Unwrap(info[2].As<Napi::Object>());
        return Wrap(info.Env(), Tensor::where(c->tensor(), x->tensor(), y->tensor()));
    }
    
    static Napi::Value FromBuffer(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        Napi::Float32Array f32 = info[0].As<Napi::Float32Array>();
        Shape shape = parseShape(info, 1);
        return Wrap(env, Tensor::from_buffer(f32.Data(), (int)f32.ElementLength(), shape));
    }
    
    static Napi::Value Randn(const Napi::CallbackInfo& info) {
        return Wrap(info.Env(), Tensor::randn(parseShape(info, 0)));
    }
    
    static Napi::Value Full(const Napi::CallbackInfo& info) {
        Shape shape = parseShape(info, 0);
        float value = info[1].As<Napi::Number>().FloatValue();
        return Wrap(info.Env(), Tensor::full(shape, value));
    }
    
    static Napi::Value Arange(const Napi::CallbackInfo& info) {
        float start = info[0].As<Napi::Number>().FloatValue();
        float end = info[1].As<Napi::Number>().FloatValue();
        float step = info.Length() > 2 ? info[2].As<Napi::Number>().FloatValue() : 1.0f;
        return Wrap(info.Env(), Tensor::arange(start, end, step));
    }
    
    static Napi::Value FromIntArray(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        Napi::Array arr = info[0].As<Napi::Array>();
        std::vector<int> data;
        for (uint32_t i = 0; i < arr.Length(); i++)
            data.push_back(arr.Get(i).As<Napi::Number>().Int32Value());
        Shape shape = info.Length() > 1 ? parseShape(info, 1) : Shape{(int)data.size()};
        return Wrap(env, Tensor::from_int_array(data.data(), shape));
    }
};

Napi::Object Init(Napi::Env env, Napi::Object exports) {
    return TensorWrap::Init(env, exports);
}

NODE_API_MODULE(jstorch, Init)
