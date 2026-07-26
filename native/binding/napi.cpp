#include <napi.h>
#include "../core/tensor.hpp"
#include "../core/autograd.hpp"
#include <vector>



using namespace jstorch;

// Store both constructors
struct CtorRefs {
    Napi::FunctionReference tensor_ctor;
    Napi::FunctionReference grad_ctor;
};

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
    napi_env raw_env_ = nullptr;  // raw env for finalizer use
    int64_t ext_bytes_ = 0;      // GPU bytes reported to V8 GC

    // Helper: wrap a C++ Tensor into a JS TensorWrap
    static Napi::Object Wrap(Napi::Env env, Tensor t) {
        int64_t bytes = (int64_t)t.size() * 4;  // float32 = 4 bytes
        auto* refs = env.GetInstanceData<CtorRefs>();
        Napi::Object obj = refs->tensor_ctor.New({});
        auto* tw = Napi::ObjectWrap<TensorWrap>::Unwrap(obj);
        tw->tensor_ = std::move(t);
        tw->raw_env_ = env;
        tw->ext_bytes_ = bytes;
        if (bytes > 0) {
            int64_t unused;
            napi_adjust_external_memory(env, bytes, &unused);
        }
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
            InstanceMethod("max_pool2d", &TensorWrap::MaxPool2d),
            
            // Backward ops
            InstanceMethod("scatter_add", &TensorWrap::ScatterAdd),
            InstanceMethod("interp1d_backward", &TensorWrap::Interp1dBackward),
            InstanceMethod("avgpool2d_backward", &TensorWrap::AvgPool2dBackward),
            StaticMethod("maxpool2dBackward", &TensorWrap::MaxPool2dBackward),
            StaticMethod("conv1dBackwardWeight", &TensorWrap::Conv1dBackwardWeight),
            StaticMethod("conv2dBackwardWeight", &TensorWrap::Conv2dBackwardWeight),
            StaticMethod("convTranspose1dBackwardWeight", &TensorWrap::ConvTranspose1dBackwardWeight),
            StaticMethod("convTranspose2dBackwardWeight", &TensorWrap::ConvTranspose2dBackwardWeight),
            
            // Static methods
            StaticMethod("rnnForward", &TensorWrap::RnnForward),
            StaticMethod("rnnBackward", &TensorWrap::RnnBackward),
            StaticMethod("cat", &TensorWrap::Cat),
            StaticMethod("where", &TensorWrap::Where),
            StaticMethod("fromBuffer", &TensorWrap::FromBuffer),
            StaticMethod("randn", &TensorWrap::Randn),
            StaticMethod("fromIntArray", &TensorWrap::FromIntArray),
            StaticMethod("full", &TensorWrap::Full),
            StaticMethod("arange", &TensorWrap::Arange),
            StaticMethod("adamStep", &TensorWrap::AdamStep),
            StaticMethod("clearCache", &TensorWrap::ClearCache),
        });
        
        auto* refs = env.GetInstanceData<CtorRefs>();
        if (!refs) {
            refs = new CtorRefs();
            env.SetInstanceData(refs);
        }
        refs->tensor_ctor = Napi::Persistent(func);
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
    
    ~TensorWrap() {
        if (ext_bytes_ > 0 && raw_env_) {
            int64_t unused;
            napi_adjust_external_memory(raw_env_, -ext_bytes_, &unused);
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
    
    Napi::Value MaxPool2d(const Napi::CallbackInfo& info) {
        int kH = info[0].As<Napi::Number>().Int32Value();
        int kW = info[1].As<Napi::Number>().Int32Value();
        int sH = info[2].As<Napi::Number>().Int32Value();
        int sW = info[3].As<Napi::Number>().Int32Value();
        int pH = info[4].As<Napi::Number>().Int32Value();
        int pW = info[5].As<Napi::Number>().Int32Value();
        auto [out, indices] = tensor_.max_pool2d(kH, kW, sH, sW, pH, pW);
        Napi::Env env = info.Env();
        Napi::Array result = Napi::Array::New(env, 2);
        result[(uint32_t)0] = Wrap(env, std::move(out));
        result[1u] = Wrap(env, std::move(indices));
        return result;
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
    static Napi::Value MaxPool2dBackward(const Napi::CallbackInfo& info) {
        auto* grad = Napi::ObjectWrap<TensorWrap>::Unwrap(info[0].As<Napi::Object>());
        auto* indices = Napi::ObjectWrap<TensorWrap>::Unwrap(info[1].As<Napi::Object>());
        int B = info[2].As<Napi::Number>().Int32Value();
        int C = info[3].As<Napi::Number>().Int32Value();
        int H = info[4].As<Napi::Number>().Int32Value();
        int W = info[5].As<Napi::Number>().Int32Value();
        return Wrap(info.Env(), Tensor::maxpool2d_backward(grad->tensor_, indices->tensor_, B, C, H, W));
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
    // Helper to extract Tensor array from JS array
    static std::vector<Tensor> extractTensorArray(const Napi::Array& arr) {
        std::vector<Tensor> v;
        for (uint32_t i = 0; i < arr.Length(); i++)
            v.push_back(Napi::ObjectWrap<TensorWrap>::Unwrap(arr.Get(i).As<Napi::Object>())->tensor_);
        return v;
    }

    static const Tensor* nullableArg(const Napi::Value& val) {
        if (val.IsNull() || val.IsUndefined()) return nullptr;
        return &Napi::ObjectWrap<TensorWrap>::Unwrap(val.As<Napi::Object>())->tensor_;
    }

    // rnnForward(x, hx_or_null, cx_or_null, wih[], whh[], bih[], bhh[], mode, hiddenSize, numLayers, bidir)
    static Napi::Value RnnForward(const Napi::CallbackInfo& info) {
        auto env = info.Env();
        auto& x = Napi::ObjectWrap<TensorWrap>::Unwrap(info[0].As<Napi::Object>())->tensor_;
        auto wih = extractTensorArray(info[3].As<Napi::Array>());
        auto whh = extractTensorArray(info[4].As<Napi::Array>());
        auto bih = extractTensorArray(info[5].As<Napi::Array>());
        auto bhh = extractTensorArray(info[6].As<Napi::Array>());

        auto result = Tensor::rnn_forward(x, nullableArg(info[1]), nullableArg(info[2]),
            wih, whh, bih, bhh,
            info[7].As<Napi::Number>().Int32Value(),
            info[8].As<Napi::Number>().Int32Value(),
            info[9].As<Napi::Number>().Int32Value(),
            info[10].As<Napi::Number>().Int32Value());

        auto arr = Napi::Array::New(env, result.size());
        for (size_t i = 0; i < result.size(); i++)
            arr.Set(i, TensorWrap::Wrap(env, std::move(result[i])));
        return arr;
    }

    // rnnBackward(x, hx, cx, y, dy, dhy, dcy, wih[], whh[], bih[], bhh[], mode, hiddenSize, numLayers, bidir)
    static Napi::Value RnnBackward(const Napi::CallbackInfo& info) {
        auto env = info.Env();
        auto& x = Napi::ObjectWrap<TensorWrap>::Unwrap(info[0].As<Napi::Object>())->tensor_;
        auto& y = Napi::ObjectWrap<TensorWrap>::Unwrap(info[3].As<Napi::Object>())->tensor_;
        auto& dy = Napi::ObjectWrap<TensorWrap>::Unwrap(info[4].As<Napi::Object>())->tensor_;
        auto wih = extractTensorArray(info[7].As<Napi::Array>());
        auto whh = extractTensorArray(info[8].As<Napi::Array>());
        auto bih = extractTensorArray(info[9].As<Napi::Array>());
        auto bhh = extractTensorArray(info[10].As<Napi::Array>());

        auto result = Tensor::rnn_backward(x, nullableArg(info[1]), nullableArg(info[2]),
            y, dy, nullableArg(info[5]), nullableArg(info[6]),
            wih, whh, bih, bhh,
            info[11].As<Napi::Number>().Int32Value(),
            info[12].As<Napi::Number>().Int32Value(),
            info[13].As<Napi::Number>().Int32Value(),
            info[14].As<Napi::Number>().Int32Value());

        auto arr = Napi::Array::New(env, result.size());
        for (size_t i = 0; i < result.size(); i++)
            arr.Set(i, TensorWrap::Wrap(env, std::move(result[i])));
        return arr;
    }

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

    static Napi::Value AdamStep(const Napi::CallbackInfo& info) {
        auto* param = Napi::ObjectWrap<TensorWrap>::Unwrap(info[0].As<Napi::Object>());
        auto* grad = Napi::ObjectWrap<TensorWrap>::Unwrap(info[1].As<Napi::Object>());
        auto* m = Napi::ObjectWrap<TensorWrap>::Unwrap(info[2].As<Napi::Object>());
        auto* v = Napi::ObjectWrap<TensorWrap>::Unwrap(info[3].As<Napi::Object>());
        float lr = info[4].As<Napi::Number>().FloatValue();
        float beta1 = info[5].As<Napi::Number>().FloatValue();
        float beta2 = info[6].As<Napi::Number>().FloatValue();
        float eps = info[7].As<Napi::Number>().FloatValue();
        float bc1 = info[8].As<Napi::Number>().FloatValue();
        float bc2 = info[9].As<Napi::Number>().FloatValue();
        float wd = info[10].As<Napi::Number>().FloatValue();
        Tensor::adam_step(param->tensor(), grad->tensor(), m->tensor(), v->tensor(),
            lr, beta1, beta2, eps, bc1, bc2, wd);
        return info.Env().Undefined();
    }

    // clearCache(): cudaDeviceSynchronize then release all fully-unused GPU
    // segments back to CUDA (segments still containing live tensor data are kept).
    static Napi::Value ClearCache(const Napi::CallbackInfo& info) {
        Tensor::clear_cache();
        return info.Env().Undefined();
    }
};

// ==================== GradTensorWrap ====================
class GradTensorWrap : public Napi::ObjectWrap<GradTensorWrap> {
public:
    GradPtr gt_;
    napi_env raw_env_ = nullptr;  // raw env for finalizer use
    int64_t ext_bytes_ = 0;      // GPU bytes reported to V8 GC

    static Napi::Object Wrap(Napi::Env env, GradPtr g) {
        int64_t bytes = g ? (int64_t)g->data.size() * 4 : 0;
        auto* refs = env.GetInstanceData<CtorRefs>();
        Napi::Object obj = refs->grad_ctor.New({});
        auto* gw = Napi::ObjectWrap<GradTensorWrap>::Unwrap(obj);
        gw->gt_ = std::move(g);
        gw->raw_env_ = env;
        gw->ext_bytes_ = bytes;
        if (bytes > 0) {
            int64_t unused;
            napi_adjust_external_memory(env, bytes, &unused);
        }
        return obj;
    }

    // Extract GradPtr from JS value — accepts both GradTensor and raw Tensor
    static GradPtr Extract(Napi::Value v) {
        auto obj = v.As<Napi::Object>();
        auto* refs = v.Env().GetInstanceData<CtorRefs>();
        // Check if it's a GradTensor instance
        if (obj.InstanceOf(refs->grad_ctor.Value())) {
            return Napi::ObjectWrap<GradTensorWrap>::Unwrap(obj)->gt_;
        }
        // Fallback: raw Tensor → wrap as non-grad GradTensor
        auto* tw = Napi::ObjectWrap<TensorWrap>::Unwrap(obj);
        return GradTensor::make(tw->tensor_, false);
    }

    static Napi::Object Init(Napi::Env env, Napi::Object exports) {
        Napi::Function func = DefineClass(env, "GradTensor", {
            InstanceAccessor("shape", &GradTensorWrap::GetShape, nullptr),
            InstanceAccessor("ndim", &GradTensorWrap::GetNdim, nullptr),
            InstanceAccessor("requires_grad", &GradTensorWrap::GetRequiresGrad, &GradTensorWrap::SetRequiresGrad),
            InstanceAccessor("data", &GradTensorWrap::GetData, &GradTensorWrap::SetData),
            InstanceAccessor("grad", &GradTensorWrap::GetGradProp, &GradTensorWrap::SetGradProp),

            InstanceMethod("toArray", &GradTensorWrap::ToArray),
            InstanceMethod("backward", &GradTensorWrap::Backward),
            InstanceMethod("getGrad", &GradTensorWrap::GetGrad),
            InstanceMethod("setGrad", &GradTensorWrap::SetGrad),
            InstanceMethod("clearGrad", &GradTensorWrap::ClearGrad),
            InstanceMethod("detach", &GradTensorWrap::Detach),
            InstanceMethod("getData", &GradTensorWrap::GetData),

            // View ops
            InstanceMethod("reshape", &GradTensorWrap::Reshape),
            InstanceMethod("transpose", &GradTensorWrap::Transpose),
            InstanceMethod("squeeze", &GradTensorWrap::Squeeze),
            InstanceMethod("unsqueeze", &GradTensorWrap::Unsqueeze),
            InstanceMethod("clone", &GradTensorWrap::Clone),
            InstanceMethod("contiguous", &GradTensorWrap::Contiguous),
            InstanceMethod("flatten", &GradTensorWrap::Flatten),
            InstanceMethod("slice", &GradTensorWrap::Slice),

            // Unary ops
            InstanceMethod("exp", &GradTensorWrap::Exp),
            InstanceMethod("log", &GradTensorWrap::Log),
            InstanceMethod("sqrt", &GradTensorWrap::Sqrt),
            InstanceMethod("square", &GradTensorWrap::Square),
            InstanceMethod("neg", &GradTensorWrap::Neg),
            InstanceMethod("abs", &GradTensorWrap::Abs),
            InstanceMethod("sin", &GradTensorWrap::Sin),
            InstanceMethod("cos", &GradTensorWrap::Cos),
            InstanceMethod("sigmoid", &GradTensorWrap::Sigmoid),
            InstanceMethod("tanh", &GradTensorWrap::Tanh),
            InstanceMethod("relu", &GradTensorWrap::Relu),
            InstanceMethod("silu", &GradTensorWrap::Silu),
            InstanceMethod("gelu", &GradTensorWrap::Gelu),
            InstanceMethod("softplus", &GradTensorWrap::Softplus),
            InstanceMethod("reciprocal", &GradTensorWrap::Reciprocal),
            InstanceMethod("sign", &GradTensorWrap::Sign),
            InstanceMethod("log1p", &GradTensorWrap::Log1p),
            InstanceMethod("pow_scalar", &GradTensorWrap::PowScalar),
            InstanceMethod("mul_scalar", &GradTensorWrap::MulScalar),
            InstanceMethod("add_scalar", &GradTensorWrap::AddScalar),

            // Binary ops
            InstanceMethod("add", &GradTensorWrap::Add),
            InstanceMethod("sub", &GradTensorWrap::Sub),
            InstanceMethod("mul", &GradTensorWrap::Mul),
            InstanceMethod("div", &GradTensorWrap::Div),
            InstanceMethod("pow", &GradTensorWrap::Pow),
            InstanceMethod("gt", &GradTensorWrap::Gt),
            InstanceMethod("lt", &GradTensorWrap::Lt),
            InstanceMethod("ge", &GradTensorWrap::Ge),
            InstanceMethod("le", &GradTensorWrap::Le),
            InstanceMethod("eq", &GradTensorWrap::Eq),
            InstanceMethod("ne", &GradTensorWrap::Ne),

            // Matmul
            InstanceMethod("matmul", &GradTensorWrap::Matmul),

            // Reduce
            InstanceMethod("sum", &GradTensorWrap::Sum),
            InstanceMethod("mean", &GradTensorWrap::Mean),

            // Conv/Pool
            InstanceMethod("conv2d", &GradTensorWrap::Conv2d),
            InstanceMethod("avg_pool2d", &GradTensorWrap::AvgPool2d),
            InstanceMethod("max_pool2d", &GradTensorWrap::MaxPool2d),
            InstanceMethod("conv1d", &GradTensorWrap::Conv1d),
            InstanceMethod("conv_transpose1d", &GradTensorWrap::ConvTranspose1d),
            InstanceMethod("conv_transpose2d", &GradTensorWrap::ConvTranspose2d),

            // Parameterized unary
            InstanceMethod("leaky_relu", &GradTensorWrap::LeakyRelu),
            InstanceMethod("clamp", &GradTensorWrap::Clamp),
            InstanceMethod("clamp_min", &GradTensorWrap::ClampMin),
            InstanceMethod("clamp_max", &GradTensorWrap::ClampMax),
            InstanceMethod("fmod", &GradTensorWrap::Fmod),

            // Binary
            InstanceMethod("maximum", &GradTensorWrap::Maximum),
            InstanceMethod("minimum", &GradTensorWrap::Minimum),

            // Misc ops
            InstanceMethod("embedding", &GradTensorWrap::Embedding),
            InstanceMethod("flip", &GradTensorWrap::Flip),
            InstanceMethod("pad", &GradTensorWrap::Pad),
            InstanceMethod("interpolate", &GradTensorWrap::Interpolate),
            InstanceMethod("cumsum", &GradTensorWrap::Cumsum),
            InstanceMethod("max", &GradTensorWrap::Max),
            InstanceMethod("min", &GradTensorWrap::Min),

            // Static
            StaticMethod("cat", &GradTensorWrap::CatGrad),
            StaticMethod("rnnForward", &GradTensorWrap::RnnForward_),
            StaticMethod("randn", &GradTensorWrap::Randn),
            StaticMethod("zeros", &GradTensorWrap::Zeros),
            StaticMethod("ones", &GradTensorWrap::Ones),
            StaticMethod("full", &GradTensorWrap::FullS),
            StaticMethod("fromBuffer", &GradTensorWrap::FromBuffer),
            StaticMethod("fromTensor", &GradTensorWrap::FromTensor),
            StaticMethod("adamStep", &GradTensorWrap::AdamStep),
            StaticMethod("adamStepMulti", &GradTensorWrap::AdamStepMulti),
        });

        auto* refs = env.GetInstanceData<CtorRefs>();
        refs->grad_ctor = Napi::Persistent(func);
        exports.Set("GradTensor", func);
        return exports;
    }

    GradTensorWrap(const Napi::CallbackInfo& info)
        : Napi::ObjectWrap<GradTensorWrap>(info) {
        if (info.Length() >= 1 && info[0].IsObject()) {
            // new GradTensor(tensor, requires_grad?)
            auto* tw = Napi::ObjectWrap<TensorWrap>::Unwrap(info[0].As<Napi::Object>());
            bool rg = (info.Length() > 1 && info[1].IsBoolean()) ? info[1].As<Napi::Boolean>().Value() : false;
            gt_ = GradTensor::make(tw->tensor_, rg);
        } else {
            gt_ = GradTensor::make(Tensor({1}));
        }
    }

    ~GradTensorWrap() {
        if (ext_bytes_ > 0 && raw_env_) {
            int64_t unused;
            napi_adjust_external_memory(raw_env_, -ext_bytes_, &unused);
        }
    }

    // ==================== Accessors ====================
    Napi::Value GetShape(const Napi::CallbackInfo& info) {
        auto& s = gt_->shape();
        Napi::Array arr = Napi::Array::New(info.Env(), s.size());
        for (size_t i = 0; i < s.size(); i++) arr[i] = Napi::Number::New(info.Env(), s[i]);
        return arr;
    }
    Napi::Value GetNdim(const Napi::CallbackInfo& info) {
        return Napi::Number::New(info.Env(), gt_->ndim());
    }
    Napi::Value GetRequiresGrad(const Napi::CallbackInfo& info) {
        return Napi::Boolean::New(info.Env(), gt_->requires_grad);
    }
    void SetRequiresGrad(const Napi::CallbackInfo& info, const Napi::Value& val) {
        gt_->requires_grad = val.As<Napi::Boolean>().Value();
    }
    Napi::Value ToArray(const Napi::CallbackInfo& info) {
        auto data = gt_->data.to_array();
        return buildNestedArray(info.Env(), data, gt_->shape(), 0, 0);
    }
    Napi::Value GetData(const Napi::CallbackInfo& info) {
        return TensorWrap::Wrap(info.Env(), gt_->data);
    }
    void SetData(const Napi::CallbackInfo& info, const Napi::Value& val) {
        gt_->data = Napi::ObjectWrap<TensorWrap>::Unwrap(val.As<Napi::Object>())->tensor_;
    }
    Napi::Value GetGradProp(const Napi::CallbackInfo& info) {
        if (!gt_->has_grad) return info.Env().Null();
        return TensorWrap::Wrap(info.Env(), *gt_->grad_);
    }
    void SetGradProp(const Napi::CallbackInfo& info, const Napi::Value& val) {
        if (val.IsNull() || val.IsUndefined()) {
            gt_->has_grad = false;
        } else {
            gt_->grad_ = std::make_shared<Tensor>(
                Napi::ObjectWrap<TensorWrap>::Unwrap(val.As<Napi::Object>())->tensor_);
            gt_->has_grad = true;
        }
    }
    Napi::Value Detach(const Napi::CallbackInfo& info) {
        return Wrap(info.Env(), GradTensor::make(gt_->data, false));
    }

    // ==================== Backward ====================
    Napi::Value Backward(const Napi::CallbackInfo& info) {
        Tensor upstream = (info.Length() > 0 && !info[0].IsUndefined())
            ? Napi::ObjectWrap<TensorWrap>::Unwrap(info[0].As<Napi::Object>())->tensor_
            : Tensor::full(gt_->shape(), 1.0f);
        GradTensor::backward(gt_, std::move(upstream));
        return info.Env().Undefined();
    }
    Napi::Value GetGrad(const Napi::CallbackInfo& info) {
        if (!gt_->has_grad) return info.Env().Null();
        return TensorWrap::Wrap(info.Env(), *gt_->grad_);
    }
    void SetGrad(const Napi::CallbackInfo& info) {
        if (info[0].IsNull() || info[0].IsUndefined()) {
            gt_->has_grad = false;
        } else {
            gt_->grad_ = std::make_shared<Tensor>(
                Napi::ObjectWrap<TensorWrap>::Unwrap(info[0].As<Napi::Object>())->tensor_);
            gt_->has_grad = true;
        }
    }
    Napi::Value ClearGrad(const Napi::CallbackInfo& info) {
        gt_->has_grad = false;
        return info.Env().Undefined();
    }

    // ==================== View ops ====================
    Napi::Value Reshape(const Napi::CallbackInfo& info) {
        return Wrap(info.Env(), gt_->reshape_(parseShape(info, 0)));
    }
    Napi::Value Transpose(const Napi::CallbackInfo& info) {
        if (info.Length() >= 2)
            return Wrap(info.Env(), gt_->transpose_(
                info[0].As<Napi::Number>().Int32Value(),
                info[1].As<Napi::Number>().Int32Value()));
        return Wrap(info.Env(), gt_->transpose_());
    }
    Napi::Value Squeeze(const Napi::CallbackInfo& info) {
        // Squeeze is just reshape
        int axis = info.Length() > 0 ? info[0].As<Napi::Number>().Int32Value() : -1;
        Tensor out = gt_->data.squeeze(axis);
        if (!gt_->requires_grad) return Wrap(info.Env(), GradTensor::make(std::move(out)));
        Shape orig = gt_->shape();
        return Wrap(info.Env(), GradTensor::make_with_grad(std::move(out),
            [orig](const Tensor& g) { return std::vector<Tensor>{g.reshape(orig)}; },
            {gt_}));
    }
    Napi::Value Unsqueeze(const Napi::CallbackInfo& info) {
        return Wrap(info.Env(), gt_->unsqueeze_(info[0].As<Napi::Number>().Int32Value()));
    }
    Napi::Value Clone(const Napi::CallbackInfo& info) {
        return Wrap(info.Env(), gt_->clone_());
    }
    Napi::Value Contiguous(const Napi::CallbackInfo& info) {
        return Wrap(info.Env(), gt_->contiguous_());
    }
    Napi::Value Flatten(const Napi::CallbackInfo& info) {
        int s = info[0].As<Napi::Number>().Int32Value();
        int e = info.Length() > 1 ? info[1].As<Napi::Number>().Int32Value() : -1;
        Tensor out = gt_->data.flatten(s, e);
        if (!gt_->requires_grad) return Wrap(info.Env(), GradTensor::make(std::move(out)));
        Shape orig = gt_->shape();
        return Wrap(info.Env(), GradTensor::make_with_grad(std::move(out),
            [orig](const Tensor& g) { return std::vector<Tensor>{g.reshape(orig)}; },
            {gt_}));
    }
    Napi::Value Slice(const Napi::CallbackInfo& info) {
        int dim = info[0].As<Napi::Number>().Int32Value();
        int start = info[1].As<Napi::Number>().Int32Value();
        int end = info[2].As<Napi::Number>().Int32Value();
        Tensor out = gt_->data.slice(dim, start, end);
        if (!gt_->requires_grad) return Wrap(info.Env(), GradTensor::make(std::move(out)));
        Shape orig = gt_->shape();
        return Wrap(info.Env(), GradTensor::make_with_grad(std::move(out),
            [orig, dim, start, end](const Tensor& g) {
                // Build zero tensor, cat [zeros_before, g, zeros_after]
                std::vector<Tensor> parts;
                if (start > 0) {
                    Shape bs = g.shape(); bs[dim] = start;
                    parts.push_back(Tensor::full(bs, 0.0f));
                }
                parts.push_back(g);
                int after = orig[dim] - end;
                if (after > 0) {
                    Shape as = g.shape(); as[dim] = after;
                    parts.push_back(Tensor::full(as, 0.0f));
                }
                return std::vector<Tensor>{Tensor::cat(parts, dim)};
            }, {gt_}));
    }

    // ==================== Unary ops (delegate to autograd.hpp) ====================
    #define GRAD_UNARY(JsName, CppMethod) \
    Napi::Value JsName(const Napi::CallbackInfo& info) { return Wrap(info.Env(), gt_->CppMethod()); }

    GRAD_UNARY(Exp, exp_)      GRAD_UNARY(Log, log_)      GRAD_UNARY(Sqrt, sqrt_)
    GRAD_UNARY(Square, square_) GRAD_UNARY(Neg, neg_)      GRAD_UNARY(Abs, abs_)
    GRAD_UNARY(Sin, sin_)      GRAD_UNARY(Cos, cos_)      GRAD_UNARY(Sigmoid, sigmoid_)
    GRAD_UNARY(Tanh, tanh_)    GRAD_UNARY(Relu, relu_)    GRAD_UNARY(Silu, silu_)
    GRAD_UNARY(Gelu, gelu_)    GRAD_UNARY(Softplus, softplus_)
    GRAD_UNARY(Reciprocal, reciprocal_) GRAD_UNARY(Sign, sign_) GRAD_UNARY(Log1p, log1p_)

    Napi::Value PowScalar(const Napi::CallbackInfo& info) {
        return Wrap(info.Env(), gt_->pow_scalar_(info[0].As<Napi::Number>().FloatValue()));
    }
    Napi::Value MulScalar(const Napi::CallbackInfo& info) {
        return Wrap(info.Env(), gt_->mul_scalar_(info[0].As<Napi::Number>().FloatValue()));
    }
    Napi::Value AddScalar(const Napi::CallbackInfo& info) {
        return Wrap(info.Env(), gt_->add_scalar_(info[0].As<Napi::Number>().FloatValue()));
    }

    // ==================== Binary ops ====================
    // Scalars: wrap as single-element GradTensor or use scalar ops
    Napi::Value Add(const Napi::CallbackInfo& info) {
        if (info[0].IsNumber())
            return Wrap(info.Env(), gt_->add_scalar_(info[0].As<Napi::Number>().FloatValue()));
        return Wrap(info.Env(), gt_->add_(Extract(info[0])));
    }
    Napi::Value Sub(const Napi::CallbackInfo& info) {
        if (info[0].IsNumber())
            return Wrap(info.Env(), gt_->add_scalar_(-info[0].As<Napi::Number>().FloatValue()));
        return Wrap(info.Env(), gt_->sub_(Extract(info[0])));
    }
    Napi::Value Mul(const Napi::CallbackInfo& info) {
        if (info[0].IsNumber())
            return Wrap(info.Env(), gt_->mul_scalar_(info[0].As<Napi::Number>().FloatValue()));
        return Wrap(info.Env(), gt_->mul_(Extract(info[0])));
    }
    Napi::Value Div(const Napi::CallbackInfo& info) {
        if (info[0].IsNumber())
            return Wrap(info.Env(), gt_->mul_scalar_(1.0f / info[0].As<Napi::Number>().FloatValue()));
        return Wrap(info.Env(), gt_->div_(Extract(info[0])));
    }
    Napi::Value Pow(const Napi::CallbackInfo& info) {
        if (info[0].IsNumber())
            return Wrap(info.Env(), gt_->pow_scalar_(info[0].As<Napi::Number>().FloatValue()));
        return Wrap(info.Env(), gt_->pow_(Extract(info[0])));
    }
    Napi::Value Gt(const Napi::CallbackInfo& info) { return Wrap(info.Env(), gt_->gt_(Extract(info[0]))); }
    Napi::Value Lt(const Napi::CallbackInfo& info) { return Wrap(info.Env(), gt_->lt_(Extract(info[0]))); }
    Napi::Value Ge(const Napi::CallbackInfo& info) { return Wrap(info.Env(), gt_->ge_(Extract(info[0]))); }
    Napi::Value Le(const Napi::CallbackInfo& info) { return Wrap(info.Env(), gt_->le_(Extract(info[0]))); }
    Napi::Value Eq(const Napi::CallbackInfo& info) { return Wrap(info.Env(), gt_->eq_(Extract(info[0]))); }
    Napi::Value Ne(const Napi::CallbackInfo& info) { return Wrap(info.Env(), gt_->ne_(Extract(info[0]))); }

    // ==================== Matmul ====================
    Napi::Value Matmul(const Napi::CallbackInfo& info) {
        return Wrap(info.Env(), gt_->matmul_(Extract(info[0])));
    }

    // ==================== Reduce ====================
    Napi::Value Sum(const Napi::CallbackInfo& info) {
        int dim = info.Length() > 0 ? info[0].As<Napi::Number>().Int32Value() : -1;
        bool keepdim = info.Length() > 1 ? info[1].As<Napi::Boolean>().Value() : false;
        return Wrap(info.Env(), gt_->sum_(dim, keepdim));
    }
    Napi::Value Mean(const Napi::CallbackInfo& info) {
        int dim = info.Length() > 0 ? info[0].As<Napi::Number>().Int32Value() : -1;
        bool keepdim = info.Length() > 1 ? info[1].As<Napi::Boolean>().Value() : false;
        return Wrap(info.Env(), gt_->mean_(dim, keepdim));
    }

    // ==================== Conv/Pool ====================
    Napi::Value Conv2d(const Napi::CallbackInfo& info) {
        GradPtr w = Extract(info[0]);
        GradPtr b = nullptr;
        if (!info[1].IsNull() && !info[1].IsUndefined())
            b = Extract(info[1]);
        int sH = info[2].As<Napi::Number>().Int32Value();
        int sW = info[3].As<Napi::Number>().Int32Value();
        int pH = info[4].As<Napi::Number>().Int32Value();
        int pW = info[5].As<Napi::Number>().Int32Value();
        int dH = info[6].As<Napi::Number>().Int32Value();
        int dW = info[7].As<Napi::Number>().Int32Value();
        int groups = info[8].As<Napi::Number>().Int32Value();
        return Wrap(info.Env(), gt_->conv2d_(w, b, sH, sW, pH, pW, dH, dW, groups));
    }
    Napi::Value AvgPool2d(const Napi::CallbackInfo& info) {
        int kH = info[0].As<Napi::Number>().Int32Value();
        int kW = info[1].As<Napi::Number>().Int32Value();
        int sH = info[2].As<Napi::Number>().Int32Value();
        int sW = info[3].As<Napi::Number>().Int32Value();
        int pH = info[4].As<Napi::Number>().Int32Value();
        int pW = info[5].As<Napi::Number>().Int32Value();
        bool cip = info.Length() > 6 ? info[6].As<Napi::Boolean>().Value() : true;
        return Wrap(info.Env(), gt_->avg_pool2d_(kH, kW, sH, sW, pH, pW, cip));
    }

    Napi::Value MaxPool2d(const Napi::CallbackInfo& info) {
        int kH = info[0].As<Napi::Number>().Int32Value();
        int kW = info[1].As<Napi::Number>().Int32Value();
        int sH = info[2].As<Napi::Number>().Int32Value();
        int sW = info[3].As<Napi::Number>().Int32Value();
        int pH = info[4].As<Napi::Number>().Int32Value();
        int pW = info[5].As<Napi::Number>().Int32Value();
        return Wrap(info.Env(), gt_->max_pool2d_(kH, kW, sH, sW, pH, pW));
    }

    // ==================== New parameterized unary ====================
    Napi::Value LeakyRelu(const Napi::CallbackInfo& info) {
        float slope = info.Length() > 0 ? info[0].As<Napi::Number>().FloatValue() : 0.01f;
        return Wrap(info.Env(), gt_->leaky_relu_(slope));
    }
    Napi::Value Clamp(const Napi::CallbackInfo& info) {
        float lo = info[0].As<Napi::Number>().FloatValue();
        float hi = info[1].As<Napi::Number>().FloatValue();
        return Wrap(info.Env(), gt_->clamp_(lo, hi));
    }
    Napi::Value ClampMin(const Napi::CallbackInfo& info) {
        return Wrap(info.Env(), gt_->clamp_min_(info[0].As<Napi::Number>().FloatValue()));
    }
    Napi::Value ClampMax(const Napi::CallbackInfo& info) {
        return Wrap(info.Env(), gt_->clamp_max_(info[0].As<Napi::Number>().FloatValue()));
    }
    Napi::Value Fmod(const Napi::CallbackInfo& info) {
        return Wrap(info.Env(), gt_->fmod_(info[0].As<Napi::Number>().FloatValue()));
    }

    // ==================== New binary ====================
    Napi::Value Maximum(const Napi::CallbackInfo& info) {
        return Wrap(info.Env(), gt_->maximum_(Extract(info[0])));
    }
    Napi::Value Minimum(const Napi::CallbackInfo& info) {
        return Wrap(info.Env(), gt_->minimum_(Extract(info[0])));
    }

    // ==================== Conv1d/ConvTranspose ====================
    Napi::Value Conv1d(const Napi::CallbackInfo& info) {
        GradPtr w = Extract(info[0]);
        GradPtr b = nullptr;
        if (!info[1].IsNull() && !info[1].IsUndefined()) b = Extract(info[1]);
        int stride = info[2].As<Napi::Number>().Int32Value();
        int pad = info[3].As<Napi::Number>().Int32Value();
        int dil = info[4].As<Napi::Number>().Int32Value();
        int groups = info[5].As<Napi::Number>().Int32Value();
        return Wrap(info.Env(), gt_->conv1d_(w, b, stride, pad, dil, groups));
    }
    Napi::Value ConvTranspose1d(const Napi::CallbackInfo& info) {
        GradPtr w = Extract(info[0]);
        GradPtr b = nullptr;
        if (!info[1].IsNull() && !info[1].IsUndefined()) b = Extract(info[1]);
        int stride = info[2].As<Napi::Number>().Int32Value();
        int pad = info[3].As<Napi::Number>().Int32Value();
        int opad = info[4].As<Napi::Number>().Int32Value();
        int dil = info[5].As<Napi::Number>().Int32Value();
        int groups = info[6].As<Napi::Number>().Int32Value();
        return Wrap(info.Env(), gt_->conv_transpose1d_(w, b, stride, pad, opad, dil, groups));
    }
    Napi::Value ConvTranspose2d(const Napi::CallbackInfo& info) {
        GradPtr w = Extract(info[0]);
        GradPtr b = nullptr;
        if (!info[1].IsNull() && !info[1].IsUndefined()) b = Extract(info[1]);
        int sH = info[2].As<Napi::Number>().Int32Value();
        int sW = info[3].As<Napi::Number>().Int32Value();
        int pH = info[4].As<Napi::Number>().Int32Value();
        int pW = info[5].As<Napi::Number>().Int32Value();
        int opH = info[6].As<Napi::Number>().Int32Value();
        int opW = info[7].As<Napi::Number>().Int32Value();
        int dH = info[8].As<Napi::Number>().Int32Value();
        int dW = info[9].As<Napi::Number>().Int32Value();
        int groups = info[10].As<Napi::Number>().Int32Value();
        return Wrap(info.Env(), gt_->conv_transpose2d_(w, b, sH, sW, pH, pW, opH, opW, dH, dW, groups));
    }

    // ==================== Misc ops ====================
    Napi::Value Embedding(const Napi::CallbackInfo& info) {
        auto& indices = Napi::ObjectWrap<TensorWrap>::Unwrap(info[0].As<Napi::Object>())->tensor_;
        return Wrap(info.Env(), gt_->embedding_(indices));
    }
    Napi::Value Flip(const Napi::CallbackInfo& info) {
        return Wrap(info.Env(), gt_->flip_(info[0].As<Napi::Number>().Int32Value()));
    }
    Napi::Value Pad(const Napi::CallbackInfo& info) {
        auto arr = info[0].As<Napi::Array>();
        std::vector<int> padding;
        for (uint32_t i = 0; i < arr.Length(); i++)
            padding.push_back(((Napi::Value)arr[i]).As<Napi::Number>().Int32Value());
        int mode = info.Length() > 1 ? info[1].As<Napi::Number>().Int32Value() : 0;
        float val = info.Length() > 2 ? info[2].As<Napi::Number>().FloatValue() : 0.0f;
        return Wrap(info.Env(), gt_->pad_(padding, mode, val));
    }
    Napi::Value Interpolate(const Napi::CallbackInfo& info) {
        int target = info[0].As<Napi::Number>().Int32Value();
        int mode = info.Length() > 1 ? info[1].As<Napi::Number>().Int32Value() : 0;
        bool align = info.Length() > 2 ? info[2].As<Napi::Boolean>().Value() : false;
        return Wrap(info.Env(), gt_->interpolate_(target, mode, align));
    }
    Napi::Value Cumsum(const Napi::CallbackInfo& info) {
        return Wrap(info.Env(), gt_->cumsum_(info[0].As<Napi::Number>().Int32Value()));
    }
    Napi::Value FlattenGrad(const Napi::CallbackInfo& info) {
        int start = info.Length() > 0 ? info[0].As<Napi::Number>().Int32Value() : 0;
        int end = info.Length() > 1 ? info[1].As<Napi::Number>().Int32Value() : -1;
        return Wrap(info.Env(), gt_->flatten_(start, end));
    }
    Napi::Value Max(const Napi::CallbackInfo& info) {
        int dim = info[0].As<Napi::Number>().Int32Value();
        bool keepdim = info.Length() > 1 ? info[1].As<Napi::Boolean>().Value() : false;
        return Wrap(info.Env(), gt_->max_(dim, keepdim));
    }
    Napi::Value Min(const Napi::CallbackInfo& info) {
        int dim = info[0].As<Napi::Number>().Int32Value();
        bool keepdim = info.Length() > 1 ? info[1].As<Napi::Boolean>().Value() : false;
        return Wrap(info.Env(), gt_->min_(dim, keepdim));
    }
    static Napi::Value CatGrad(const Napi::CallbackInfo& info) {
        auto arr = info[0].As<Napi::Array>();
        std::vector<GradPtr> tensors;
        for (uint32_t i = 0; i < arr.Length(); i++)
            tensors.push_back(Extract((Napi::Value)arr[i]));
        int dim = info[1].As<Napi::Number>().Int32Value();
        return Wrap(info.Env(), GradTensor::cat_(tensors, dim));
    }

    // ==================== Static factories ====================
    static Napi::Value Randn(const Napi::CallbackInfo& info) {
        return Wrap(info.Env(), GradTensor::make(Tensor::randn(parseShape(info, 0))));
    }
    static Napi::Value Zeros(const Napi::CallbackInfo& info) {
        Shape s = parseShape(info, 0);
        return Wrap(info.Env(), GradTensor::make(Tensor::full(s, 0.0f)));
    }
    static Napi::Value Ones(const Napi::CallbackInfo& info) {
        return Wrap(info.Env(), GradTensor::make(Tensor::full(parseShape(info, 0), 1.0f)));
    }
    static Napi::Value FullS(const Napi::CallbackInfo& info) {
        Shape s = parseShape(info, 0);
        float v = info[1].As<Napi::Number>().FloatValue();
        return Wrap(info.Env(), GradTensor::make(Tensor::full(s, v)));
    }
    static Napi::Value FromBuffer(const Napi::CallbackInfo& info) {
        Napi::Float32Array f32 = info[0].As<Napi::Float32Array>();
        Shape shape = parseShape(info, 1);
        return Wrap(info.Env(), GradTensor::make(
            Tensor::from_buffer(f32.Data(), (int)f32.ElementLength(), shape)));
    }
    static Napi::Value FromTensor(const Napi::CallbackInfo& info) {
        auto* tw = Napi::ObjectWrap<TensorWrap>::Unwrap(info[0].As<Napi::Object>());
        bool rg = info.Length() > 1 ? info[1].As<Napi::Boolean>().Value() : false;
        return Wrap(info.Env(), GradTensor::make(tw->tensor_, rg));
    }

    // ==================== Adam (single + multi) ====================
    static Napi::Value AdamStep(const Napi::CallbackInfo& info) {
        auto p = Extract(info[0]), g = Extract(info[1]), m = Extract(info[2]), v = Extract(info[3]);
        float lr = info[4].As<Napi::Number>().FloatValue();
        float b1 = info[5].As<Napi::Number>().FloatValue();
        float b2 = info[6].As<Napi::Number>().FloatValue();
        float eps = info[7].As<Napi::Number>().FloatValue();
        float bc1 = info[8].As<Napi::Number>().FloatValue();
        float bc2 = info[9].As<Napi::Number>().FloatValue();
        float wd = info[10].As<Napi::Number>().FloatValue();
        Tensor::adam_step(p->data, g->data, m->data, v->data, lr, b1, b2, eps, bc1, bc2, wd);
        return info.Env().Undefined();
    }

    // ==================== RNN forward with autograd ====================
    static std::vector<GradPtr> extractGradArray(const Napi::Array& arr) {
        std::vector<GradPtr> v;
        for (uint32_t i = 0; i < arr.Length(); i++)
            v.push_back(Extract(arr.Get(i)));
        return v;
    }

    // rnnForward(x, hx_or_null, cx_or_null, wih[], whh[], bih[], bhh[], mode, hiddenSize, numLayers, bidir)
    // Returns GradTensor array: [y, hy] for RNN/GRU, [y, hy, cy] for LSTM
    static Napi::Value RnnForward_(const Napi::CallbackInfo& info) {
        auto env = info.Env();

        GradPtr gx = Extract(info[0]);

        bool has_hx = !(info[1].IsNull() || info[1].IsUndefined());
        GradPtr ghx = has_hx ? Extract(info[1]) : GradTensor::make(Tensor(), false);

        bool has_cx = !(info[2].IsNull() || info[2].IsUndefined());
        GradPtr gcx = has_cx ? Extract(info[2]) : nullptr;

        auto gwih = extractGradArray(info[3].As<Napi::Array>());
        auto gwhh = extractGradArray(info[4].As<Napi::Array>());
        auto gbih = extractGradArray(info[5].As<Napi::Array>());
        auto gbhh = extractGradArray(info[6].As<Napi::Array>());

        int mode        = info[7].As<Napi::Number>().Int32Value();
        int hidden_size = info[8].As<Napi::Number>().Int32Value();
        int num_layers  = info[9].As<Napi::Number>().Int32Value();
        int bidir       = info[10].As<Napi::Number>().Int32Value();
        bool is_lstm    = (mode == 2);
        int n           = (int)gwih.size();

        std::vector<Tensor> wih_t, whh_t, bih_t, bhh_t;
        for (auto& g : gwih) wih_t.push_back(g->data);
        for (auto& g : gwhh) whh_t.push_back(g->data);
        for (auto& g : gbih) bih_t.push_back(g->data);
        for (auto& g : gbhh) bhh_t.push_back(g->data);

        const Tensor* hx_raw = has_hx ? &ghx->data : nullptr;
        const Tensor* cx_raw = (has_cx && gcx) ? &gcx->data : nullptr;

        auto result = Tensor::rnn_forward(gx->data, hx_raw, cx_raw,
            wih_t, whh_t, bih_t, bhh_t, mode, hidden_size, num_layers, bidir);

        // Check if any input requires grad
        bool need = gx->requires_grad || ghx->requires_grad;
        if (has_cx && gcx) need = need || gcx->requires_grad;
        for (auto& g : gwih) need = need || g->requires_grad;
        for (auto& g : gwhh) need = need || g->requires_grad;
        for (auto& g : gbih) need = need || g->requires_grad;
        for (auto& g : gbhh) need = need || g->requires_grad;

        if (!need) {
            auto arr = Napi::Array::New(env, result.size());
            for (uint32_t i = 0; i < (uint32_t)result.size(); i++)
                arr.Set(i, Wrap(env, GradTensor::make(std::move(result[i]))));
            return arr;
        }

        // Build shared backward context
        struct RnnCtx {
            Tensor x, y, hx, cx;
            bool has_hx = false, has_cx = false;
            std::vector<Tensor> wih_t, whh_t, bih_t, bhh_t;
            int mode, hidden_size, num_layers, bidir;
        };
        auto ctx = std::make_shared<RnnCtx>();
        ctx->x = gx->data;
        ctx->y = result[0];   // shallow copy — shares GPU buffer
        if (has_hx)           { ctx->hx = ghx->data; ctx->has_hx = true; }
        if (has_cx && gcx)    { ctx->cx = gcx->data; ctx->has_cx = true; }
        ctx->wih_t = std::move(wih_t);
        ctx->whh_t = std::move(whh_t);
        ctx->bih_t = std::move(bih_t);
        ctx->bhh_t = std::move(bhh_t);
        ctx->mode = mode; ctx->hidden_size = hidden_size;
        ctx->num_layers = num_layers; ctx->bidir = bidir;

        // Parents order matches rnn_backward output:
        // [gx, ghx, (gcx for LSTM), gwih[0], gwhh[0], gbih[0], gbhh[0], ...]
        std::vector<GradPtr> parents;
        parents.push_back(gx);
        parents.push_back(ghx);   // dummy (requires_grad=false) if h0 was null
        if (is_lstm)
            parents.push_back((has_cx && gcx) ? gcx : GradTensor::make(Tensor(), false));
        for (int i = 0; i < n; i++) {
            parents.push_back(gwih[i]);
            parents.push_back(gwhh[i]);
            parents.push_back(gbih[i]);
            parents.push_back(gbhh[i]);
        }

        // grad_fn for y output: dy → run backward with dy, dhy=null
        auto y_grad_fn = [ctx](const Tensor& dy) -> std::vector<Tensor> {
            const Tensor* hxp = ctx->has_hx ? &ctx->hx : nullptr;
            const Tensor* cxp = ctx->has_cx ? &ctx->cx : nullptr;
            return Tensor::rnn_backward(ctx->x, hxp, cxp, ctx->y, dy, nullptr, nullptr,
                ctx->wih_t, ctx->whh_t, ctx->bih_t, ctx->bhh_t,
                ctx->mode, ctx->hidden_size, ctx->num_layers, ctx->bidir);
        };
        // grad_fn for hy output: dhy → run backward with dy=zeros, dhy
        auto hy_grad_fn = [ctx](const Tensor& dhy) -> std::vector<Tensor> {
            const Tensor* hxp = ctx->has_hx ? &ctx->hx : nullptr;
            const Tensor* cxp = ctx->has_cx ? &ctx->cx : nullptr;
            Tensor zeros_dy = Tensor::full(ctx->y.shape(), 0.f);
            return Tensor::rnn_backward(ctx->x, hxp, cxp, ctx->y, zeros_dy, &dhy, nullptr,
                ctx->wih_t, ctx->whh_t, ctx->bih_t, ctx->bhh_t,
                ctx->mode, ctx->hidden_size, ctx->num_layers, ctx->bidir);
        };

        GradPtr gy  = GradTensor::make_with_grad(ctx->y,              y_grad_fn, parents);
        GradPtr ghy = GradTensor::make_with_grad(std::move(result[1]), hy_grad_fn, parents);

        auto out_arr = Napi::Array::New(env, is_lstm ? 3u : 2u);
        out_arr.Set(0u, Wrap(env, gy));
        out_arr.Set(1u, Wrap(env, ghy));

        if (is_lstm) {
            auto cy_grad_fn = [ctx](const Tensor& dcy) -> std::vector<Tensor> {
                const Tensor* hxp = ctx->has_hx ? &ctx->hx : nullptr;
                const Tensor* cxp = ctx->has_cx ? &ctx->cx : nullptr;
                Tensor zeros_dy = Tensor::full(ctx->y.shape(), 0.f);
                return Tensor::rnn_backward(ctx->x, hxp, cxp, ctx->y, zeros_dy, nullptr, &dcy,
                    ctx->wih_t, ctx->whh_t, ctx->bih_t, ctx->bhh_t,
                    ctx->mode, ctx->hidden_size, ctx->num_layers, ctx->bidir);
            };
            out_arr.Set(2u, Wrap(env, GradTensor::make_with_grad(std::move(result[2]), cy_grad_fn, parents)));
        }
        return out_arr;
    }

    // Multi-tensor Adam: process all parameters in a batch
    static Napi::Value AdamStepMulti(const Napi::CallbackInfo& info) {
        Napi::Array params = info[0].As<Napi::Array>();
        Napi::Array grads = info[1].As<Napi::Array>();
        Napi::Array ms = info[2].As<Napi::Array>();
        Napi::Array vs = info[3].As<Napi::Array>();
        float lr = info[4].As<Napi::Number>().FloatValue();
        float b1 = info[5].As<Napi::Number>().FloatValue();
        float b2 = info[6].As<Napi::Number>().FloatValue();
        float eps = info[7].As<Napi::Number>().FloatValue();
        float bc1 = info[8].As<Napi::Number>().FloatValue();
        float bc2 = info[9].As<Napi::Number>().FloatValue();
        float wd = info[10].As<Napi::Number>().FloatValue();
        uint32_t n = params.Length();
        for (uint32_t i = 0; i < n; i++) {
            auto p = Extract(params.Get(i));
            auto g = Extract(grads.Get(i));
            auto m = Extract(ms.Get(i));
            auto v = Extract(vs.Get(i));
            Tensor::adam_step(p->data, g->data, m->data, v->data, lr, b1, b2, eps, bc1, bc2, wd);
        }
        return info.Env().Undefined();
    }
};

// ==================== CompiledGraphWrap ====================
class CompiledGraphWrap : public Napi::ObjectWrap<CompiledGraphWrap> {
    CompiledGraph graph_;

public:
    static Napi::Object Init(Napi::Env env, Napi::Object exports) {
        Napi::Function func = DefineClass(env, "CompiledGraph", {
            InstanceMethod("input", &CompiledGraphWrap::Input),
            InstanceMethod("target", &CompiledGraphWrap::Target),
            InstanceMethod("param", &CompiledGraphWrap::Param),
            InstanceMethod("setOutput", &CompiledGraphWrap::SetOutput),
            // Unary
            InstanceMethod("relu", &CompiledGraphWrap::Relu),
            InstanceMethod("exp", &CompiledGraphWrap::Exp),
            InstanceMethod("log", &CompiledGraphWrap::Log),
            InstanceMethod("sqrt", &CompiledGraphWrap::Sqrt),
            InstanceMethod("square", &CompiledGraphWrap::Square),
            InstanceMethod("neg", &CompiledGraphWrap::Neg_),
            InstanceMethod("abs", &CompiledGraphWrap::Abs_),
            InstanceMethod("sigmoid", &CompiledGraphWrap::Sigmoid),
            InstanceMethod("tanh", &CompiledGraphWrap::Tanh_),
            InstanceMethod("silu", &CompiledGraphWrap::Silu),
            InstanceMethod("gelu", &CompiledGraphWrap::Gelu),
            InstanceMethod("softplus", &CompiledGraphWrap::Softplus),
            InstanceMethod("sin", &CompiledGraphWrap::Sin_),
            InstanceMethod("cos", &CompiledGraphWrap::Cos_),
            // Parameterized unary
            InstanceMethod("pow_scalar", &CompiledGraphWrap::PowScalar),
            InstanceMethod("mul_scalar", &CompiledGraphWrap::MulScalar_),
            InstanceMethod("add_scalar", &CompiledGraphWrap::AddScalar_),
            // Binary
            InstanceMethod("add", &CompiledGraphWrap::Add_),
            InstanceMethod("sub", &CompiledGraphWrap::Sub_),
            InstanceMethod("mul", &CompiledGraphWrap::Mul_),
            InstanceMethod("div", &CompiledGraphWrap::Div__),
            InstanceMethod("pow", &CompiledGraphWrap::Pow__),
            // Matmul
            InstanceMethod("matmul", &CompiledGraphWrap::Matmul_),
            // Reduce
            InstanceMethod("sum", &CompiledGraphWrap::Sum_),
            InstanceMethod("mean", &CompiledGraphWrap::Mean_),
            // View
            InstanceMethod("transpose", &CompiledGraphWrap::Transpose_),
            InstanceMethod("reshape", &CompiledGraphWrap::Reshape_),
            InstanceMethod("flatten", &CompiledGraphWrap::Flatten_),
            // Misc
            InstanceMethod("cat", &CompiledGraphWrap::Cat_),
            InstanceMethod("slice", &CompiledGraphWrap::Slice_),
            InstanceMethod("unsqueeze", &CompiledGraphWrap::Unsqueeze_),
            InstanceMethod("squeeze", &CompiledGraphWrap::Squeeze_),
            // CNN
            InstanceMethod("conv2d", &CompiledGraphWrap::Conv2d_),
            InstanceMethod("max_pool2d", &CompiledGraphWrap::MaxPool2d_),
            InstanceMethod("avg_pool2d", &CompiledGraphWrap::AvgPool2d_),
            InstanceMethod("batch_norm2d", &CompiledGraphWrap::BatchNorm2d_),
            // Buffer
            InstanceMethod("buffer", &CompiledGraphWrap::Buffer),
            // Loop
            InstanceMethod("loop_begin", &CompiledGraphWrap::LoopBegin),
            InstanceMethod("loop_slice", &CompiledGraphWrap::LoopSlice),
            InstanceMethod("loop_carry", &CompiledGraphWrap::LoopCarry),
            InstanceMethod("loop_end", &CompiledGraphWrap::LoopEnd),
            // Fused ops
            InstanceMethod("rnn_cudnn", &CompiledGraphWrap::RnnCudnn),
            // Run
            InstanceMethod("run", &CompiledGraphWrap::Run),
        });
        exports.Set("CompiledGraph", func);
        return exports;
    }

    CompiledGraphWrap(const Napi::CallbackInfo& info)
        : Napi::ObjectWrap<CompiledGraphWrap>(info) {}

    // Slot allocation
    Napi::Value Input(const Napi::CallbackInfo& info) {
        return Napi::Number::New(info.Env(), graph_.input());
    }
    Napi::Value Target(const Napi::CallbackInfo& info) {
        return Napi::Number::New(info.Env(), graph_.target());
    }
    Napi::Value Param(const Napi::CallbackInfo& info) {
        return Napi::Number::New(info.Env(), graph_.param());
    }
    Napi::Value SetOutput(const Napi::CallbackInfo& info) {
        graph_.set_output(info[0].As<Napi::Number>().Int32Value());
        return info.Env().Undefined();
    }

    // Unary ops
    #define CG_UNARY(Name, OpT) \
    Napi::Value Name(const Napi::CallbackInfo& info) { \
        return Napi::Number::New(info.Env(), graph_.op1(OpType::OpT, info[0].As<Napi::Number>().Int32Value())); \
    }
    CG_UNARY(Relu, RELU)       CG_UNARY(Exp, EXP)         CG_UNARY(Log, LOG)
    CG_UNARY(Sqrt, SQRT)       CG_UNARY(Square, SQUARE)   CG_UNARY(Neg_, NEG)
    CG_UNARY(Abs_, ABS)        CG_UNARY(Sigmoid, SIGMOID) CG_UNARY(Tanh_, TANH)
    CG_UNARY(Silu, SILU)       CG_UNARY(Gelu, GELU)       CG_UNARY(Softplus, SOFTPLUS)
    CG_UNARY(Sin_, SIN)        CG_UNARY(Cos_, COS)

    // Parameterized unary
    Napi::Value PowScalar(const Napi::CallbackInfo& info) {
        return Napi::Number::New(info.Env(), graph_.op1(OpType::POW_SCALAR,
            info[0].As<Napi::Number>().Int32Value(), info[1].As<Napi::Number>().FloatValue()));
    }
    Napi::Value MulScalar_(const Napi::CallbackInfo& info) {
        return Napi::Number::New(info.Env(), graph_.op1(OpType::MUL_SCALAR,
            info[0].As<Napi::Number>().Int32Value(), info[1].As<Napi::Number>().FloatValue()));
    }
    Napi::Value AddScalar_(const Napi::CallbackInfo& info) {
        return Napi::Number::New(info.Env(), graph_.op1(OpType::ADD_SCALAR,
            info[0].As<Napi::Number>().Int32Value(), info[1].As<Napi::Number>().FloatValue()));
    }

    // Binary ops
    #define CG_BINARY(Name, OpT) \
    Napi::Value Name(const Napi::CallbackInfo& info) { \
        return Napi::Number::New(info.Env(), graph_.op2(OpType::OpT, \
            info[0].As<Napi::Number>().Int32Value(), info[1].As<Napi::Number>().Int32Value())); \
    }
    CG_BINARY(Add_, ADD)   CG_BINARY(Sub_, SUB)   CG_BINARY(Mul_, MUL)
    CG_BINARY(Div__, DIV)  CG_BINARY(Pow__, POW)  CG_BINARY(Matmul_, MATMUL)

    // Reduce
    Napi::Value Sum_(const Napi::CallbackInfo& info) {
        int a = info[0].As<Napi::Number>().Int32Value();
        int dim = info.Length() > 1 ? info[1].As<Napi::Number>().Int32Value() : -1;
        bool keepdim = info.Length() > 2 ? info[2].As<Napi::Boolean>().Value() : false;
        return Napi::Number::New(info.Env(), graph_.op1(OpType::SUM, a, 0, dim, keepdim ? 1 : 0));
    }
    Napi::Value Mean_(const Napi::CallbackInfo& info) {
        int a = info[0].As<Napi::Number>().Int32Value();
        int dim = info.Length() > 1 ? info[1].As<Napi::Number>().Int32Value() : -1;
        bool keepdim = info.Length() > 2 ? info[2].As<Napi::Boolean>().Value() : false;
        return Napi::Number::New(info.Env(), graph_.op1(OpType::MEAN, a, 0, dim, keepdim ? 1 : 0));
    }

    // View
    Napi::Value Transpose_(const Napi::CallbackInfo& info) {
        int a = info[0].As<Napi::Number>().Int32Value();
        if (info.Length() >= 3) {
            int d0 = info[1].As<Napi::Number>().Int32Value();
            int d1 = info[2].As<Napi::Number>().Int32Value();
            return Napi::Number::New(info.Env(), graph_.op1(OpType::TRANSPOSE2, a, 0, d0, d1));
        }
        return Napi::Number::New(info.Env(), graph_.op1(OpType::TRANSPOSE, a));
    }

    // Reshape
    Napi::Value Reshape_(const Napi::CallbackInfo& info) {
        int a = info[0].As<Napi::Number>().Int32Value();
        Napi::Array shape = info[1].As<Napi::Array>();
        std::vector<int> s;
        for (uint32_t i = 0; i < shape.Length(); i++)
            s.push_back(shape.Get(i).As<Napi::Number>().Int32Value());
        return Napi::Number::New(info.Env(), graph_.reshape(a, s));
    }

    Napi::Value Flatten_(const Napi::CallbackInfo& info) {
        int a = info[0].As<Napi::Number>().Int32Value();
        int start_dim = info.Length() > 1 ? info[1].As<Napi::Number>().Int32Value() : 1;
        return Napi::Number::New(info.Env(), graph_.flatten(a, start_dim));
    }

    // Misc ops
    Napi::Value Cat_(const Napi::CallbackInfo& info) {
        auto arr = info[0].As<Napi::Array>();
        std::vector<int> inputs;
        for (uint32_t i = 0; i < arr.Length(); i++)
            inputs.push_back(((Napi::Value)arr[i]).As<Napi::Number>().Int32Value());
        int dim = info[1].As<Napi::Number>().Int32Value();
        return Napi::Number::New(info.Env(), graph_.cat(inputs, dim));
    }
    Napi::Value Slice_(const Napi::CallbackInfo& info) {
        int input = info[0].As<Napi::Number>().Int32Value();
        int dim = info[1].As<Napi::Number>().Int32Value();
        int start = info[2].As<Napi::Number>().Int32Value();
        int end = info[3].As<Napi::Number>().Int32Value();
        return Napi::Number::New(info.Env(), graph_.slice(input, dim, start, end));
    }
    Napi::Value Unsqueeze_(const Napi::CallbackInfo& info) {
        int input = info[0].As<Napi::Number>().Int32Value();
        int dim = info[1].As<Napi::Number>().Int32Value();
        return Napi::Number::New(info.Env(), graph_.unsqueeze(input, dim));
    }
    Napi::Value Squeeze_(const Napi::CallbackInfo& info) {
        int input = info[0].As<Napi::Number>().Int32Value();
        int dim = info[1].As<Napi::Number>().Int32Value();
        return Napi::Number::New(info.Env(), graph_.squeeze(input, dim));
    }

    // CNN ops
    Napi::Value Conv2d_(const Napi::CallbackInfo& info) {
        int input = info[0].As<Napi::Number>().Int32Value();
        int weight = info[1].As<Napi::Number>().Int32Value();
        int bias = info[2].IsNull() || info[2].IsUndefined() ? -1 : info[2].As<Napi::Number>().Int32Value();
        int sH = info[3].As<Napi::Number>().Int32Value();
        int sW = info[4].As<Napi::Number>().Int32Value();
        int pH = info[5].As<Napi::Number>().Int32Value();
        int pW = info[6].As<Napi::Number>().Int32Value();
        int dH = info[7].As<Napi::Number>().Int32Value();
        int dW = info[8].As<Napi::Number>().Int32Value();
        int groups = info[9].As<Napi::Number>().Int32Value();
        return Napi::Number::New(info.Env(), graph_.conv2d(input, weight, bias, sH, sW, pH, pW, dH, dW, groups));
    }

    Napi::Value MaxPool2d_(const Napi::CallbackInfo& info) {
        int input = info[0].As<Napi::Number>().Int32Value();
        int kH = info[1].As<Napi::Number>().Int32Value();
        int kW = info[2].As<Napi::Number>().Int32Value();
        int sH = info[3].As<Napi::Number>().Int32Value();
        int sW = info[4].As<Napi::Number>().Int32Value();
        int pH = info[5].As<Napi::Number>().Int32Value();
        int pW = info[6].As<Napi::Number>().Int32Value();
        return Napi::Number::New(info.Env(), graph_.max_pool2d(input, kH, kW, sH, sW, pH, pW));
    }

    Napi::Value AvgPool2d_(const Napi::CallbackInfo& info) {
        int input = info[0].As<Napi::Number>().Int32Value();
        int kH = info[1].As<Napi::Number>().Int32Value();
        int kW = info[2].As<Napi::Number>().Int32Value();
        int sH = info[3].As<Napi::Number>().Int32Value();
        int sW = info[4].As<Napi::Number>().Int32Value();
        int pH = info[5].As<Napi::Number>().Int32Value();
        int pW = info[6].As<Napi::Number>().Int32Value();
        bool cip = info.Length() > 7 ? info[7].As<Napi::Boolean>().Value() : true;
        return Napi::Number::New(info.Env(), graph_.avg_pool2d(input, kH, kW, sH, sW, pH, pW, cip));
    }

    Napi::Value BatchNorm2d_(const Napi::CallbackInfo& info) {
        int input = info[0].As<Napi::Number>().Int32Value();
        int weight = info[1].As<Napi::Number>().Int32Value();
        int bias = info[2].As<Napi::Number>().Int32Value();
        int rmean = info[3].As<Napi::Number>().Int32Value();
        int rvar = info[4].As<Napi::Number>().Int32Value();
        float eps = info.Length() > 5 ? info[5].As<Napi::Number>().FloatValue() : 1e-5f;
        return Napi::Number::New(info.Env(), graph_.batch_norm2d(input, weight, bias, rmean, rvar, eps));
    }

    Napi::Value Buffer(const Napi::CallbackInfo& info) {
        return Napi::Number::New(info.Env(), graph_.buffer());
    }

    // Loop
    Napi::Value LoopBegin(const Napi::CallbackInfo& info) {
        graph_.loop_begin(info[0].As<Napi::Number>().Int32Value());
        return info.Env().Undefined();
    }
    Napi::Value LoopSlice(const Napi::CallbackInfo& info) {
        int seq = info[0].As<Napi::Number>().Int32Value();
        int dim = info[1].As<Napi::Number>().Int32Value();
        return Napi::Number::New(info.Env(), graph_.loop_slice(seq, dim));
    }
    Napi::Value LoopCarry(const Napi::CallbackInfo& info) {
        int from = info[0].As<Napi::Number>().Int32Value();
        int to = info[1].As<Napi::Number>().Int32Value();
        graph_.loop_carry(from, to);
        return info.Env().Undefined();
    }
    Napi::Value LoopEnd(const Napi::CallbackInfo& info) {
        graph_.loop_end();
        return info.Env().Undefined();
    }

    // rnn_cudnn(x_slot, hx_slot, cx_slot, wih_slots[], whh_slots[], bih_slots[], bhh_slots[],
    //           mode, hidden_size, num_layers, bidirectional)
    // returns [y_slot, hy_slot, cy_slot]
    Napi::Value RnnCudnn(const Napi::CallbackInfo& info) {
        auto env = info.Env();
        int x_slot = info[0].As<Napi::Number>().Int32Value();
        int hx_slot = info[1].IsNull() || info[1].IsUndefined() ? -1 : info[1].As<Napi::Number>().Int32Value();
        int cx_slot = info[2].IsNull() || info[2].IsUndefined() ? -1 : info[2].As<Napi::Number>().Int32Value();

        auto extractSlots = [](const Napi::Array& arr) -> std::vector<int> {
            std::vector<int> v;
            for (uint32_t i = 0; i < arr.Length(); i++)
                v.push_back(arr.Get(i).As<Napi::Number>().Int32Value());
            return v;
        };
        auto wih = extractSlots(info[3].As<Napi::Array>());
        auto whh = extractSlots(info[4].As<Napi::Array>());
        auto bih = extractSlots(info[5].As<Napi::Array>());
        auto bhh = extractSlots(info[6].As<Napi::Array>());
        int mode = info[7].As<Napi::Number>().Int32Value();
        int hidden = info[8].As<Napi::Number>().Int32Value();
        int layers = info[9].As<Napi::Number>().Int32Value();
        int bidir = info[10].As<Napi::Number>().Int32Value();

        auto result = graph_.rnn_cudnn(x_slot, hx_slot, cx_slot, wih, whh, bih, bhh,
                                        mode, hidden, layers, bidir);
        auto arr = Napi::Array::New(env, 3);
        arr.Set((uint32_t)0, Napi::Number::New(env, result[0]));
        arr.Set((uint32_t)1, Napi::Number::New(env, result[1]));
        arr.Set((uint32_t)2, Napi::Number::New(env, result[2]));
        return arr;
    }

    // Run: forward + backward + Adam, all in C++
    Napi::Value Run(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        // args: input(Tensor), target(Tensor), params(Tensor[]), m(Tensor[]), v(Tensor[]),
        //       lr, beta1, beta2, eps, bc1, bc2, wd
        auto* inp = Napi::ObjectWrap<TensorWrap>::Unwrap(info[0].As<Napi::Object>());
        auto* tgt = Napi::ObjectWrap<TensorWrap>::Unwrap(info[1].As<Napi::Object>());

        Napi::Array p_arr = info[2].As<Napi::Array>();
        Napi::Array m_arr = info[3].As<Napi::Array>();
        Napi::Array v_arr = info[4].As<Napi::Array>();
        uint32_t np = p_arr.Length();

        std::vector<Tensor> params, ms, vs;
        params.reserve(np); ms.reserve(np); vs.reserve(np);
        for (uint32_t i = 0; i < np; i++) {
            params.push_back(Napi::ObjectWrap<TensorWrap>::Unwrap(p_arr.Get(i).As<Napi::Object>())->tensor_);
            ms.push_back(Napi::ObjectWrap<TensorWrap>::Unwrap(m_arr.Get(i).As<Napi::Object>())->tensor_);
            vs.push_back(Napi::ObjectWrap<TensorWrap>::Unwrap(v_arr.Get(i).As<Napi::Object>())->tensor_);
        }

        float lr  = info[5].As<Napi::Number>().FloatValue();
        float b1  = info[6].As<Napi::Number>().FloatValue();
        float b2  = info[7].As<Napi::Number>().FloatValue();
        float eps = info[8].As<Napi::Number>().FloatValue();
        float bc1 = info[9].As<Napi::Number>().FloatValue();
        float bc2 = info[10].As<Napi::Number>().FloatValue();
        float wd  = info[11].As<Napi::Number>().FloatValue();

        // Optional buffers array (arg 12)
        std::vector<Tensor> bufs;
        std::vector<Tensor>* bufs_ptr = nullptr;
        if (info.Length() > 12 && info[12].IsArray()) {
            Napi::Array b_arr = info[12].As<Napi::Array>();
            uint32_t nb = b_arr.Length();
            bufs.reserve(nb);
            for (uint32_t i = 0; i < nb; i++)
                bufs.push_back(Napi::ObjectWrap<TensorWrap>::Unwrap(b_arr.Get(i).As<Napi::Object>())->tensor_);
            bufs_ptr = &bufs;
        }

        // GradTensor-based run() — supports all ops and dynamic batch sizes.
        graph_.run(inp->tensor_, tgt->tensor_, params, ms, vs, lr, b1, b2, eps, bc1, bc2, wd, bufs_ptr);

        return env.Undefined();
    }
};

// (Handle API removed — was zero-overhead experiment, no callers.)

Napi::Object Init(Napi::Env env, Napi::Object exports) {
    TensorWrap::Init(env, exports);
    GradTensorWrap::Init(env, exports);
    CompiledGraphWrap::Init(env, exports);
    return exports;
}

NODE_API_MODULE(jstorch, Init)
