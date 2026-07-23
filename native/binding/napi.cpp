#include <napi.h>
#include "../core/tensor.hpp"
#include <vector>

using namespace jstorch;

// Helper: 递归构建nested array
Napi::Value buildNestedArray(Napi::Env env, const std::vector<float>& data, const Shape& shape, int dim, int offset) {
    if (dim == shape.size() - 1) {
        Napi::Array arr = Napi::Array::New(env, shape[dim]);
        for (int i = 0; i < shape[dim]; i++) {
            arr[i] = Napi::Number::New(env, data[offset + i]);
        }
        return arr;
    }
    
    int stride = 1;
    for (int i = dim + 1; i < shape.size(); i++) stride *= shape[i];
    
    Napi::Array arr = Napi::Array::New(env, shape[dim]);
    for (int i = 0; i < shape[dim]; i++) {
        arr[i] = buildNestedArray(env, data, shape, dim + 1, offset + i * stride);
    }
    return arr;
}

// Helper: 解析JS array到flat vector
void parseArray(Napi::Value val, std::vector<float>& data) {
    if (val.IsNumber()) {
        data.push_back(val.As<Napi::Number>().FloatValue());
    } else if (val.IsArray()) {
        Napi::Array arr = val.As<Napi::Array>();
        for (uint32_t i = 0; i < arr.Length(); i++) {
            parseArray(arr[i], data);
        }
    }
}

Shape inferShape(Napi::Value val) {
    if (val.IsNumber()) return {};
    
    Shape shape;
    Napi::Value current = val;
    while (current.IsArray()) {
        Napi::Array arr = current.As<Napi::Array>();
        shape.push_back(arr.Length());
        current = arr.Get(uint32_t(0));
    }
    return shape;
}

// Tensor wrapper
class TensorWrap : public Napi::ObjectWrap<TensorWrap> {
private:
    Tensor tensor_;

public:
    static Napi::Object Init(Napi::Env env, Napi::Object exports) {
        Napi::Function func = DefineClass(env, "Tensor", {
            InstanceAccessor("shape", &TensorWrap::GetShape, nullptr),
            InstanceMethod("toArray", &TensorWrap::ToArray),
            InstanceMethod("add", &TensorWrap::add),
            InstanceMethod("sub", &TensorWrap::sub),
            InstanceMethod("mul", &TensorWrap::mul),
            InstanceMethod("div", &TensorWrap::div),
            InstanceMethod("abs", &TensorWrap::abs),
            InstanceMethod("sqrt", &TensorWrap::sqrt),
            InstanceMethod("square", &TensorWrap::square),
            InstanceMethod("exp", &TensorWrap::exp),
            InstanceMethod("log", &TensorWrap::log),
            InstanceMethod("sin", &TensorWrap::sin),
            InstanceMethod("cos", &TensorWrap::cos),
            InstanceMethod("sigmoid", &TensorWrap::sigmoid),
            InstanceMethod("tanh", &TensorWrap::tanh),
            InstanceMethod("relu", &TensorWrap::relu),
            InstanceMethod("sum", &TensorWrap::sum),
            InstanceMethod("mean", &TensorWrap::mean),
            InstanceMethod("reshape", &TensorWrap::reshape),
            InstanceMethod("transpose", &TensorWrap::transpose),
        });
        
        Napi::FunctionReference* constructor = new Napi::FunctionReference();
        *constructor = Napi::Persistent(func);
        env.SetInstanceData(constructor);
        
        exports.Set("Tensor", func);
        return exports;
    }
    
    TensorWrap(const Napi::CallbackInfo& info) : Napi::ObjectWrap<TensorWrap>(info), tensor_(Shape{1}) {
        Napi::Env env = info.Env();
        
        if (info[0].IsArray()) {
            std::vector<float> data;
            parseArray(info[0], data);
            Shape shape = inferShape(info[0]);
            tensor_ = Tensor::from_array(data.data(), shape);
        }
    }
    
    Tensor& tensor() { return tensor_; }
    
    Napi::Value GetShape(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        Napi::Array arr = Napi::Array::New(env, tensor_.shape().size());
        for (size_t i = 0; i < tensor_.shape().size(); i++) {
            arr[i] = Napi::Number::New(env, tensor_.shape()[i]);
        }
        return arr;
    }
    
    Napi::Value ToArray(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        auto data = tensor_.to_array();
        return buildNestedArray(env, data, tensor_.shape(), 0, 0);
    }
    
    #define DEFINE_BINARY_METHOD(name) \
    Napi::Value name(const Napi::CallbackInfo& info) { \
        Napi::Env env = info.Env(); \
        TensorWrap* other = Napi::ObjectWrap<TensorWrap>::Unwrap(info[0].As<Napi::Object>()); \
        Tensor result = tensor_.name(other->tensor()); \
        auto* ctor = env.GetInstanceData<Napi::FunctionReference>(); \
        Napi::Object obj = ctor->New({}); \
        TensorWrap* wrap = Napi::ObjectWrap<TensorWrap>::Unwrap(obj); \
        wrap->tensor_ = result; \
        return obj; \
    }
    
    DEFINE_BINARY_METHOD(add)
    DEFINE_BINARY_METHOD(sub)
    DEFINE_BINARY_METHOD(mul)
    DEFINE_BINARY_METHOD(div)
    
    #define DEFINE_UNARY_METHOD(name) \
    Napi::Value name(const Napi::CallbackInfo& info) { \
        Napi::Env env = info.Env(); \
        Tensor result = tensor_.name(); \
        auto* ctor = env.GetInstanceData<Napi::FunctionReference>(); \
        Napi::Object obj = ctor->New({}); \
        TensorWrap* wrap = Napi::ObjectWrap<TensorWrap>::Unwrap(obj); \
        wrap->tensor_ = result; \
        return obj; \
    }
    
    DEFINE_UNARY_METHOD(abs)
    DEFINE_UNARY_METHOD(sqrt)
    DEFINE_UNARY_METHOD(square)
    DEFINE_UNARY_METHOD(exp)
    DEFINE_UNARY_METHOD(log)
    DEFINE_UNARY_METHOD(sin)
    DEFINE_UNARY_METHOD(cos)
    DEFINE_UNARY_METHOD(sigmoid)
    DEFINE_UNARY_METHOD(tanh)
    DEFINE_UNARY_METHOD(relu)
    DEFINE_UNARY_METHOD(transpose)
    
    Napi::Value sum(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        int dim = info.Length() > 0 ? info[0].As<Napi::Number>().Int32Value() : -1;
        Tensor result = tensor_.sum(dim, false);
        auto* ctor = env.GetInstanceData<Napi::FunctionReference>();
        Napi::Object obj = ctor->New({});
        TensorWrap* wrap = Napi::ObjectWrap<TensorWrap>::Unwrap(obj);
        wrap->tensor_ = result;
        return obj;
    }
    
    Napi::Value mean(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        int dim = info.Length() > 0 ? info[0].As<Napi::Number>().Int32Value() : -1;
        Tensor result = tensor_.mean(dim, false);
        auto* ctor = env.GetInstanceData<Napi::FunctionReference>();
        Napi::Object obj = ctor->New({});
        TensorWrap* wrap = Napi::ObjectWrap<TensorWrap>::Unwrap(obj);
        wrap->tensor_ = result;
        return obj;
    }
    
    Napi::Value reshape(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        Napi::Array arr = info[0].As<Napi::Array>();
        Shape shape;
        for (uint32_t i = 0; i < arr.Length(); i++) {
            shape.push_back(arr.Get(i).As<Napi::Number>().Int32Value());
        }
        Tensor result = tensor_.reshape(shape);
        auto* ctor = env.GetInstanceData<Napi::FunctionReference>();
        Napi::Object obj = ctor->New({});
        TensorWrap* wrap = Napi::ObjectWrap<TensorWrap>::Unwrap(obj);
        wrap->tensor_ = result;
        return obj;
    }
};

Napi::Object Init(Napi::Env env, Napi::Object exports) {
    return TensorWrap::Init(env, exports);
}

NODE_API_MODULE(jstorch, Init)
