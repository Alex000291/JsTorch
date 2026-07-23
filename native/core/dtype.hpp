#pragma once
#include <cuda_runtime.h>
#include <cuComplex.h>

namespace jstorch {

enum class DType {
    Float32,
    Complex64
};

inline size_t dtype_size(DType dtype) {
    return dtype == DType::Float32 ? sizeof(float) : sizeof(cuComplex);
}

inline const char* dtype_name(DType dtype) {
    return dtype == DType::Float32 ? "float32" : "complex64";
}

}
