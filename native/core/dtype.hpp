#pragma once
#include <cuda_runtime.h>

namespace jstorch {

enum class DType {
    Float32
};

inline size_t dtype_size(DType) {
    return sizeof(float);
}

inline const char* dtype_name(DType) {
    return "float32";
}

}
