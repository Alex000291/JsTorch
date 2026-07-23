#pragma once
#include <vector>
#include <numeric>
#include <algorithm>
#include <stdexcept>

namespace jstorch {

using Shape = std::vector<int>;
using Strides = std::vector<int>;

inline Strides compute_strides(const Shape& shape) {
    Strides strides(shape.size());
    int stride = 1;
    for (int i = shape.size() - 1; i >= 0; i--) {
        strides[i] = stride;
        stride *= shape[i];
    }
    return strides;
}

inline int total_size(const Shape& shape) {
    return shape.empty() ? 0 : std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
}

inline bool is_contiguous(const Shape& shape, const Strides& strides) {
    int expected = 1;
    for (int i = shape.size() - 1; i >= 0; i--) {
        if (strides[i] != expected) return false;
        expected *= shape[i];
    }
    return true;
}

inline Shape broadcast_shape(const Shape& s1, const Shape& s2) {
    size_t max_dim = std::max(s1.size(), s2.size());
    Shape result(max_dim);
    
    for (int i = 0; i < max_dim; i++) {
        int idx1 = i - (max_dim - s1.size());
        int idx2 = i - (max_dim - s2.size());
        
        int d1 = idx1 >= 0 ? s1[idx1] : 1;
        int d2 = idx2 >= 0 ? s2[idx2] : 1;
        
        if (d1 == d2) result[i] = d1;
        else if (d1 == 1) result[i] = d2;
        else if (d2 == 1) result[i] = d1;
        else throw std::runtime_error("Broadcast shape mismatch");
    }
    return result;
}

inline int normalize_axis(int axis, int ndim) {
    if (axis < 0) axis += ndim;
    if (axis < 0 || axis >= ndim) throw std::runtime_error("Axis out of range");
    return axis;
}

}
