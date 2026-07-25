#pragma once
#include <cuda_runtime.h>
#include <map>
#include <vector>
#include <mutex>

namespace jstorch {

class CudaAllocator {
    std::map<size_t, std::vector<void*>> free_pools_;
    std::mutex mu_;
    
    static size_t round_up(size_t bytes) {
        if (bytes <= 512) return 512;
        // Round to next power of 2
        size_t v = bytes - 1;
        v |= v >> 1; v |= v >> 2; v |= v >> 4;
        v |= v >> 8; v |= v >> 16; v |= v >> 32;
        return v + 1;
    }
    
public:
    void* allocate(size_t bytes) {
        if (bytes == 0) bytes = 4;
        size_t bucket = round_up(bytes);
        std::lock_guard<std::mutex> lock(mu_);
        auto it = free_pools_.find(bucket);
        if (it != free_pools_.end() && !it->second.empty()) {
            void* ptr = it->second.back();
            it->second.pop_back();
            return ptr;
        }
        void* ptr;
        cudaMalloc(&ptr, bucket);
        return ptr;
    }
    
    void free(void* ptr, size_t bytes) {
        if (!ptr) return;
        size_t bucket = round_up(bytes);
        std::lock_guard<std::mutex> lock(mu_);
        free_pools_[bucket].push_back(ptr);
    }
    
    ~CudaAllocator() {
        for (auto& [sz, ptrs] : free_pools_)
            for (auto* p : ptrs) cudaFree(p);
    }
};

inline CudaAllocator& get_allocator() {
    static CudaAllocator alloc;
    return alloc;
}

}
