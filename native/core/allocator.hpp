#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <stdexcept>

namespace jstorch {

// ==============================================================================
// CudaAllocator: Size-binned O(1) sub-allocation.
//
// - Pre-allocates large GPU segments (256MB default, grows as needed)
// - Size-binned free lists: same-size allocs are O(1) pop/push
// - No mutex (JS is single-threaded)
// - Never calls cudaFree (except at shutdown)
// - All allocations aligned to 512 bytes
// ==============================================================================

class CudaAllocator {
    static constexpr size_t ALIGNMENT = 512;
    static constexpr size_t SEGMENT_SIZE = 256 * 1024 * 1024; // 256 MB

    // Size-binned free lists: aligned_size → [ptr, ptr, ...]
    std::unordered_map<size_t, std::vector<uintptr_t>> bins_;
    // All segments (for cleanup)
    std::vector<void*> segments_;
    // Current segment bump pointer
    uintptr_t bump_ptr_ = 0;
    size_t bump_remaining_ = 0;

    static size_t align_up(size_t n) {
        return (n + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
    }

    void add_segment(size_t min_size) {
        size_t seg_size = (min_size > SEGMENT_SIZE) ? align_up(min_size) : SEGMENT_SIZE;
        void* ptr = nullptr;
        cudaError_t err = cudaMalloc(&ptr, seg_size);
        if (err != cudaSuccess || !ptr)
            throw std::runtime_error("cudaMalloc failed for segment");
        segments_.push_back(ptr);
        bump_ptr_ = (uintptr_t)ptr;
        bump_remaining_ = seg_size;
    }

public:
    void* allocate(size_t bytes) {
        if (bytes == 0) bytes = ALIGNMENT;
        size_t size = align_up(bytes);

        // O(1) path: check size bin
        auto it = bins_.find(size);
        if (it != bins_.end() && !it->second.empty()) {
            uintptr_t addr = it->second.back();
            it->second.pop_back();
            return (void*)addr;
        }

        // Bump allocate from current segment
        if (bump_remaining_ < size) {
            add_segment(size);
        }

        void* ptr = (void*)bump_ptr_;
        bump_ptr_ += size;
        bump_remaining_ -= size;
        return ptr;
    }

    void free(void* ptr, size_t bytes) {
        if (!ptr) return;
        size_t size = align_up(bytes);
        bins_[size].push_back((uintptr_t)ptr);
    }

    ~CudaAllocator() {
        for (auto* p : segments_)
            cudaFree(p);
    }
};

inline CudaAllocator& get_allocator() {
    static CudaAllocator alloc;
    return alloc;
}

// ==============================================================================
// ArenaAllocator: Bump allocator for CompiledGraph execution.
// Zero overhead: O(1) allocate, no mutex, no free list.
// Reset offset to 0 at start of each run_tape() call.
// ==============================================================================
class ArenaAllocator {
    static constexpr size_t ALIGNMENT = 512;
    void* base_ = nullptr;
    size_t capacity_ = 0;
    size_t offset_ = 0;

    static size_t align_up(size_t n) {
        return (n + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
    }

public:
    void ensure(size_t min_cap) {
        if (capacity_ >= min_cap) return;
        if (base_) cudaFree(base_);
        // Add 20% headroom
        capacity_ = min_cap + min_cap / 5;
        capacity_ = align_up(capacity_);
        cudaError_t err = cudaMalloc(&base_, capacity_);
        if (err != cudaSuccess)
            throw std::runtime_error("Arena cudaMalloc failed");
    }

    void reset() { offset_ = 0; }

    void* allocate(size_t bytes) {
        if (bytes == 0) bytes = ALIGNMENT;
        size_t size = align_up(bytes);
        if (offset_ + size > capacity_) {
            // Grow arena
            size_t new_cap = (offset_ + size) * 2;
            void* new_base = nullptr;
            cudaError_t err = cudaMalloc(&new_base, new_cap);
            if (err != cudaSuccess)
                throw std::runtime_error("Arena grow failed");
            cudaMemcpy(new_base, base_, offset_, cudaMemcpyDeviceToDevice);
            if (base_) cudaFree(base_);
            base_ = new_base;
            capacity_ = new_cap;
        }
        void* ptr = (char*)base_ + offset_;
        offset_ += size;
        return ptr;
    }

    void free(void*, size_t) {} // no-op

    size_t used() const { return offset_; }
    size_t cap() const { return capacity_; }

    ~ArenaAllocator() { if (base_) cudaFree(base_); }
};

// Thread-local arena mode: when active, allocations go to arena instead of pool
inline thread_local ArenaAllocator* active_arena = nullptr;

inline void* alloc_gpu(size_t bytes) {
    if (active_arena) return active_arena->allocate(bytes);
    return get_allocator().allocate(bytes);
}

inline void free_gpu(void* ptr, size_t bytes) {
    if (active_arena) { active_arena->free(ptr, bytes); return; }
    get_allocator().free(ptr, bytes);
}

}
