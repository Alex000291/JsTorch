#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <stdexcept>
#include <algorithm>

namespace jstorch {

// ==============================================================================
// CudaAllocator: Size-binned O(1) sub-allocation with per-segment live tracking.
//
// A CPU-side map (ptr_to_seg_) records which segment each allocation came from.
// Each Segment tracks how many live user-held allocs it has.
// trim() cudaFrees any segment whose live count is zero (all allocs returned to
// bins), then clears the free lists so the next batch starts fresh.
// ==============================================================================

class CudaAllocator {
    static constexpr size_t ALIGNMENT    = 512;
    static constexpr size_t SEGMENT_SIZE = 256ULL * 1024 * 1024; // 256 MB

    struct Segment {
        void*    ptr  = nullptr;
        size_t   size = 0;
        uint32_t live = 0;   // live user-held allocs (not yet freed to bins)
    };

    std::vector<Segment>                               segments_;
    std::unordered_map<size_t, std::vector<uintptr_t>> bins_;       // user_size → [ptr…]
    std::unordered_map<uintptr_t, uint32_t>            ptr_to_seg_; // ptr → seg_idx
    uintptr_t bump_ptr_       = 0;
    size_t    bump_remaining_ = 0;
    uint32_t  cur_seg_idx_    = 0;

    static size_t align_up(size_t n) {
        return (n + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
    }

    void add_segment(size_t min_size) {
        size_t seg_size = (min_size > SEGMENT_SIZE)
                          ? align_up(min_size) : SEGMENT_SIZE;
        void* ptr = nullptr;
        if (cudaMalloc(&ptr, seg_size) != cudaSuccess || !ptr)
            throw std::runtime_error("cudaMalloc failed for segment");
        cur_seg_idx_ = (uint32_t)segments_.size();
        segments_.push_back({ptr, seg_size, 0});
        bump_ptr_       = (uintptr_t)ptr;
        bump_remaining_ = seg_size;
    }

public:
    void* allocate(size_t bytes) {
        if (bytes == 0) bytes = ALIGNMENT;
        size_t size = align_up(bytes);

        // O(1) reuse from bin
        auto it = bins_.find(size);
        if (it != bins_.end() && !it->second.empty()) {
            uintptr_t addr = it->second.back();
            it->second.pop_back();
            auto jt = ptr_to_seg_.find(addr);
            if (jt != ptr_to_seg_.end()) {
                segments_[jt->second].live++;
                return (void*)addr;
            }
            // entry missing (shouldn't happen): fall through to bump
        }

        // Bump allocate from current segment
        if (bump_remaining_ < size)
            add_segment(size);

        void* ptr = (void*)bump_ptr_;
        ptr_to_seg_[(uintptr_t)ptr] = cur_seg_idx_;
        segments_[cur_seg_idx_].live++;
        bump_ptr_       += size;
        bump_remaining_ -= size;
        return ptr;
    }

    void free(void* ptr, size_t bytes) {
        if (!ptr) return;
        size_t size = align_up(bytes);
        auto it = ptr_to_seg_.find((uintptr_t)ptr);
        if (it != ptr_to_seg_.end()) {
            uint32_t sidx = it->second;
            if (sidx < (uint32_t)segments_.size() && segments_[sidx].ptr != nullptr) {
                // Segment still alive: decrement live count and return to bin.
                // Keep ptr_to_seg_ entry — allocate() needs it when reusing from bins_.
                segments_[sidx].live--;
                bins_[size].push_back((uintptr_t)ptr);
            } else {
                // Segment was already cudaFree'd by trim(): clean up stale entry.
                ptr_to_seg_.erase(it);
            }
        }
    }

    // Release all GPU segments whose live count is zero.
    // Should be called between benchmark batch sizes to reclaim VRAM.
    void trim() {
        cudaDeviceSynchronize();

        // Identify segments to free (live == 0 means all their allocs are in bins)
        for (uint32_t i = 0; i < (uint32_t)segments_.size(); i++) {
            if (!segments_[i].ptr || segments_[i].live != 0) continue;
            // Remove ptr_to_seg_ entries and bin entries for this segment
            for (auto& [sz, ptrs] : bins_) {
                ptrs.erase(
                    std::remove_if(ptrs.begin(), ptrs.end(),
                        [&](uintptr_t p) {
                            auto jt = ptr_to_seg_.find(p);
                            if (jt != ptr_to_seg_.end() && jt->second == i) {
                                ptr_to_seg_.erase(jt);
                                return true;
                            }
                            return false;
                        }),
                    ptrs.end());
            }
            cudaFree(segments_[i].ptr);
            segments_[i].ptr = nullptr;
        }

        bins_.clear();
        bump_ptr_       = 0;
        bump_remaining_ = 0;
        // Next allocate() will call add_segment() as needed.
    }

    ~CudaAllocator() {
        for (auto& s : segments_)
            if (s.ptr) cudaFree(s.ptr);
    }
};

inline CudaAllocator& get_allocator() {
    static CudaAllocator alloc;
    return alloc;
}

// ==============================================================================
// ArenaAllocator: Bump allocator for CompiledGraph execution.
// ==============================================================================
class ArenaAllocator {
    static constexpr size_t ALIGNMENT = 512;
    void*  base_     = nullptr;
    size_t capacity_ = 0;
    size_t offset_   = 0;

    static size_t align_up(size_t n) {
        return (n + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
    }

public:
    void ensure(size_t min_cap) {
        if (capacity_ >= min_cap) return;
        if (base_) cudaFree(base_);
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
            size_t new_cap = (offset_ + size) * 2;
            void* new_base = nullptr;
            cudaError_t err = cudaMalloc(&new_base, new_cap);
            if (err != cudaSuccess)
                throw std::runtime_error("Arena grow failed");
            cudaMemcpy(new_base, base_, offset_, cudaMemcpyDeviceToDevice);
            if (base_) cudaFree(base_);
            base_     = new_base;
            capacity_ = new_cap;
        }
        void* ptr = (char*)base_ + offset_;
        offset_ += size;
        return ptr;
    }

    void free(void*, size_t) {}

    size_t used() const { return offset_; }
    size_t cap()  const { return capacity_; }

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
