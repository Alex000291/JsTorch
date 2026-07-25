#pragma once
#include <cuda_runtime.h>
#include <mutex>
#include <set>
#include <map>
#include <vector>
#include <cstdint>
#include <stdexcept>

namespace jstorch {

// ==============================================================================
// CudaAllocator: PyTorch-style sub-allocation with free-list and coalescing.
//
// - Pre-allocates large GPU segments (256MB default, grows as needed)
// - Internal free-list sorted by address for O(log N) best-fit + coalescing
// - Never calls cudaFree (except at shutdown)
// - All allocations aligned to 512 bytes
// ==============================================================================

class CudaAllocator {
    static constexpr size_t ALIGNMENT = 512;
    static constexpr size_t SEGMENT_SIZE = 256 * 1024 * 1024; // 256 MB
    static constexpr size_t MIN_SPLIT = 512; // Don't split if remainder < this

    struct Block {
        uintptr_t addr;
        size_t size;
        bool operator<(const Block& o) const { return addr < o.addr; }
    };

    // Free blocks sorted by address (for coalescing)
    std::set<Block> free_blocks_;
    // Allocated blocks: addr → size
    std::map<uintptr_t, size_t> alloc_map_;
    // All segments (for cleanup)
    std::vector<void*> segments_;
    std::mutex mu_;

    static size_t align_up(size_t n) {
        return (n + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
    }

    // Allocate a new segment from CUDA
    void add_segment(size_t min_size) {
        size_t seg_size = (min_size > SEGMENT_SIZE) ? align_up(min_size) : SEGMENT_SIZE;
        void* ptr = nullptr;
        cudaError_t err = cudaMalloc(&ptr, seg_size);
        if (err != cudaSuccess || !ptr)
            throw std::runtime_error("cudaMalloc failed for segment");
        segments_.push_back(ptr);
        free_blocks_.insert({(uintptr_t)ptr, seg_size});
    }

public:
    void* allocate(size_t bytes) {
        if (bytes == 0) bytes = ALIGNMENT;
        size_t size = align_up(bytes);

        std::lock_guard<std::mutex> lock(mu_);

        // Best-fit: find smallest free block >= size
        Block best = {0, SIZE_MAX};
        for (auto& b : free_blocks_) {
            if (b.size >= size && b.size < best.size)
                best = b;
        }

        if (best.size == SIZE_MAX) {
            // No suitable block — allocate new segment
            add_segment(size);
            // Retry: the new segment is guaranteed to fit
            for (auto& b : free_blocks_) {
                if (b.size >= size && b.size < best.size)
                    best = b;
            }
        }

        free_blocks_.erase(best);

        // Split if remainder is large enough
        if (best.size - size >= MIN_SPLIT) {
            free_blocks_.insert({best.addr + size, best.size - size});
        } else {
            size = best.size; // use full block, avoid tiny fragment
        }

        alloc_map_[best.addr] = size;
        return (void*)best.addr;
    }

    void free(void* ptr, size_t /*bytes*/) {
        if (!ptr) return;
        uintptr_t addr = (uintptr_t)ptr;

        std::lock_guard<std::mutex> lock(mu_);

        auto it = alloc_map_.find(addr);
        if (it == alloc_map_.end()) return; // Double free protection
        size_t size = it->second;
        alloc_map_.erase(it);

        // Insert and coalesce with neighbors
        Block b = {addr, size};

        // Try merge with right neighbor
        auto right = free_blocks_.lower_bound({addr + size, 0});
        if (right != free_blocks_.end() && right->addr == addr + size) {
            b.size += right->size;
            free_blocks_.erase(right);
        }

        // Try merge with left neighbor
        if (!free_blocks_.empty()) {
            auto left = free_blocks_.lower_bound({addr, 0});
            if (left != free_blocks_.begin()) {
                --left;
                if (left->addr + left->size == addr) {
                    b.addr = left->addr;
                    b.size += left->size;
                    free_blocks_.erase(left);
                }
            }
        }

        free_blocks_.insert(b);
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

}
