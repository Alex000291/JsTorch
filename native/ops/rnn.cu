#include <cudnn.h>
#include <cuda_runtime.h>
#include <map>
#include <vector>
#include <cstring>
#include <stdexcept>

namespace jstorch {

static cudnnHandle_t get_cudnn_handle() {
    static cudnnHandle_t h = nullptr;
    if (!h) cudnnCreate(&h);
    return h;
}

// Shared workspace (defined in conv.cu in jstorch::ops)
namespace ops { extern void* get_workspace(size_t needed); }
using ops::get_workspace;

// Cache key for RNN descriptors
struct RNNKey {
    int mode, input_size, hidden_size, num_layers, bidirectional, batch_size, seq_len;
    bool operator<(const RNNKey& o) const {
        return std::tie(mode, input_size, hidden_size, num_layers, bidirectional, batch_size, seq_len)
             < std::tie(o.mode, o.input_size, o.hidden_size, o.num_layers, o.bidirectional, o.batch_size, o.seq_len);
    }
};

struct RNNPlan {
    cudnnRNNDescriptor_t rnnDesc;
    cudnnRNNDataDescriptor_t xDesc, yDesc;
    cudnnTensorDescriptor_t hDesc, cDesc;
    size_t weight_space_size;
    size_t workspace_size;
    size_t reserve_size;
    int* dev_seq_lengths;  // device array of seq lengths
    void* reserve_space;   // per-plan (needed for backward to match forward)
    void* weight_space;    // cached (avoid malloc/free per call)
    void* dweight_space;   // cached gradient weight space
};

static std::map<RNNKey, RNNPlan> rnn_cache;

static RNNPlan& get_or_create_plan(int mode, int B, int seq_len, int input_size, int hidden_size,
                                    int num_layers, int bidirectional, cudnnHandle_t handle) {
    RNNKey key{mode, input_size, hidden_size, num_layers, bidirectional, B, seq_len};
    auto it = rnn_cache.find(key);
    if (it != rnn_cache.end()) return it->second;

    RNNPlan p{};
    int num_dir = bidirectional ? 2 : 1;

    // RNN descriptor
    cudnnStatus_t st;
    cudnnCreateRNNDescriptor(&p.rnnDesc);
    st = cudnnSetRNNDescriptor_v8(p.rnnDesc,
        CUDNN_RNN_ALGO_STANDARD,
        (cudnnRNNMode_t)mode,  // 0=RNN_RELU, 1=RNN_TANH, 2=LSTM, 3=GRU
        CUDNN_RNN_DOUBLE_BIAS,
        bidirectional ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL,
        CUDNN_LINEAR_INPUT,
        CUDNN_DATA_FLOAT, CUDNN_DATA_FLOAT,
        CUDNN_DEFAULT_MATH,
        input_size, hidden_size, hidden_size, // projSize=hiddenSize (no projection)
        num_layers, nullptr, // no dropout
        CUDNN_RNN_PADDED_IO_ENABLED);
    // if (st != CUDNN_STATUS_SUCCESS) printf("cudnnSetRNNDescriptor_v8: %d\n", st);

    // Build for this batch size
    st = cudnnBuildRNNDynamic(handle, p.rnnDesc, B);
    // if (st != CUDNN_STATUS_SUCCESS) printf("cudnnBuildRNNDynamic: %d\n", st);

    // Data descriptors (batch-major: [B, seq_len, features])
    std::vector<int> seq_lengths(B, seq_len);
    cudaMalloc(&p.dev_seq_lengths, B * sizeof(int));
    cudaMemcpy(p.dev_seq_lengths, seq_lengths.data(), B * sizeof(int), cudaMemcpyHostToDevice);

    cudnnCreateRNNDataDescriptor(&p.xDesc);
    cudnnSetRNNDataDescriptor(p.xDesc, CUDNN_DATA_FLOAT,
        CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED,
        seq_len, B, input_size, seq_lengths.data(), nullptr);

    int out_size = hidden_size * num_dir;
    cudnnCreateRNNDataDescriptor(&p.yDesc);
    cudnnSetRNNDataDescriptor(p.yDesc, CUDNN_DATA_FLOAT,
        CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED,
        seq_len, B, out_size, seq_lengths.data(), nullptr);

    // Hidden state descriptor: [num_layers * num_dir, B, hidden_size]
    cudnnCreateTensorDescriptor(&p.hDesc);
    int hDims[3] = {num_layers * num_dir, B, hidden_size};
    int hStrides[3] = {B * hidden_size, hidden_size, 1};
    cudnnSetTensorNdDescriptor(p.hDesc, CUDNN_DATA_FLOAT, 3, hDims, hStrides);

    // Cell state descriptor (same shape, used for LSTM)
    cudnnCreateTensorDescriptor(&p.cDesc);
    cudnnSetTensorNdDescriptor(p.cDesc, CUDNN_DATA_FLOAT, 3, hDims, hStrides);

    // Weight space
    cudnnGetRNNWeightSpaceSize(handle, p.rnnDesc, &p.weight_space_size);

    // Workspace + reserve
    cudnnGetRNNTempSpaceSizes(handle, p.rnnDesc, CUDNN_FWD_MODE_TRAINING, p.xDesc,
        &p.workspace_size, &p.reserve_size);

    // Reserve space is per-plan (links forward to backward)
    p.reserve_space = nullptr;
    if (p.reserve_size > 0) cudaMalloc(&p.reserve_space, p.reserve_size);

    // Cache weight space buffers (avoid per-call malloc/free)
    p.weight_space = nullptr;
    p.dweight_space = nullptr;
    if (p.weight_space_size > 0) {
        cudaMalloc(&p.weight_space, p.weight_space_size);
        cudaMalloc(&p.dweight_space, p.weight_space_size);
    }

    return rnn_cache.emplace(key, p).first->second;
}

// Pack JsTorch separate weights into cuDNN packed format
// mode: 0=RNN_RELU, 1=RNN_TANH(2 linLayers), 2=LSTM(8), 3=GRU(6)
static void pack_weights(cudnnHandle_t handle, const RNNPlan& plan,
                         const float* const* weights_ih, const float* const* weights_hh,
                         const float* const* biases_ih, const float* const* biases_hh,
                         void* weight_space, int num_layers, int num_dir,
                         int input_size, int hidden_size, int mode) {
    // Zero the weight space first
    cudaMemsetAsync(weight_space, 0, plan.weight_space_size, 0);

    int gates = (mode == 2) ? 4 : (mode == 3) ? 3 : 1;  // LSTM=4, GRU=3, RNN=1
    int lin_layers = gates * 2;  // gates * (ih + hh)

    for (int layer = 0; layer < num_layers; layer++) {
        for (int dir = 0; dir < num_dir; dir++) {
            int pseudo_layer = layer * num_dir + dir;
            int param_idx = pseudo_layer;  // index into weights_ih/hh arrays

            for (int lin = 0; lin < lin_layers; lin++) {
                cudnnTensorDescriptor_t mDesc, bDesc;
                cudnnCreateTensorDescriptor(&mDesc);
                cudnnCreateTensorDescriptor(&bDesc);
                void* mAddr = nullptr;
                void* bAddr = nullptr;

                cudnnGetRNNWeightParams(handle, plan.rnnDesc, pseudo_layer,
                    plan.weight_space_size, weight_space,
                    lin, mDesc, &mAddr, bDesc, &bAddr);

                if (lin < gates) {
                    // Input weights (weight_ih): each gate is [hidden_size, in_size]
                    int in_size = (layer == 0) ? input_size : hidden_size * num_dir;
                    int offset = lin * hidden_size * in_size;
                    if (mAddr)
                        cudaMemcpyAsync(mAddr, weights_ih[param_idx] + offset,
                            hidden_size * in_size * sizeof(float), cudaMemcpyDeviceToDevice, 0);
                    if (bAddr)
                        cudaMemcpyAsync(bAddr, biases_ih[param_idx] + lin * hidden_size,
                            hidden_size * sizeof(float), cudaMemcpyDeviceToDevice, 0);
                } else {
                    // Hidden weights (weight_hh): each gate is [hidden_size, hidden_size]
                    int g = lin - gates;
                    int offset = g * hidden_size * hidden_size;
                    if (mAddr)
                        cudaMemcpyAsync(mAddr, weights_hh[param_idx] + offset,
                            hidden_size * hidden_size * sizeof(float), cudaMemcpyDeviceToDevice, 0);
                    if (bAddr)
                        cudaMemcpyAsync(bAddr, biases_hh[param_idx] + g * hidden_size,
                            hidden_size * sizeof(float), cudaMemcpyDeviceToDevice, 0);
                }
                cudnnDestroyTensorDescriptor(mDesc);
                cudnnDestroyTensorDescriptor(bDesc);
            }
        }
    }
}

// Unpack gradients from cuDNN packed format back to separate tensors
static void unpack_weight_grads(cudnnHandle_t handle, const RNNPlan& plan,
                                float* const* dweights_ih, float* const* dweights_hh,
                                float* const* dbiases_ih, float* const* dbiases_hh,
                                const void* dweight_space, int num_layers, int num_dir,
                                int input_size, int hidden_size, int mode) {
    int gates = (mode == 2) ? 4 : (mode == 3) ? 3 : 1;
    int lin_layers = gates * 2;

    for (int layer = 0; layer < num_layers; layer++) {
        for (int dir = 0; dir < num_dir; dir++) {
            int pseudo_layer = layer * num_dir + dir;
            int param_idx = pseudo_layer;

            for (int lin = 0; lin < lin_layers; lin++) {
                cudnnTensorDescriptor_t mDesc, bDesc;
                cudnnCreateTensorDescriptor(&mDesc);
                cudnnCreateTensorDescriptor(&bDesc);
                void* mAddr = nullptr;
                void* bAddr = nullptr;

                cudnnGetRNNWeightParams(handle, plan.rnnDesc, pseudo_layer,
                    plan.weight_space_size, const_cast<void*>(dweight_space),
                    lin, mDesc, &mAddr, bDesc, &bAddr);

                if (lin < gates) {
                    int in_size = (layer == 0) ? input_size : hidden_size * num_dir;
                    int offset = lin * hidden_size * in_size;
                    if (mAddr)
                        cudaMemcpyAsync(dweights_ih[param_idx] + offset, mAddr,
                            hidden_size * in_size * sizeof(float), cudaMemcpyDeviceToDevice, 0);
                    if (bAddr)
                        cudaMemcpyAsync(dbiases_ih[param_idx] + lin * hidden_size, bAddr,
                            hidden_size * sizeof(float), cudaMemcpyDeviceToDevice, 0);
                } else {
                    int g = lin - gates;
                    int offset = g * hidden_size * hidden_size;
                    if (mAddr)
                        cudaMemcpyAsync(dweights_hh[param_idx] + offset, mAddr,
                            hidden_size * hidden_size * sizeof(float), cudaMemcpyDeviceToDevice, 0);
                    if (bAddr)
                        cudaMemcpyAsync(dbiases_hh[param_idx] + g * hidden_size, bAddr,
                            hidden_size * sizeof(float), cudaMemcpyDeviceToDevice, 0);
                }
                cudnnDestroyTensorDescriptor(mDesc);
                cudnnDestroyTensorDescriptor(bDesc);
            }
        }
    }
}

// ==================== Public API ====================

// Forward: x [B, seq_len, input_size] -> y [B, seq_len, hidden*num_dir], hy [num_layers*num_dir, B, hidden]
void launch_rnn_forward_cudnn(
    const float* x, const float* hx, const float* cx,
    float* y, float* hy, float* cy,
    const float* const* weights_ih, const float* const* weights_hh,
    const float* const* biases_ih, const float* const* biases_hh,
    int mode, int B, int seq_len, int input_size, int hidden_size,
    int num_layers, int bidirectional, cudaStream_t s) {

    cudnnHandle_t handle = get_cudnn_handle();
    cudnnSetStream(handle, s);
    int num_dir = bidirectional ? 2 : 1;

    auto& plan = get_or_create_plan(mode, B, seq_len, input_size, hidden_size,
                                     num_layers, bidirectional, handle);

    // Pack weights into cached weight space
    pack_weights(handle, plan, weights_ih, weights_hh, biases_ih, biases_hh,
                 plan.weight_space, num_layers, num_dir, input_size, hidden_size, mode);

    void* ws = get_workspace(plan.workspace_size);

    cudnnRNNForward(handle, plan.rnnDesc,
        CUDNN_FWD_MODE_TRAINING,
        plan.dev_seq_lengths,
        plan.xDesc, x,
        plan.yDesc, y,
        plan.hDesc, hx, hy,
        plan.cDesc, cx, cy,
        plan.weight_space_size, plan.weight_space,
        plan.workspace_size, ws,
        plan.reserve_size, plan.reserve_space);
}

// Backward: compute dx, dhx, dcx + dweights
void launch_rnn_backward_cudnn(
    const float* x, const float* hx, const float* cx,
    const float* y, const float* dy, const float* dhy, const float* dcy,
    float* dx, float* dhx, float* dcx,
    const float* const* weights_ih, const float* const* weights_hh,
    const float* const* biases_ih, const float* const* biases_hh,
    float* const* dweights_ih, float* const* dweights_hh,
    float* const* dbiases_ih, float* const* dbiases_hh,
    int mode, int B, int seq_len, int input_size, int hidden_size,
    int num_layers, int bidirectional, cudaStream_t s) {

    cudnnHandle_t handle = get_cudnn_handle();
    cudnnSetStream(handle, s);
    int num_dir = bidirectional ? 2 : 1;

    auto& plan = get_or_create_plan(mode, B, seq_len, input_size, hidden_size,
                                     num_layers, bidirectional, handle);

    // Pack weights (same as forward, uses cached buffer)
    pack_weights(handle, plan, weights_ih, weights_hh, biases_ih, biases_hh,
                 plan.weight_space, num_layers, num_dir, input_size, hidden_size, mode);

    void* ws = get_workspace(plan.workspace_size);

    // Backward data
    cudnnRNNBackwardData_v8(handle, plan.rnnDesc,
        plan.dev_seq_lengths,
        plan.yDesc, y, dy,
        plan.xDesc, dx,
        plan.hDesc, hx, dhy, dhx,
        plan.cDesc, cx, dcy, dcx,
        plan.weight_space_size, plan.weight_space,
        plan.workspace_size, ws,
        plan.reserve_size, plan.reserve_space);

    // Backward weights (use cached dweight_space)
    cudaMemsetAsync(plan.dweight_space, 0, plan.weight_space_size, s);

    cudnnRNNBackwardWeights_v8(handle, plan.rnnDesc,
        CUDNN_WGRAD_MODE_ADD,
        plan.dev_seq_lengths,
        plan.xDesc, x,
        plan.hDesc, hx,
        plan.yDesc, y,
        plan.weight_space_size, plan.dweight_space,
        plan.workspace_size, ws,
        plan.reserve_size, plan.reserve_space);

    // Unpack weight gradients
    unpack_weight_grads(handle, plan, dweights_ih, dweights_hh, dbiases_ih, dbiases_hh,
                        plan.dweight_space, num_layers, num_dir, input_size, hidden_size, mode);
}

} // namespace jstorch
