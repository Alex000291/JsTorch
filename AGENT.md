# JsTorch Codebase Map

PyTorch-like tensor lib for Node.js. C++/CUDA native addon via N-API. Float32 only.

## Architecture

```
src/ (JS)           → user-facing API, autograd, nn modules, optimizers
native/binding/     → N-API bridge (TensorWrap, GradTensorWrap)
native/core/        → C++ Tensor class, allocator, autograd engine
native/ops/         → CUDA kernels (unary, binary, reduce, matmul, conv, misc)
native/audio/       → STFT (cuFFT)
build/win/          → output .node + bundled DLLs
```

Data flow: `JS GradTensor → N-API → C++ Tensor → CUDA kernel`
Memory: `CudaAllocator` (sub-allocation, free-list, coalescing, 256MB segments)
Autograd: **dual implementation** — JS (src/autograd.js) and C++ (native/core/autograd.hpp)

---

## native/core/

### `dtype.hpp`
```
enum DType { Float32, Complex64 }
dtype_size(DType) → size_t
```

### `shape.hpp`
```
using Shape = vector<int>;  using Strides = vector<int>;
compute_strides(Shape) → Strides         // row-major strides
total_size(Shape) → int
is_contiguous(Shape, Strides) → bool
broadcast_shape(s1, s2) → Shape          // NumPy broadcast rules
normalize_axis(axis, ndim) → int         // handle negative axis
```

### `allocator.hpp`
```
class CudaAllocator
  ALIGNMENT=512, SEGMENT_SIZE=256MB, MIN_SPLIT=512
  struct Block { uintptr_t addr; size_t size; }
  set<Block> free_blocks_                 // sorted by addr for coalescing
  map<uintptr_t,size_t> alloc_map_        // active allocations
  vector<void*> segments_                 // all cudaMalloc'd segments
  allocate(bytes) → void*                 // best-fit, split remainder
  free(ptr, bytes)                        // coalesce with left/right neighbors
  add_segment(min_size)                   // grow pool
get_allocator() → CudaAllocator&          // singleton
```

### `tensor.hpp` — Tensor class
```
class Tensor
  data_: shared_ptr<void>                 // GPU memory (ref-counted, freed via allocator)
  dtype_: DType
  shape_: Shape, strides_: Strides
  stream_: cudaStream_t

  // Ctor
  Tensor(shape, dtype=Float32)            // allocates GPU memory
  Tensor(shared_ptr, shape, strides, dtype, stream)  // view ctor (shares data)

  // View ops (no data copy, share data_)
  view/reshape/squeeze/unsqueeze/transpose/slice/split/flatten

  // Unary ops (allocate new tensor, launch kernel)
  abs/sqrt/square/exp/log/log1p/sin/cos/neg/floor/ceil/round
  sigmoid/tanh/relu/silu/gelu/softplus/reciprocal/sign
  leaky_relu(slope)/clamp(lo,hi)/clamp_min/clamp_max/fmod(d)
  pow_scalar(e)/mul_scalar(s)/add_scalar(s)

  // Binary ops (broadcast)
  add/sub/mul/div/maximum/minimum/pow
  gt/lt/ge/le/eq/ne

  // Reduce
  sum(dim,keepdim)/mean(dim,keepdim)/max(dim,keepdim)/min(dim,keepdim)
  argmax(dim)/argmin(dim)

  // Matmul (cuBLAS)
  matmul(other)                           // 2D, 3D(bmm), 2Dx1D

  // Conv (cuDNN for conv2d fwd, im2col+cuBLAS for rest)
  conv1d/conv_transpose1d/conv2d/conv_transpose2d/avg_pool2d
  embedding(indices)
  interpolate(target_size, mode, align_corners)

  // Backward helper ops
  scatter_add/interp1d_backward/avgpool2d_backward
  conv1d_backward_weight/conv2d_backward_weight (static)
  conv_transpose1d_backward_weight/conv_transpose2d_backward_weight (static)

  // Factory (static)
  from_array/from_buffer/randn/from_int_array/full/arange
  randn_like()
  to_array() → vector<float>             // GPU→CPU, sync

  // Optimizer
  static adam_step(param,grad,m,v, lr,b1,b2,eps,bc1,bc2,wd)  // fused CUDA kernel
```

### `tensor.cpp` — implementations
- Ctor: `get_allocator().allocate(bytes)`, data_ shared_ptr with custom deleter
- Unary: `#define UNARY(name,launcher)` macro → contiguous() + alloc + launch
- Binary: `#define BINARY(name,launcher)` macro → broadcast shape + contiguous + alloc + launch
- Reduce: flatten to (outer, reduce_dim, inner), launch reduce kernel
- Matmul: `cublasSgemm` (col-major trick: C^T = B^T * A^T → row-major C = A * B)
- Conv2d fwd: `cudnnConvolutionForward` with plan cache (`map<ConvParams, ConvPlan>`)
- Conv bwd: im2col + cuBLAS gemm

### `autograd.hpp` — C++ autograd engine
```
using GradPtr = shared_ptr<GradTensor>;

class GradTensor : enable_shared_from_this
  Tensor data
  bool requires_grad
  shared_ptr<Tensor> grad_               // accumulated gradient
  bool has_grad
  using GradFn = function<vector<Tensor>(const Tensor&)>
  GradFn grad_fn                          // backward function (closure)
  vector<GradPtr> parents

  // Engine
  static backward(root, upstream)         // topo sort + reverse traverse, all in C++

  // Factory
  static make(Tensor, bool rg) → GradPtr
  static make_with_grad(Tensor, GradFn, parents) → GradPtr

  // Ops (forward compute + backward recording)
  // Unary: exp_/neg_/sigmoid_/tanh_/relu_/log_/sqrt_/square_/silu_/gelu_
  //        abs_/sin_/cos_/reciprocal_/softplus_/sign_/log1p_
  //        pow_scalar_/mul_scalar_/add_scalar_
  // Binary: add_/sub_/mul_/div_/pow_ (with unbroadcast)
  //         gt_/lt_/ge_/le_/eq_/ne_ (no grad)
  // Matmul: matmul_
  // Reduce: sum_/mean_
  // View: reshape_/transpose_/transpose_(d0,d1)
  // Conv: conv2d_/avg_pool2d_

  static unbroadcast(grad, target_shape)  // reverse broadcast for binary grad
  static topo_sort(node, order, visited)  // DFS topo sort

// ---- CompiledGraph: forward+backward+Adam in 1 N-API call ----
enum OpType : uint8_t
  RELU EXP LOG SQRT SQUARE NEG ABS SIGMOID TANH SILU GELU SOFTPLUS RECIPROCAL SIGN LOG1P SIN COS
  POW_SCALAR MUL_SCALAR ADD_SCALAR
  ADD SUB MUL DIV POW MATMUL SUM MEAN TRANSPOSE TRANSPOSE2

struct TracedOp { OpType type; int16_t out,in0,in1; float fparam; int iparam0,iparam1; }

class CompiledGraph
  vector<TracedOp> ops_
  int input_slot_, target_slot_, output_slot_
  vector<int> param_slots_

  input()/target()/param() → int               // allocate named slots
  set_output(slot)
  op1(type, in, fp, ip0, ip1) → int            // unary/reduce/view op
  op2(type, a, b) → int                        // binary/matmul op

  run(input, target, params[], m[], v[],        // execute forward+backward+Adam
      lr, beta1, beta2, eps, bc1, bc2, wd)      // ALL in C++, 0 intermediate JS objects
  // Internally: creates GradTensors in slots, switch-dispatches ops,
  //             calls GradTensor::backward(), then adam_step per param
```

---

## native/ops/

### `kernels.cuh` — unary kernel template + op functors
```
// Vectorized: float4 load/store, 128 threads, grid-stride loop
unary_kernel_vec4<Op>(float4* in, float4* out, int n4, Op)
unary_kernel_tail<Op>(float* in, float* out, int offset, int size, Op)
launch_unary<Op>(float* in, float* out, int size, Op, stream)
  → n4 = size/4, launch vec4 kernel + tail kernel

// Op functors (19 simple + 8 parameterized):
AbsOp SqrtOp SquareOp ExpOp LogOp SinOp CosOp SigmoidOp TanhOp
ReluOp NegOp FloorOp CeilOp RoundOp SiluOp GeluOp SoftplusOp
Log1pOp ReciprocalOp SignOp
LeakyReluOp{slope} ClampOp{lo,hi} FmodOp{d} ClampMinOp{lo}
ClampMaxOp{hi} PowScalarOp{e} MulScalarOp{s} AddScalarOp{s}
// PowScalarOp: handles negative base (copysign trick)
```

### `unary.cu` — extern "C" launchers
```
// 1:1 mapping: launch_<name>() → launch_unary(..., <Op>(), stream)
launch_abs/sqrt/square/exp/log/sin/cos/sigmoid/tanh/relu/neg/floor/ceil/round
launch_silu/gelu/softplus/log1p/reciprocal/sign
launch_leaky_relu(..slope)/launch_clamp(..lo,hi)/launch_fmod(..d)
launch_clamp_min/clamp_max/pow_scalar/mul_scalar/add_scalar
```

### `binary.cu` — broadcast binary ops
```
struct BroadcastArgs { a/b shape+strides, out_shape, ndim (MAX_DIMS=8) }

// Fast path: same shape → vectorized float4
binary_vec4<Op>(float4* a, float4* b, float4* out, int n4, Op)
binary_tail<Op>(..., int offset, int size, Op)

// Slow path: broadcast index calculation
broadcast_kernel<Op>(..., BroadcastArgs, total_size, Op)

launch_broadcast<Op>() → detects same_shape → fast or slow path

// Op functors: AddOp SubOp MulOp DivOp MaxOp MinOp PowOp
//              GtOp LtOp GeOp LeOp EqOp NeOp
// extern "C" via BROADCAST_EXTERN macro: launch_broadcast_<name>()
```

### `reduce.cu`
```
reduce_sum_kernel(in, out, outer, reduce_size, inner)   // 1 thread per output elem, sequential reduce
reduce_mean_kernel(...)                                  // same + /reduce_size
reduce_max_kernel(in, out, *indices, ...)                // tracks argmax
reduce_min_kernel(...)
// Launchers: launch_reduce_sum/mean/max/min
```

### `matmul.cu`
```
get_handle() → cublasHandle_t (singleton)
launch_matmul(a,b,c, M,K,N, stream)      // cublasSgemm, col-major trick
launch_bmm(a,b,c, B,M,K,N, stream)       // cublasSgemmStridedBatched
```

### `misc.cu` — everything else
```
// Kernels:
flip_kernel          → reverse along one dim
pad_kernel           → constant (mode=0) or reflect (mode=1) padding
cumsum_kernel        → prefix sum along dim (sequential per thread)
where_kernel         → cond > 0 ? x : y
randn_kernel         → curand_normal per element
embedding_kernel     → gather rows by index
interp1d_nearest/linear_kernel → 1D interpolation
cat_kernel           → concat along dim
scatter_add_kernel   → atomicAdd for embedding backward
interp1d_backward_nearest/linear_kernel → interpolation backward
avgpool2d_backward_kernel → distribute grad / count
strided_copy_kernel  → non-contiguous → contiguous copy
adam_step_kernel      → fused Adam update (m,v,param in one kernel)

// Launchers: launch_flip/pad/cumsum/where/randn/embedding/interp1d/cat
//            launch_scatter_add/interp1d_backward/avgpool2d_backward
//            launch_strided_copy/launch_adam_step
```

### `conv.cu` — convolution ops
```
get_cudnn_handle() → cudnnHandle_t (singleton)
struct ConvParams { B,Ci,H,W,Co,kH,kW,sH,sW,pH,pW,dH,dW,groups }
struct ConvPlan { algo, workspace }

// 1D conv: im2col + cuBLAS gemm
im2col_1d_kernel / col2im_1d_kernel / add_bias_1d_kernel
launch_conv1d / launch_conv_transpose1d

// 2D conv
im2col_2d_kernel / col2im_2d_kernel / add_bias_2d_kernel / avgpool2d_kernel
launch_conv2d            → im2col + cublasSgemm (grouped)
launch_conv_transpose2d  → cublasSgemm + col2im
launch_avgpool2d
launch_conv2d_cudnn      → cudnnConvolutionForward (cached plan, TENSOR_OP_MATH)

// Backward weight (im2col of input/grad_output + gemm)
launch_conv1d_backward_weight / launch_conv2d_backward_weight
launch_conv_transpose1d_backward_weight / launch_conv_transpose2d_backward_weight
```

---

## native/audio/

### `stft.cu`
```
hann_window_kernel / stft_windowing_kernel / istft_overlap_add_kernel
stft_create_window(win_length) → void* (device Hann window)
stft_destroy_window(void*)
stft_forward(input, output, window, batch, len, n_fft, hop, win_len, stream)  // cuFFT R2C
stft_inverse(input, output, window, batch, len, n_fft, hop, win_len, stream)  // cuFFT C2R + overlap-add
```

---

## native/binding/

### `napi.cpp` — N-API bridge
```
struct CtorRefs { tensor_ctor, grad_ctor }  // stored via env.SetInstanceData

// Helpers
buildNestedArray(env, data, shape) → Napi::Value   // flat GPU data → nested JS array
parseArray(val) → vector<float>                     // nested JS array → flat
inferShape(val) → Shape
parseShape(info, idx) → Shape

class TensorWrap : ObjectWrap<TensorWrap>
  Tensor tensor_
  static Wrap(env, Tensor) → Napi::Object
  // Exposes ALL Tensor methods 1:1
  // Unary: DEF_UNARY macro
  // Binary: DEF_BINARY macro + optimized scalar (mul/add/sub/div/pow)
  // Static: fromBuffer/randn/full/arange/fromIntArray/cat/where/adamStep

class GradTensorWrap : ObjectWrap<GradTensorWrap>
  GradPtr gt_                                        // shared_ptr<GradTensor>
  static Wrap(env, GradPtr) → Napi::Object
  static Extract(Napi::Value) → GradPtr

  // Accessors: shape, ndim, requires_grad (r/w), toArray, getData, detach
  // Backward: backward(upstream?), getGrad, setGrad, clearGrad
  // View: reshape/transpose/squeeze/unsqueeze/clone/contiguous/flatten/slice
  // Unary: GRAD_UNARY macro → delegates to autograd.hpp methods
  // Binary: Add/Sub/Mul/Div/Pow (scalar→*_scalar_, tensor→*_())
  //         Gt/Lt/Ge/Le/Eq/Ne
  // Matmul/Reduce/Conv2d/AvgPool2d
  // Static: randn/zeros/ones/full/fromBuffer/fromTensor
  // Adam: adamStep (single), adamStepMulti (batch all params, 1 N-API call)

class CompiledGraphWrap : ObjectWrap<CompiledGraphWrap>
  CompiledGraph graph_
  // Slot builders: input/target/param/setOutput
  // Unary: relu/exp/log/sqrt/square/neg/abs/sigmoid/tanh/silu/gelu/softplus/sin/cos
  // Parameterized: pow_scalar/mul_scalar/add_scalar
  // Binary: add/sub/mul/div/pow/matmul
  // Reduce: sum(slot,dim,keepdim)/mean(slot,dim,keepdim)
  // View: transpose(slot[,d0,d1])
  // Run: run(input,target,params[],m[],v[], lr,b1,b2,eps,bc1,bc2,wd)
```

---

## src/ (JS layer)

### `native.js`
Loads `../build/win/jstorch.node` with DLL path setup. Exports raw native module.

### `tensor.js`
```
class Tensor extends native.Tensor
  // Static factories: zeros/ones/randn/rand/fromBuffer/fromFlat/cat/where/
  //                   fromIntArray/full/arange/linspace/stack
  // Instance: size(dim)/dim()/numel()/split/chunk/masked_fill/permute/view/expand/item
  // Stubs: to()/half()/float() → no-op (float32 only)

// Standalone: tril(input,diag) / triu(input,diag) — via arange+mask+where
```

### `autograd.js` — JS autograd (LEGACY, being replaced by C++ autograd.hpp)
```
// Grad context
no_grad(fn) / enable_grad(fn) / is_grad_enabled() / set_grad_enabled(bool)

class GradTensor
  data: native.Tensor
  requires_grad: bool
  grad: native.Tensor | null
  grad_fn: (grad) => [grad_for_parent, ...]
  _parents: GradTensor[]

  backward(upstream?)    // JS topo sort + reverse traverse (N-API per op!)

  // View ops: reshape/view/squeeze/unsqueeze/transpose/contiguous/flatten/slice/expand/permute/split/chunk
  // Static: zeros/ones/randn/rand/full/arange/linspace/fromBuffer/fromFlat/fromIntArray
  //         cat/stack/where/zeros_like/ones_like

// Backward registry B = {}
  // Unary: neg/abs/exp/log/log1p/sqrt/square/sin/cos/sigmoid/tanh/relu/silu/gelu/softplus/reciprocal/sign
  //        leaky_relu/clamp/clamp_min/clamp_max/fmod/pow_scalar/mul_scalar/add_scalar
  // Binary: add/sub/mul/div/pow/maximum/minimum/gt/lt/ge/le/eq/ne
  // Matmul: da = g@b^T, db = a^T@g
  // Reduce: sum/mean (expand grad back), max/min (one-hot mask)
  // Conv: conv1d/conv_transpose1d/conv2d/conv_transpose2d/avg_pool2d/interpolate
  // Misc: pad/cumsum/flip/embedding

// Methods generated via metaprogramming loops:
  SIMPLE_UNARY (17 ops) — prototype[op] = fn
  BINARY_OPS (13 ops) — handles scalar + tensor, unbroadcast
  Reduce ops: sum/mean/max/min/argmax/argmin
  Conv ops: conv1d/conv_transpose1d/conv2d/conv_transpose2d/avg_pool2d/interpolate
  Misc: flip/pad/cumsum/embedding
```

### `nn.js` — neural network modules
```
class Module                              // base: _parameters, _buffers, _modules, parameters(), _init_grad()
                                          //        train/eval, load_state_dict, named_modules
class Linear(in, out, bias?)              // x @ w^T + b
class Embedding(num, dim)                 // weight.embedding(indices)
class Conv1d(Ci, Co, K, {stride,pad,dil,groups,bias})
class ConvTranspose1d(Ci, Co, K, {stride,pad,opad,dil,groups,bias})
class Conv2d(Ci, Co, K, {stride,pad,dil,groups,bias})
class ConvTranspose2d(Ci, Co, K, {stride,pad,opad,dil,groups,bias})
class LayerNorm(shape, eps)               // (x-mean)/sqrt(var+eps) * w + b
class GroupNorm(groups, channels, eps)
class BatchNorm1d(features, eps)          // inference only (running stats)
class BatchNorm2d extends BatchNorm1d
class AvgPool2d(kernel, stride, padding)
class Upsample({scale_factor,size,mode})
class GRU(input, hidden, {layers,bidir,batch_first})  // full GRU with cell impl
class ReLU/LeakyReLU/Sigmoid/Tanh/GELU/SiLU/Dropout
class Sequential(...layers)
class ModuleList(modules[])
weight_norm/remove_weight_norm            // no-op for inference

F = { leaky_relu, relu, sigmoid, tanh, gelu, silu, softmax, interpolate, pad }
```

### `optim.js` — optimizers
```
class Optimizer                           // base: parameters[], state Map, zero_grad()
class SGD({ lr, momentum, weight_decay }) // manual m,v in JS
class Adam({ lr, betas, eps, wd })        // uses native.Tensor.adamStep (fused CUDA kernel)
class AdamW({ lr, betas, eps, wd })       // decoupled weight decay, manual in JS
```

### `loader.js` — model serialization
```
loadModel(path) → { stateDict, config }   // .bin format: magic + header JSON + raw float32 data
saveModel(path, stateDict, config)
```

### `index.js` — entry point
```
export { Tensor, zeros, ones, randn, ... } from './tensor.js'
export { GradTensor, no_grad, enable_grad, ... } from './autograd.js'
export { nn } from './nn.js'
export { optim } from './optim.js'
export { loadModel, saveModel } from './loader.js'
```

---

## Build

### `Makefile.js`
```
CUDA_PATH = C:\...\CUDA\v13.4
MSVC_PATH = C:\...\VS 2022 Community
// 1. Load vcvars64.bat environment
// 2. nvcc: -std=c++17 -O3 --use_fast_math -Xcompiler /MD, sm_89 + sm_90
// 3. cl: /std:c++17 /O2 /MD
// 4. link: /DLL → build/win/jstorch.node
//    Links: node.lib cudart.lib cublas.lib cufft.lib curand.lib cudnn.lib
```

---

## Key patterns

- **Tensor memory**: shared_ptr with custom deleter → allocator.free()
- **View ops**: share data_ (aliasing via shared_ptr), change shape/strides only
- **contiguous()**: if already contiguous, returns `*this` (no copy)
- **UNARY/BINARY macros**: `tensor.cpp` reduces boilerplate for 30+ ops
- **Conv2d forward**: cuDNN with algorithm cache; backward: im2col + cuBLAS
- **powf(negative)**: explicit sign handling to avoid NaN
- **Two autograd systems**: JS (src/autograd.js) — full featured but slow (N-API per op in backward); C++ (native/core/autograd.hpp) — fast (backward stays in C++, 1 N-API call)
- **CompiledGraph**: declarative graph builder → forward+backward+Adam in single N-API call. Eliminates ALL JS↔C++ overhead for training. MLP train B=32: 403us (beats PyTorch 671us)
- **randn is slow** (~3ms for [512,784]): curand_normal per-element kernel. fromBuffer is 116us. Benchmark should avoid randn in hot loop.
