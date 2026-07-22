# JsTorch

A JavaScript version of PyTorch with CUDA acceleration for Node.js.

## Features

- 🚀 CUDA-accelerated tensor operations
- 📦 Native Node.js addon (N-API)
- 🎯 PyTorch-like API
- 🔧 Universal binary supporting multiple GPU architectures

## Requirements

- **OS**: Windows x64 (Linux/Mac support coming soon)
- **Node.js**: >= 18.0.0
- **CUDA**: 13.0 or higher
- **GPU**: NVIDIA GPU with compute capability >= 7.5
  - RTX 20xx/30xx/40xx series
  - GTX 16xx series
  - Tesla T4/A100/H100
  - Quadro RTX series

**Note**: GTX 10xx and older GPUs are not supported (require compute capability < 7.5).

## Installation

```bash
npm install @alex000291/jstorch
```

**Important**: Make sure CUDA runtime DLLs are in your PATH:
- Add `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.x\bin` to your system PATH
- Or ensure `cudart64_XX.dll` is accessible

## Quick Start

```javascript
import { torch } from '@alex000291/jstorch';

// Create matrices (row-major order)
const a = new Float32Array([
    1, 2, 3,
    4, 5, 6
]); // 2x3 matrix

const b = new Float32Array([
    1, 2, 3, 4,
    5, 6, 7, 8,
    9, 10, 11, 12
]); // 3x4 matrix

// Matrix multiplication on GPU: C = A × B
const result = torch.matmul(a, b, 2, 3, 4); // (M, K, N)
console.log(result);
// Float32Array(8) [ 38, 44, 50, 56, 83, 98, 113, 128 ]
```

## API Reference

### `torch.matmul(a, b, M, K, N)`

Performs matrix multiplication on GPU: **C = A × B**

**Parameters:**
- `a` (Float32Array): Matrix A with shape (M, K), stored in row-major order
- `b` (Float32Array): Matrix B with shape (K, N), stored in row-major order
- `M` (number): Number of rows in A
- `K` (number): Number of columns in A / rows in B
- `N` (number): Number of columns in B

**Returns:**
- `Float32Array`: Result matrix C with shape (M, N)

**Example:**
```javascript
// 2×3 matrix
const A = new Float32Array([1, 2, 3, 4, 5, 6]);

// 3×2 matrix
const B = new Float32Array([7, 8, 9, 10, 11, 12]);

// Result: 2×2 matrix
const C = torch.matmul(A, B, 2, 3, 2);
```

## Supported GPU Architectures

The package includes a universal binary that supports:
- **sm_75**: Turing (RTX 20xx, GTX 16xx, Tesla T4)
- **sm_80**: Ampere (A100, A30, A40)
- **sm_86**: Ampere (RTX 30xx, A10, A16)
- **sm_89**: Ada Lovelace (RTX 40xx, L4, L40)
- **sm_90**: Hopper (H100, H800)
- **PTX**: Future architectures (JIT compiled at runtime)

The CUDA runtime automatically selects the optimal code for your GPU.

## Troubleshooting

### `Error: Cannot find module 'jstorch.node'`
Make sure the package was installed correctly and `build/win/jstorch.node` exists.

### `Error: the provided PTX was compiled with an unsupported toolchain`
Your GPU may be too old. JsTorch requires compute capability >= 7.5 (RTX 20xx or newer).

### `Error loading CUDA runtime`
Ensure CUDA is installed and `cudart64_XX.dll` is in your PATH.

## Roadmap

- [x] Matrix multiplication (matmul)
- [ ] Tensor class with automatic shape tracking
- [ ] Element-wise operations (add, mul, div, etc.)
- [ ] Reduction operations (sum, mean, max, etc.)
- [ ] Broadcasting support
- [ ] Autograd / backpropagation
- [ ] Linux / macOS support
- [ ] CPU fallback

## Performance

JsTorch uses CUDA for GPU acceleration. Performance depends on:
- Matrix size (larger matrices benefit more from GPU)
- GPU model (newer GPUs are faster)
- Memory transfer overhead (minimize CPU↔GPU transfers)

## Contributing

Contributions welcome! Please open an issue or PR on [GitHub](https://github.com/Alex000291/JsTorch).

## License

MIT

## Author

Alex000291

## Links

- [GitHub Repository](https://github.com/Alex000291/JsTorch)
- [Issue Tracker](https://github.com/Alex000291/JsTorch/issues)
- [npm Package](https://www.npmjs.com/package/@alex000291/jstorch)
