/**
 * Tensor wrapper with autograd support
 */

import { createRequire } from 'module';
import { is_grad_enabled } from './autograd.js';

const require = createRequire(import.meta.url);
const { Tensor: NativeTensor, zeros: native_zeros, ones: native_ones } = require('../build/win/jstorch.node');

// Wrap native Tensor
export class Tensor {
  constructor(data, requires_grad = false) {
    this._native = new NativeTensor(data);
    this.requires_grad = requires_grad && is_grad_enabled();
    this.grad = null;
    
    // Computational graph
    this._grad_fn = null;   // Backward function
    this._prev = [];        // Parent nodes (inputs)
    this._op = '';          // Operation name (for debugging)
  }
  
  // Properties
  get shape() {
    return this._native.shape;
  }
  
  get size() {
    return this._native.size;
  }
  
  // Operations with autograd support
  add(other) {
    const result = new Tensor(null);
    result._native = this._native.add(other._native);
    
    if (is_grad_enabled() && (this.requires_grad || other.requires_grad)) {
      result.requires_grad = true;
      result._prev = [this, other];
      result._op = 'add';
      
      // Gradient function: d(a+b)/da = 1, d(a+b)/db = 1
      result._grad_fn = () => {
        if (this.requires_grad) {
          this.grad = this.grad ? this.grad.add(result.grad) : result.grad;
        }
        if (other.requires_grad) {
          other.grad = other.grad ? other.grad.add(result.grad) : result.grad;
        }
      };
    }
    return result;
  }
  
  add_(other) {
    const result = new Tensor(null);
    result._native = this._native.add(other._native);
    return result;
  }
  
  sub(other) {
    const result = new Tensor(null);
    result._native = this._native.sub(other._native);
    
    if (is_grad_enabled() && (this.requires_grad || other.requires_grad)) {
      result.requires_grad = true;
      result._prev = [this, other];
      result._op = 'sub';
      
      // d(a-b)/da = 1, d(a-b)/db = -1
      result._grad_fn = () => {
        if (this.requires_grad) {
          this.grad = this.grad ? this.grad.add(result.grad) : result.grad;
        }
        if (other.requires_grad) {
          const neg_grad = result.grad.neg();
          other.grad = other.grad ? other.grad.add(neg_grad) : neg_grad;
        }
      };
    }
    return result;
  }
  
  mul(other) {
    const result = new Tensor(null);
    result._native = this._native.mul(other._native);
    
    if (is_grad_enabled() && (this.requires_grad || other.requires_grad)) {
      result.requires_grad = true;
      result._prev = [this, other];
      result._op = 'mul';
      
      // d(a*b)/da = b, d(a*b)/db = a
      result._grad_fn = () => {
        if (this.requires_grad) {
          const grad_a = result.grad.mul(other);
          this.grad = this.grad ? this.grad.add(grad_a) : grad_a;
        }
        if (other.requires_grad) {
          const grad_b = result.grad.mul(this);
          other.grad = other.grad ? other.grad.add(grad_b) : grad_b;
        }
      };
    }
    return result;
  }
  
  div(other) {
    const result = new Tensor(null);
    result._native = this._native.div(other._native);
    if (is_grad_enabled() && (this.requires_grad || other.requires_grad)) {
      result.requires_grad = true;
    }
    return result;
  }
  
  mulScalar(scalar) {
    const result = new Tensor(null);
    result._native = this._native.mulScalar(scalar);
    
    if (is_grad_enabled() && this.requires_grad) {
      result.requires_grad = true;
      result._prev = [this];
      result._op = 'mulScalar';
      
      // d(a*c)/da = c (where c is scalar)
      result._grad_fn = () => {
        if (this.requires_grad) {
          const grad_a = result.grad.mulScalar(scalar);
          this.grad = this.grad ? this.grad.add(grad_a) : grad_a;
        }
      };
    }
    return result;
  }
  
  exp() {
    const result = new Tensor(null);
    result._native = this._native.exp();
    if (is_grad_enabled() && this.requires_grad) {
      result.requires_grad = true;
    }
    return result;
  }
  
  log() {
    const result = new Tensor(null);
    result._native = this._native.log();
    if (is_grad_enabled() && this.requires_grad) {
      result.requires_grad = true;
    }
    return result;
  }
  
  neg() {
    const result = new Tensor(null);
    result._native = this._native.neg();
    if (is_grad_enabled() && this.requires_grad) {
      result.requires_grad = true;
    }
    return result;
  }
  
  sum() {
    const result = new Tensor(null);
    result._native = this._native.sum();
    
    if (is_grad_enabled() && this.requires_grad) {
      result.requires_grad = true;
      result._prev = [this];
      result._op = 'sum';
      
      // d(sum(a))/da = ones_like(a) * grad_output
      result._grad_fn = () => {
        if (this.requires_grad) {
          // Gradient broadcasts to all elements
          const grad_value = result.grad.item();
          const grad_a = ones(this.shape).mulScalar(grad_value);
          this.grad = this.grad ? this.grad.add(grad_a) : grad_a;
        }
      };
    }
    return result;
  }
  
  mean() {
    const result = new Tensor(null);
    result._native = this._native.mean();
    
    if (is_grad_enabled() && this.requires_grad) {
      result.requires_grad = true;
      result._prev = [this];
      result._op = 'mean';
      
      // d(mean(a))/da = ones_like(a) / size * grad_output
      result._grad_fn = () => {
        if (this.requires_grad) {
          const grad_value = result.grad.item();
          const grad_a = ones(this.shape).mulScalar(grad_value / this.size);
          this.grad = this.grad ? this.grad.add(grad_a) : grad_a;
        }
      };
    }
    return result;
  }
  
  matmul(other) {
    const result = new Tensor(null);
    result._native = this._native.matmul(other._native);
    
    if (is_grad_enabled() && (this.requires_grad || other.requires_grad)) {
      result.requires_grad = true;
      result._prev = [this, other];
      result._op = 'matmul';
      
      // d(A@B)/dA = dC @ B^T, d(A@B)/dB = A^T @ dC
      result._grad_fn = () => {
        if (this.requires_grad) {
          const grad_a = result.grad.matmul(other.transpose());
          this.grad = this.grad ? this.grad.add(grad_a) : grad_a;
        }
        if (other.requires_grad) {
          const grad_b = this.transpose().matmul(result.grad);
          other.grad = other.grad ? other.grad.add(grad_b) : grad_b;
        }
      };
    }
    
    return result;
  }
  
  matmul_(other) {
    const result = new Tensor(null);
    result._native = this._native.matmul(other._native);
    return result;
  }
  
  relu() {
    const result = new Tensor(null);
    result._native = this._native.relu();
    
    if (is_grad_enabled() && this.requires_grad) {
      result.requires_grad = true;
      result._prev = [this];
      result._op = 'relu';
      
      // Save input for backward
      const input_data = this.toArray();
      
      // d(relu(a))/da = (a > 0) ? 1 : 0
      result._grad_fn = () => {
        if (this.requires_grad) {
          // Create mask: 1 where input > 0, 0 otherwise
          const mask = this._createMask(input_data, x => x > 0 ? 1 : 0);
          const grad_a = result.grad.mul(mask);
          this.grad = this.grad ? this.grad.add(grad_a) : grad_a;
        }
      };
    }
    
    return result;
  }
  
  // Helper: create mask tensor from nested array
  _createMask(data, fn) {
    const applyMask = (arr) => {
      if (Array.isArray(arr)) {
        return arr.map(applyMask);
      }
      return fn(arr);
    };
    return tensor(applyMask(data));
  }
  
  relu_() {
    const result = new Tensor(null);
    result._native = this._native.relu();
    return result;
  }
  
  // Reshape (view with new shape, shares underlying data)
  reshape(new_shape) {
    // Calculate total size
    const old_size = this.shape.reduce((a, b) => a * b, 1);
    let new_size = 1;
    let infer_dim = -1;
    
    // Handle -1 (infer dimension)
    for (let i = 0; i < new_shape.length; i++) {
      if (new_shape[i] === -1) {
        if (infer_dim !== -1) {
          throw new Error('Only one dimension can be inferred');
        }
        infer_dim = i;
      } else {
        new_size *= new_shape[i];
      }
    }
    
    // Infer missing dimension
    const final_shape = [...new_shape];
    if (infer_dim !== -1) {
      final_shape[infer_dim] = old_size / new_size;
      new_size = old_size;
    }
    
    // Check size matches
    if (new_size !== old_size) {
      throw new Error(`Cannot reshape tensor of size ${old_size} to shape [${final_shape}] (size ${new_size})`);
    }
    
    // Get flattened data
    const data = this._native.toArray();
    
    // Rebuild nested array with new shape
    const buildNested = (flat, shape, offset = 0) => {
      if (shape.length === 1) {
        return flat.slice(offset, offset + shape[0]);
      }
      
      const result = [];
      const stride = shape.slice(1).reduce((a, b) => a * b, 1);
      for (let i = 0; i < shape[0]; i++) {
        result.push(buildNested(flat, shape.slice(1), offset + i * stride));
      }
      return result;
    };
    
    // Flatten data first
    const flatten = (arr) => {
      if (!Array.isArray(arr)) return [arr];
      return arr.reduce((acc, val) => acc.concat(flatten(val)), []);
    };
    
    const flat_data = flatten(data);
    const reshaped_data = buildNested(flat_data, final_shape);
    
    // Create new tensor with reshaped data
    const result = tensor(reshaped_data, this.requires_grad);
    
    // Set up autograd
    if (this.requires_grad) {
      result._prev = [this];
      result._backward = () => {
        // Gradient flows back with same reshape
        if (result.grad) {
          const grad_reshaped = result.grad.reshape(this.shape);
          this.grad = this.grad ? this.grad.add(grad_reshaped) : grad_reshaped;
        }
      };
    }
    
    return result;
  }
  
  // View (alias for reshape)
  view(new_shape) {
    return this.reshape(new_shape);
  }
  
  // Flatten to 1D or 2D
  flatten(start_dim = 0, end_dim = -1) {
    if (end_dim === -1) end_dim = this.shape.length - 1;
    
    // Calculate new shape
    const new_shape = [];
    let flatten_size = 1;
    
    for (let i = 0; i < this.shape.length; i++) {
      if (i < start_dim || i > end_dim) {
        new_shape.push(this.shape[i]);
      } else {
        flatten_size *= this.shape[i];
      }
    }
    
    // Insert flattened dimension
    new_shape.splice(start_dim, 0, flatten_size);
    
    return this.reshape(new_shape);
  }
  
  // Conv2D with autograd
  // input: this [batch, in_channels, height, width]
  // weight: [out_channels, in_channels, kernel_h, kernel_w]
  // bias: [out_channels] or null
  // Returns: [batch, out_channels, out_h, out_w]
  conv2d(weight, bias, stride_h, stride_w, padding_h, padding_w) {
    const result = new Tensor(null);
    result._native = this._native.conv2d(
      weight._native,
      bias ? bias._native : null,
      stride_h, stride_w, padding_h, padding_w
    );
    
    if (is_grad_enabled() && (this.requires_grad || weight.requires_grad || (bias && bias.requires_grad))) {
      result.requires_grad = true;
      result._prev = bias ? [this, weight, bias] : [this, weight];
      result._op = 'conv2d';
      
      // Save metadata for backward
      const input_shape = this.shape;
      const weight_shape = weight.shape;
      const has_bias = bias !== null;
      
      result._grad_fn = () => {
        // Backward for input
        if (this.requires_grad) {
          const grad_input_native = result.grad._native.conv2dBackwardInput(
            weight._native,
            input_shape,
            stride_h, stride_w, padding_h, padding_w
          );
          const grad_input = new Tensor(null);
          grad_input._native = grad_input_native;
          this.grad = this.grad ? this.grad.add(grad_input) : grad_input;
        }
        
        // Backward for weight
        if (weight.requires_grad) {
          const grad_weight_native = result.grad._native.conv2dBackwardWeight(
            this._native,
            weight_shape,
            stride_h, stride_w, padding_h, padding_w
          );
          const grad_weight = new Tensor(null);
          grad_weight._native = grad_weight_native;
          weight.grad = weight.grad ? weight.grad.add(grad_weight) : grad_weight;
        }
        
        // Backward for bias
        if (has_bias && bias.requires_grad) {
          const grad_bias_native = result.grad._native.conv2dBackwardBias();
          const grad_bias = new Tensor(null);
          grad_bias._native = grad_bias_native;
          bias.grad = bias.grad ? bias.grad.add(grad_bias) : grad_bias;
        }
      };
    }
    
    return result;
  }
  
  // Utilities
  toArray() {
    return this._native.toArray();
  }
  
  item() {
    return this._native.item();
  }
  
  // Autograd
  backward(gradient = null) {
    if (!this.requires_grad) {
      throw new Error('Called backward() on tensor that does not require grad');
    }
    
    // Initialize gradient
    this.grad = gradient || ones(this.shape);
    
    // Build topological order
    const topo = [];
    const visited = new Set();
    
    const build_topo = (node) => {
      if (!node.requires_grad || visited.has(node)) return;
      visited.add(node);
      
      for (const parent of node._prev) {
        build_topo(parent);
      }
      
      topo.push(node);
    };
    
    build_topo(this);
    
    // Execute gradients in reverse topological order
    for (let i = topo.length - 1; i >= 0; i--) {
      const node = topo[i];
      if (node._grad_fn) {
        node._grad_fn();
      }
    }
  }
  
  zero_grad() {
    this.grad = null;
  }
  
  // Transpose (2D only for now)
  transpose() {
    if (this.shape.length !== 2) {
      throw new Error('transpose() only supports 2D tensors for now');
    }
    
    const data = this.toArray();
    const [rows, cols] = this.shape;
    
    // Transpose data
    const transposed = Array.from({ length: cols }, (_, i) =>
      Array.from({ length: rows }, (_, j) => data[j][i])
    );
    
    const result = tensor(transposed, this.requires_grad && is_grad_enabled());
    
    if (is_grad_enabled() && this.requires_grad) {
      result._prev = [this];
      result._op = 'transpose';
      
      // d(A^T)/dA = (dC)^T
      result._grad_fn = () => {
        if (this.requires_grad) {
          const grad_a = result.grad.transpose();
          this.grad = this.grad ? this.grad.add(grad_a) : grad_a;
        }
      };
    }
    
    return result;
  }
}

// Factory functions
export function tensor(data, requires_grad = false) {
  return new Tensor(data, requires_grad);
}

export function zeros(shape) {
  const result = new Tensor(null);
  result._native = native_zeros(shape);
  return result;
}

export function ones(shape) {
  const result = new Tensor(null);
  result._native = native_ones(shape);
  return result;
}

// Export native directly for advanced use
export { NativeTensor };
