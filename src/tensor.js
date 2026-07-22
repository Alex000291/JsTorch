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
