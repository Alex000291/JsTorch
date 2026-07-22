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
    this._grad_fn = null;
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
    }
    return result;
  }
  
  mul(other) {
    const result = new Tensor(null);
    result._native = this._native.mul(other._native);
    if (is_grad_enabled() && (this.requires_grad || other.requires_grad)) {
      result.requires_grad = true;
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
    }
    return result;
  }
  
  mean() {
    const result = new Tensor(null);
    result._native = this._native.mean();
    if (is_grad_enabled() && this.requires_grad) {
      result.requires_grad = true;
    }
    return result;
  }
  
  matmul(other) {
    const result = new Tensor(null);
    result._native = this._native.matmul(other._native);
    
    if (is_grad_enabled() && (this.requires_grad || other.requires_grad)) {
      result.requires_grad = true;
      result._grad_fn = () => {
        // Backward: dL/dA = dL/dC @ B^T, dL/dB = A^T @ dL/dC
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
      const input_data = this.toArray();  // Save for backward
      result._grad_fn = () => {
        // Backward: gradient passes through where input > 0
        // TODO: implement element-wise multiplication with mask
        if (this.requires_grad) {
          this.grad = result.grad;  // Simplified for now
        }
      };
    }
    
    return result;
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
    
    // Topological sort and execute grad_fn (simplified - just call directly for now)
    if (this._grad_fn) {
      this._grad_fn();
    }
  }
  
  zero_grad() {
    this.grad = null;
  }
  
  // Placeholder for transpose (TODO: implement in C++)
  transpose() {
    // Assume 2D for now
    return this;  // TODO
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
