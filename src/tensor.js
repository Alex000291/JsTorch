// tensor.js - Tensor JS wrapper
import { createRequire } from 'module';
const require = createRequire(import.meta.url);
const native = require('../build/jstorch.node');

export class Tensor extends native.Tensor {
    // All native methods inherited: add, sub, mul, div, abs, sqrt, square, exp, log,
    // sin, cos, neg, floor, ceil, round, sigmoid, tanh, relu, silu, gelu, softplus,
    // leaky_relu, clamp, maximum, minimum, pow, gt, lt, ge, le, eq, ne,
    // matmul, sum, mean, reshape, squeeze, unsqueeze, transpose, slice,
    // contiguous, clone, flip, pad, cumsum, embedding,
    // conv1d, conv_transpose1d, interpolate, randn_like
    
    // === Factory ===
    static zeros(shape) {
        return Tensor.fromBuffer(new Float32Array(shape.reduce((a,b) => a*b, 1)), shape);
    }
    
    static ones(shape) {
        const size = shape.reduce((a,b) => a*b, 1);
        return Tensor.fromBuffer(new Float32Array(size).fill(1), shape);
    }
    
    static randn(shape) {
        // Use native cuRAND version
        return native.Tensor.randn(shape);
    }
    
    static rand(shape) {
        const size = shape.reduce((a,b) => a*b, 1);
        const data = new Float32Array(size);
        for (let i = 0; i < size; i++) data[i] = Math.random();
        return Tensor.fromBuffer(data, shape);
    }
    
    static fromBuffer(data, shape) {
        // data: Float32Array, shape: number[]
        return native.Tensor.fromBuffer(data, shape);
    }
    
    static fromFlat(data, shape) {
        return Tensor.fromBuffer(new Float32Array(data), shape);
    }
    
    static cat(tensors, dim = 0) {
        return native.Tensor.cat(tensors, dim);
    }
    
    static where(cond, x, y) {
        return native.Tensor.where(cond, x, y);
    }
    
    static fromIntArray(data, shape) {
        return native.Tensor.fromIntArray(data, shape);
    }
    
    // === Convenience ===
    size(dim) {
        if (dim === undefined) return this.shape;
        return this.shape[dim];
    }
    
    dim() { return this.shape.length; }
    numel() { return this.shape.reduce((a,b) => a*b, 1); }
    
    // split -> array of tensors via slice
    split(dim, chunkSize) {
        const results = [];
        const total = this.shape[dim];
        for (let s = 0; s < total; s += chunkSize) {
            results.push(this.slice(dim, s, Math.min(s + chunkSize, total)));
        }
        return results;
    }
    
    // Chunk alias
    chunk(chunks, dim = 0) {
        const size = Math.ceil(this.shape[dim] / chunks);
        return this.split(dim, size);
    }
    
    // Permute (generalized transpose)
    permute(...dims) {
        if (dims.length === 1 && Array.isArray(dims[0])) dims = dims[0];
        let t = this;
        // Bubble sort dims via transpose pairs
        const perm = [...dims];
        const n = perm.length;
        for (let i = 0; i < n; i++) {
            while (perm[i] !== i) {
                const j = perm[i];
                t = t.transpose(i, j);
                [perm[i], perm[j]] = [perm[j], perm[i]];
            }
        }
        return t;
    }
    
    // F.interpolate convenience
    interpolateScale(scaleFactor, mode = 1, alignCorners = false) {
        const lastDim = this.shape[this.shape.length - 1];
        const target = Math.round(lastDim * scaleFactor);
        return this.interpolate(target, mode, alignCorners);
    }
    
    // View alias
    view(...shape) {
        if (shape.length === 1 && Array.isArray(shape[0])) shape = shape[0];
        return this.reshape(shape);
    }
    
    // Expand dims to match broadcast
    expand(shape) {
        // Simple expand: unsqueeze + repeat via add with zeros
        return this.add(Tensor.zeros(shape));
    }
    
    // Print
    toString() {
        const shapeStr = `[${this.shape.join(', ')}]`;
        return `Tensor(shape=${shapeStr})`;
    }
    
    [Symbol.for('nodejs.util.inspect.custom')]() {
        return this.toString();
    }
}

// === Exports ===
export const zeros = (shape) => Tensor.zeros(shape);
export const ones = (shape) => Tensor.ones(shape);
export const randn = (shape) => Tensor.randn(shape);
export const rand = (shape) => Tensor.rand(shape);
export const tensor = (data) => new Tensor(data);
export const cat = (tensors, dim) => Tensor.cat(tensors, dim);
export const where = (cond, x, y) => Tensor.where(cond, x, y);
