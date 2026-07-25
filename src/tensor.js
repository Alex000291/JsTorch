// tensor.js - Tensor JS wrapper
import { createRequire } from 'module';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
const require = createRequire(import.meta.url);

// Add DLL directories to PATH before loading native module
const __dirname = dirname(fileURLToPath(import.meta.url));
const dllDirs = ['build/win/dlls/cuda', 'build/win/dlls/cudnn'].map(d => join(__dirname, '..', d));
process.env.PATH = dllDirs.join(';') + ';' + process.env.PATH;

const native = require('../build/win/jstorch.node');

// Scalar ops handled natively in napi.cpp — no JS override needed
export class Tensor extends native.Tensor {
    // === Factory ===
    static zeros(shape) {
        return native.Tensor.fromBuffer(new Float32Array(shape.reduce((a,b) => a*b, 1)), shape);
    }
    static ones(shape) {
        return native.Tensor.full(shape, 1.0);
    }
    static randn(shape) {
        return native.Tensor.randn(shape);
    }
    static rand(shape) {
        const size = shape.reduce((a,b) => a*b, 1);
        const data = new Float32Array(size);
        for (let i = 0; i < size; i++) data[i] = Math.random();
        return Tensor.fromBuffer(data, shape);
    }
    static fromBuffer(data, shape) {
        return native.Tensor.fromBuffer(data, shape);
    }
    static fromFlat(data, shape) {
        return native.Tensor.fromBuffer(new Float32Array(data), shape);
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
    static full(shape, value) {
        return native.Tensor.full(shape, value);
    }
    static arange(start, end, step = 1) {
        if (end === undefined) { end = start; start = 0; }
        return native.Tensor.arange(start, end, step);
    }
    static linspace(start, end, steps) {
        const data = new Float32Array(steps);
        for (let i = 0; i < steps; i++)
            data[i] = steps === 1 ? start : start + i * (end - start) / (steps - 1);
        return Tensor.fromBuffer(data, [steps]);
    }
    static stack(tensors, dim = 0) {
        return Tensor.cat(tensors.map(t => t.unsqueeze(dim)), dim);
    }
    
    // === Convenience ===
    size(dim) {
        if (dim === undefined) return this.shape;
        return this.shape[dim];
    }
    dim() { return this.shape.length; }
    numel() { return this.shape.reduce((a,b) => a*b, 1); }
    
    split(dim, chunkSize) {
        const results = [];
        const total = this.shape[dim];
        for (let s = 0; s < total; s += chunkSize)
            results.push(this.slice(dim, s, Math.min(s + chunkSize, total)));
        return results;
    }
    
    chunk(chunks, dim = 0) {
        const size = Math.ceil(this.shape[dim] / chunks);
        return this.split(dim, size);
    }
    
    // zeros_like / ones_like / randn_like
    static zeros_like(t) { return Tensor.zeros([...t.shape]); }
    static ones_like(t) { return Tensor.ones([...t.shape]); }
    zeros_like() { return Tensor.zeros([...this.shape]); }
    ones_like() { return Tensor.ones([...this.shape]); }
    
    // masked_fill: where(mask == 0, self, value_tensor)
    masked_fill(mask, value) {
        const val = Tensor.full([...this.shape], value);
        // mask > 0 means fill with value, else keep self
        return Tensor.where(mask, val, this);
    }
    
    // tril / triu — implemented as static since native.Tensor doesn't have them
    
    // Permute
    permute(...dims) {
        if (dims.length === 1 && Array.isArray(dims[0])) dims = dims[0];
        let t = this;
        const perm = [...dims];
        for (let i = 0; i < perm.length; i++) {
            while (perm[i] !== i) {
                const j = perm[i];
                t = t.transpose(i, j);
                [perm[i], perm[j]] = [perm[j], perm[i]];
            }
        }
        return t;
    }
    
    // View alias
    view(...shape) {
        if (shape.length === 1 && Array.isArray(shape[0])) shape = shape[0];
        return this.reshape(shape);
    }
    
    // Expand (via broadcast add with zeros)
    expand(shape) {
        return this.add(Tensor.zeros(shape));
    }
    
    // F.interpolate convenience
    interpolateScale(scaleFactor, mode = 1, alignCorners = false) {
        const lastDim = this.shape[this.shape.length - 1];
        const target = Math.round(lastDim * scaleFactor);
        return this.interpolate(target, mode, alignCorners);
    }
    
    // Item — extract single value
    item() {
        const arr = this.toArray();
        return Array.isArray(arr) ? arr.flat(Infinity)[0] : arr;
    }
    
    // to() — dtype/device no-op for inference
    to() { return this; }
    
    // half/float — no-op (we only support float32)
    half() { return this; }
    float() { return this; }
    
    // Print
    toString() {
        return `Tensor(shape=[${this.shape.join(', ')}])`;
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
export const arange = (start, end, step) => Tensor.arange(start, end, step);
export const linspace = (start, end, steps) => Tensor.linspace(start, end, steps);
export const full = (shape, value) => Tensor.full(shape, value);
export const stack = (tensors, dim) => Tensor.stack(tensors, dim);
export const zeros_like = (t) => Tensor.zeros_like(t);
export const ones_like = (t) => Tensor.ones_like(t);

// tril / triu — standalone functions
export function tril(input, diagonal = 0) {
    const [H, W] = input.shape.slice(-2);
    const rows = native.Tensor.arange(0, H, 1).unsqueeze(1);
    const cols = native.Tensor.arange(0, W, 1).unsqueeze(0);
    const mask = rows.ge(cols.sub(diagonal));
    const z = native.Tensor.fromBuffer(new Float32Array(input.shape.reduce((a,b)=>a*b,1)), [...input.shape]);
    return native.Tensor.where(mask, input, z);
}

export function triu(input, diagonal = 0) {
    const [H, W] = input.shape.slice(-2);
    const rows = native.Tensor.arange(0, H, 1).unsqueeze(1);
    const cols = native.Tensor.arange(0, W, 1).unsqueeze(0);
    const mask = rows.le(cols.sub(diagonal));
    const z = native.Tensor.fromBuffer(new Float32Array(input.shape.reduce((a,b)=>a*b,1)), [...input.shape]);
    return native.Tensor.where(mask, input, z);
}
