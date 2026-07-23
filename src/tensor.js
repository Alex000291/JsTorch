// tensor.js - Tensor类的JS包装（完美重构）
import { createRequire } from 'module';
const require = createRequire(import.meta.url);
const native = require('../build/jstorch.node');

export class Tensor extends native.Tensor {
    // 继承所有native方法: add, sub, mul, div, abs, sqrt, square, exp, log, sin, cos, sigmoid, tanh, relu, sum, mean, reshape, transpose
    
    // === 构造函数 ===
    static zeros(shape) {
        const size = shape.reduce((a, b) => a * b, 1);
        const data = new Array(size).fill(0);
        return Tensor.fromFlat(data, shape);
    }
    
    static ones(shape) {
        const size = shape.reduce((a, b) => a * b, 1);
        const data = new Array(size).fill(1);
        return Tensor.fromFlat(data, shape);
    }
    
    static randn(shape) {
        const size = shape.reduce((a, b) => a * b, 1);
        const data = Array.from({length: size}, () => {
            // Box-Muller transform
            const u1 = Math.random();
            const u2 = Math.random();
            return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
        });
        return Tensor.fromFlat(data, shape);
    }
    
    static rand(shape) {
        const size = shape.reduce((a, b) => a * b, 1);
        const data = Array.from({length: size}, () => Math.random());
        return Tensor.fromFlat(data, shape);
    }
    
    static fromFlat(data, shape) {
        // 将flat数组转换为nested array
        function nest(arr, dims) {
            if (dims.length === 1) return arr.slice(0, dims[0]);
            const size = dims.slice(1).reduce((a, b) => a * b, 1);
            const result = [];
            for (let i = 0; i < dims[0]; i++) {
                result.push(nest(arr.slice(i * size, (i + 1) * size), dims.slice(1)));
            }
            return result;
        }
        return new Tensor(nest(data, shape));
    }
    
    // === In-place操作 ===
    add_(other) {
        const result = this.add(other);
        Object.setPrototypeOf(this, Object.getPrototypeOf(result));
        return this;
    }
    
    sub_(other) {
        const result = this.sub(other);
        Object.setPrototypeOf(this, Object.getPrototypeOf(result));
        return this;
    }
    
    mul_(scalar) {
        if (typeof scalar === 'number') {
            const result = this.mul(Tensor.fromFlat([scalar], [1]));
            Object.setPrototypeOf(this, Object.getPrototypeOf(result));
        }
        return this;
    }
    
    // === 便利方法 ===
    matmul(other) {
        // 简化版矩阵乘法：仅支持2D @ 2D
        // TODO: 实现真正的matmul
        throw new Error('matmul not implemented yet');
    }
    
    size(dim) {
        if (dim === undefined) return this.shape;
        return this.shape[dim];
    }
    
    dim() {
        return this.shape.length;
    }
    
    numel() {
        return this.shape.reduce((a, b) => a * b, 1);
    }
    
    // === 打印 ===
    toString() {
        const arr = this.toArray();
        const shapeStr = `[${this.shape.join(', ')}]`;
        return `Tensor(shape=${shapeStr}, data=${JSON.stringify(arr).slice(0, 100)}...)`;
    }
    
    [Symbol.for('nodejs.util.inspect.custom')]() {
        return this.toString();
    }
}

// === 导出便利函数 ===
export const zeros = (shape) => Tensor.zeros(shape);
export const ones = (shape) => Tensor.ones(shape);
export const randn = (shape) => Tensor.randn(shape);
export const rand = (shape) => Tensor.rand(shape);
export const tensor = (data) => new Tensor(data);
