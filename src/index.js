// index.js - JsTorch entry
export { Tensor, zeros, ones, randn, rand, tensor, cat, where,
         arange, linspace, full, stack, zeros_like, ones_like,
         tril, triu } from './tensor.js';
import native from './native.js';
export const GradTensor = native.GradTensor;
export { nn } from './nn.js';
export { optim } from './optim.js';
// Grad context — no-op (C++ autograd doesn't use global context)
export function no_grad(fn) { return fn(); }
export function enable_grad(fn) { return fn(); }
export function is_grad_enabled() { return true; }
export function set_grad_enabled(v) {}
export { loadModel, saveModel } from './loader.js';
