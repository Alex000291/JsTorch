// index.js - 完美架构入口
export { Tensor, zeros, ones, randn, rand, tensor } from './tensor.js';
export { nn } from './nn.js';
export { optim } from './optim.js';
export { no_grad, enable_grad, is_grad_enabled, set_grad_enabled } from './autograd.js';
