/**
 * JsTorch - PyTorch-like Tensor Library for Node.js
 * Main Entry Point
 */

// Core
import { tensor, zeros, ones, Tensor } from './tensor.js';
import { no_grad, enable_grad, is_grad_enabled, set_grad_enabled } from './autograd.js';

// Neural Networks
import * as nn_module from './nn.js';

// Optimizers
import * as optim_module from './optim.js';

// CUDA utilities
const cuda = {
  is_available: () => true,  // Assume CUDA available if built
  device_count: () => 1,
  synchronize: () => {}
};

// Named exports
export { tensor, zeros, ones, Tensor };
export { no_grad, enable_grad, is_grad_enabled, set_grad_enabled };
export const nn = nn_module;
export const optim = optim_module;
export { cuda };

// Default export
export default {
  // Core
  tensor,
  zeros,
  ones,
  no_grad,
  enable_grad,
  is_grad_enabled,
  
  // Modules
  nn: nn_module,
  optim: optim_module,
  cuda
};
