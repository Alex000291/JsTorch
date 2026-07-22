/**
 * Autograd - Automatic Differentiation with AsyncLocalStorage
 */

import { AsyncLocalStorage } from 'async_hooks';

const gradContext = new AsyncLocalStorage();

/**
 * Disable gradient computation within a function scope
 */
export function no_grad(fn) {
  return gradContext.run({ enabled: false }, fn);
}

/**
 * Enable gradient computation within a function scope
 */
export function enable_grad(fn) {
  return gradContext.run({ enabled: true }, fn);
}

/**
 * Check if gradient computation is currently enabled
 */
export function is_grad_enabled() {
  const ctx = gradContext.getStore();
  return ctx?.enabled ?? true;  // Default: enabled
}

/**
 * Set gradient mode globally (for model.train() / model.eval())
 */
export function set_grad_enabled(enabled) {
  // Returns a function that sets the mode
  return enabled ? enable_grad : no_grad;
}
