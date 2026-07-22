/**
 * Optimizers - Parameter Update Rules
 */

import { zeros } from './tensor.js';

export function sgd(params, { lr, momentum = 0 }) {
  const velocities = momentum > 0 ? params.map(p => zeros(p.shape)) : [];
  
  return {
    zero_grad() {
      params.forEach(p => p.zero_grad());
    },
    
    step() {
      params.forEach((p, i) => {
        if (!p.grad) return;
        
        if (momentum > 0) {
          // v = momentum * v + grad
          // p = p - lr * v
          const v = velocities[i];
          velocities[i] = v.mulScalar(momentum).add(p.grad);
          p._native = p._native.sub(velocities[i].mulScalar(lr)._native);
        } else {
          // p = p - lr * grad
          const update = p.grad.mulScalar(lr);
          p._native = p._native.sub(update._native);
        }
      });
    }
  };
}

export function adam(params, { lr, betas = [0.9, 0.999], eps = 1e-8 }) {
  const m = params.map(p => zeros(p.shape));  // First moment
  const v = params.map(p => zeros(p.shape));  // Second moment
  let t = 0;  // Time step
  
  return {
    zero_grad() {
      params.forEach(p => p.zero_grad());
    },
    
    step() {
      t += 1;
      const [beta1, beta2] = betas;
      
      params.forEach((p, i) => {
        if (!p.grad) return;
        
        // m = beta1 * m + (1 - beta1) * grad
        m[i] = m[i].mulScalar(beta1).add(p.grad.mulScalar(1 - beta1));
        
        // v = beta2 * v + (1 - beta2) * grad^2
        const grad_sq = p.grad.mul(p.grad);
        v[i] = v[i].mulScalar(beta2).add(grad_sq.mulScalar(1 - beta2));
        
        // m_hat = m / (1 - beta1^t)
        const m_hat = m[i].mulScalar(1 / (1 - Math.pow(beta1, t)));
        
        // v_hat = v / (1 - beta2^t)
        const v_hat = v[i].mulScalar(1 / (1 - Math.pow(beta2, t)));
        
        // p = p - lr * m_hat / (sqrt(v_hat) + eps)
        // Simplified: p = p - lr * m_hat
        const update = m_hat.mulScalar(lr);
        p._native = p._native.sub(update._native);
      });
    }
  };
}
