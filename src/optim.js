// optim.js - Optimizers for JsTorch autograd

import { createRequire } from 'module';
const require = createRequire(import.meta.url);
const native = require('../build/win/jstorch.node');

class Optimizer {
    constructor(parameters, defaults) {
        this.parameters = parameters; // GradTensor[]
        this.defaults = defaults;
        this.state = new Map();
    }
    
    zero_grad() {
        for (const p of this.parameters) {
            p.grad = null;
        }
    }
    
    step() {
        throw new Error('step() must be implemented');
    }
}

class SGD extends Optimizer {
    constructor(parameters, { lr = 0.01, momentum = 0, weight_decay = 0 } = {}) {
        super(parameters, { lr, momentum, weight_decay });
        this.lr = lr;
        this.momentum = momentum;
        this.weight_decay = weight_decay;
    }
    
    step() {
        for (const param of this.parameters) {
            if (!param.grad) continue;
            
            let grad = param.grad; // native.Tensor
            
            // Weight decay: grad += wd * param
            if (this.weight_decay !== 0) {
                grad = grad.add(param.data.mul(this.weight_decay));
            }
            
            // Momentum
            if (this.momentum !== 0) {
                let s = this.state.get(param);
                if (!s) {
                    s = { velocity: grad.clone() };
                    this.state.set(param, s);
                } else {
                    s.velocity = s.velocity.mul(this.momentum).add(grad);
                }
                grad = s.velocity;
            }
            
            // Update: param.data -= lr * grad (in-place via reassignment)
            param.data = param.data.sub(grad.mul(this.lr));
        }
    }
}

class Adam extends Optimizer {
    constructor(parameters, { lr = 0.001, betas = [0.9, 0.999], eps = 1e-8, weight_decay = 0 } = {}) {
        super(parameters, { lr, betas, eps, weight_decay });
        this.lr = lr;
        this.betas = betas;
        this.eps = eps;
        this.weight_decay = weight_decay;
        this.step_count = 0;
    }
    
    step() {
        this.step_count++;
        const bc1 = 1 - Math.pow(this.betas[0], this.step_count);
        const bc2 = 1 - Math.pow(this.betas[1], this.step_count);
        
        for (const param of this.parameters) {
            if (!param.grad) continue;
            
            let s = this.state.get(param);
            if (!s) {
                // Initialize m and v as zeros with same shape
                const zeros_buf = new Float32Array(param.data.shape.reduce((a,b) => a*b, 1));
                s = {
                    m: native.Tensor.fromBuffer(zeros_buf, [...param.data.shape]),
                    v: native.Tensor.fromBuffer(zeros_buf, [...param.data.shape])
                };
                this.state.set(param, s);
            }
            
            native.Tensor.adamStep(param.data, param.grad, s.m, s.v,
                this.lr, this.betas[0], this.betas[1], this.eps, bc1, bc2, this.weight_decay);
        }
    }
}

class AdamW extends Optimizer {
    constructor(parameters, { lr = 0.001, betas = [0.9, 0.999], eps = 1e-8, weight_decay = 0.01 } = {}) {
        super(parameters, { lr, betas, eps, weight_decay });
        this.lr = lr;
        this.betas = betas;
        this.eps = eps;
        this.weight_decay = weight_decay;
        this.step_count = 0;
    }
    
    step() {
        this.step_count++;
        
        for (const param of this.parameters) {
            if (!param.grad) continue;
            
            const grad = param.grad;
            
            let s = this.state.get(param);
            if (!s) {
                s = { m: grad.clone(), v: grad.mul(grad) };
                this.state.set(param, s);
            } else {
                s.m = s.m.mul(this.betas[0]).add(grad.mul(1 - this.betas[0]));
                s.v = s.v.mul(this.betas[1]).add(grad.mul(grad).mul(1 - this.betas[1]));
            }
            
            const bc1 = 1 - Math.pow(this.betas[0], this.step_count);
            const bc2 = 1 - Math.pow(this.betas[1], this.step_count);
            const m_hat = s.m.mul(1.0 / bc1);
            const v_hat = s.v.mul(1.0 / bc2);
            
            // Decoupled weight decay
            if (this.weight_decay !== 0) {
                param.data = param.data.mul(1 - this.lr * this.weight_decay);
            }
            
            const update = m_hat.div(v_hat.sqrt().add(this.eps)).mul(this.lr);
            param.data = param.data.sub(update);
        }
    }
}

export const optim = { Optimizer, SGD, Adam, AdamW };
