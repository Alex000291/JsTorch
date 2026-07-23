// optim.js - 优化器（完美重构）

class Optimizer {
    constructor(parameters, defaults) {
        this.parameters = parameters;
        this.defaults = defaults;
        this.state = new WeakMap();
    }
    
    zero_grad() {
        // TODO: 清零梯度
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
            
            let grad = param.grad;
            
            // Weight decay
            if (this.weight_decay !== 0) {
                grad = grad.add(param.mul(this.weight_decay));
            }
            
            // Momentum
            if (this.momentum !== 0) {
                let state = this.state.get(param);
                if (!state) {
                    state = { velocity: grad };
                    this.state.set(param, state);
                } else {
                    state.velocity = state.velocity.mul(this.momentum).add(grad);
                    grad = state.velocity;
                }
            }
            
            // Update
            param.sub_(grad.mul(this.lr));
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
        
        for (const param of this.parameters) {
            if (!param.grad) continue;
            
            let grad = param.grad;
            
            // Weight decay
            if (this.weight_decay !== 0) {
                grad = grad.add(param.mul(this.weight_decay));
            }
            
            // Get state
            let state = this.state.get(param);
            if (!state) {
                state = {
                    m: grad,
                    v: grad.square()
                };
                this.state.set(param, state);
            } else {
                // Update first moment: m = β1*m + (1-β1)*grad
                state.m = state.m.mul(this.betas[0]).add(grad.mul(1 - this.betas[0]));
                
                // Update second moment: v = β2*v + (1-β2)*grad²
                state.v = state.v.mul(this.betas[1]).add(grad.square().mul(1 - this.betas[1]));
            }
            
            // Bias correction
            const m_hat = state.m.div(1 - Math.pow(this.betas[0], this.step_count));
            const v_hat = state.v.div(1 - Math.pow(this.betas[1], this.step_count));
            
            // Update: param = param - lr * m_hat / (sqrt(v_hat) + eps)
            const update = m_hat.div(v_hat.sqrt().add(this.eps)).mul(this.lr);
            param.sub_(update);
        }
    }
}

class AdamW extends Adam {
    step() {
        this.step_count++;
        
        for (const param of this.parameters) {
            if (!param.grad) continue;
            
            const grad = param.grad;
            
            // Get state
            let state = this.state.get(param);
            if (!state) {
                state = {
                    m: grad,
                    v: grad.square()
                };
                this.state.set(param, state);
            } else {
                // Update moments (without weight decay in gradient)
                state.m = state.m.mul(this.betas[0]).add(grad.mul(1 - this.betas[0]));
                state.v = state.v.mul(this.betas[1]).add(grad.square().mul(1 - this.betas[1]));
            }
            
            // Bias correction
            const m_hat = state.m.div(1 - Math.pow(this.betas[0], this.step_count));
            const v_hat = state.v.div(1 - Math.pow(this.betas[1], this.step_count));
            
            // Update with AdamW-style weight decay
            const update = m_hat.div(v_hat.sqrt().add(this.eps)).mul(this.lr);
            
            // Decoupled weight decay: param = param * (1 - lr*wd) - lr * m_hat / (sqrt(v_hat) + eps)
            if (this.weight_decay !== 0) {
                param.mul_(1 - this.lr * this.weight_decay);
            }
            param.sub_(update);
        }
    }
}

export const optim = {
    Optimizer,
    SGD,
    Adam,
    AdamW
};
