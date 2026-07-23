// nn.js - 神经网络层（完美重构）
import { Tensor, zeros, ones } from './tensor.js';

class Module {
    constructor() {
        this._parameters = {};
        this._modules = {};
        this.training = true;
    }
    
    parameters() {
        const params = [];
        for (const p of Object.values(this._parameters)) {
            params.push(p);
        }
        for (const m of Object.values(this._modules)) {
            params.push(...m.parameters());
        }
        return params;
    }
    
    train() { this.training = true; }
    eval() { this.training = false; }
    
    forward() { throw new Error('forward() must be implemented'); }
    __call__(x) { return this.forward(x); }
}

class Linear extends Module {
    constructor(in_features, out_features) {
        super();
        this._parameters.weight = Tensor.randn([out_features, in_features]);
        this._parameters.bias = zeros([out_features]);
    }
    
    forward(x) {
        // x: [batch, in] @ weight.T: [in, out] + bias: [out]
        return x.matmul(this._parameters.weight.transpose()).add(this._parameters.bias);
    }
}

class Conv1d extends Module {
    constructor(in_channels, out_channels, kernel_size) {
        super();
        this._parameters.weight = Tensor.randn([out_channels, in_channels, kernel_size]);
        this._parameters.bias = zeros([out_channels]);
    }
    
    forward(x) {
        // TODO: 实现conv1d
        throw new Error('Conv1d not implemented yet');
    }
}

class LayerNorm extends Module {
    constructor(normalized_shape, eps = 1e-5) {
        super();
        this.normalized_shape = Array.isArray(normalized_shape) ? normalized_shape : [normalized_shape];
        this.eps = eps;
        this._parameters.weight = ones(this.normalized_shape);
        this._parameters.bias = zeros(this.normalized_shape);
    }
    
    forward(x) {
        const mean = x.mean(-1);
        const variance = x.sub(mean).square().mean(-1);
        const normalized = x.sub(mean).div(variance.add(this.eps).sqrt());
        return normalized.mul(this._parameters.weight).add(this._parameters.bias);
    }
}

class ReLU extends Module {
    forward(x) { return x.relu(); }
}

class Sigmoid extends Module {
    forward(x) { return x.sigmoid(); }
}

class Tanh extends Module {
    forward(x) { return x.tanh(); }
}

class Sequential extends Module {
    constructor(...layers) {
        super();
        this._modules = layers;
    }
    
    forward(x) {
        for (const layer of this._modules) {
            x = layer(x);
        }
        return x;
    }
}

export const nn = {
    Module,
    Linear,
    Conv1d,
    LayerNorm,
    ReLU,
    Sigmoid,
    Tanh,
    Sequential
};
