// nn.js - Neural network modules for RVC
import { Tensor, zeros, ones } from './tensor.js';

// ==================== #27 Module base ====================
class Module {
    constructor() {
        this._parameters = {};
        this._buffers = {};
        this._modules = {};
        this.training = true;
    }
    
    parameters() {
        const params = [];
        for (const p of Object.values(this._parameters)) params.push(p);
        for (const m of Object.values(this._modules)) {
            if (m && m.parameters) params.push(...m.parameters());
        }
        return params;
    }
    
    train(mode = true) {
        this.training = mode;
        for (const m of Object.values(this._modules)) {
            if (m && m.train) m.train(mode);
        }
        return this;
    }
    
    eval() { return this.train(false); }
    
    forward(...args) { throw new Error('forward() must be implemented'); }
    
    // Make module callable
    call(...args) { return this.forward(...args); }
    
    // #27: load_state_dict
    load_state_dict(stateDict, prefix = '') {
        // Load parameters
        for (const [name, param] of Object.entries(this._parameters)) {
            const key = prefix ? `${prefix}.${name}` : name;
            if (stateDict[key]) {
                this._parameters[name] = stateDict[key];
            }
        }
        // Load buffers
        for (const [name, buf] of Object.entries(this._buffers)) {
            const key = prefix ? `${prefix}.${name}` : name;
            if (stateDict[key]) {
                this._buffers[name] = stateDict[key];
            }
        }
        // Recurse into submodules
        for (const [name, mod] of Object.entries(this._modules)) {
            if (mod && mod.load_state_dict) {
                const subPrefix = prefix ? `${prefix}.${name}` : name;
                mod.load_state_dict(stateDict, subPrefix);
            }
        }
    }
    
    // Named modules iterator
    named_modules(prefix = '') {
        const result = [[prefix, this]];
        for (const [name, mod] of Object.entries(this._modules)) {
            if (mod && mod.named_modules) {
                const p = prefix ? `${prefix}.${name}` : name;
                result.push(...mod.named_modules(p));
            }
        }
        return result;
    }
}

// ==================== #22 Linear ====================
class Linear extends Module {
    constructor(in_features, out_features, bias = true) {
        super();
        this.in_features = in_features;
        this.out_features = out_features;
        this._parameters.weight = Tensor.randn([out_features, in_features]);
        if (bias) this._parameters.bias = zeros([out_features]);
    }
    
    forward(x) {
        // x: [..., in] -> [..., out]
        let out = x.matmul(this._parameters.weight.transpose());
        if (this._parameters.bias) out = out.add(this._parameters.bias);
        return out;
    }
}

// ==================== #23 Embedding ====================
class Embedding extends Module {
    constructor(num_embeddings, embedding_dim) {
        super();
        this.num_embeddings = num_embeddings;
        this.embedding_dim = embedding_dim;
        this._parameters.weight = Tensor.randn([num_embeddings, embedding_dim]);
    }
    
    forward(indices) {
        return this._parameters.weight.embedding(indices);
    }
}

// ==================== #24 Conv1d ====================
class Conv1d extends Module {
    constructor(in_channels, out_channels, kernel_size, {
        stride = 1, padding = 0, dilation = 1, groups = 1, bias = true
    } = {}) {
        super();
        this.in_channels = in_channels;
        this.out_channels = out_channels;
        this.kernel_size = kernel_size;
        this.stride = stride;
        this.padding = padding;
        this.dilation = dilation;
        this.groups = groups;
        this._parameters.weight = Tensor.randn([out_channels, in_channels / groups, kernel_size]);
        if (bias) this._parameters.bias = zeros([out_channels]);
    }
    
    forward(x) {
        return x.conv1d(
            this._parameters.weight,
            this._parameters.bias || null,
            this.stride, this.padding, this.dilation, this.groups
        );
    }
}

// ==================== #25 ConvTranspose1d ====================
class ConvTranspose1d extends Module {
    constructor(in_channels, out_channels, kernel_size, {
        stride = 1, padding = 0, output_padding = 0, dilation = 1, groups = 1, bias = true
    } = {}) {
        super();
        this.in_channels = in_channels;
        this.out_channels = out_channels;
        this.kernel_size = kernel_size;
        this.stride = stride;
        this.padding = padding;
        this.output_padding = output_padding;
        this.dilation = dilation;
        this.groups = groups;
        this._parameters.weight = Tensor.randn([in_channels, out_channels / groups, kernel_size]);
        if (bias) this._parameters.bias = zeros([out_channels]);
    }
    
    forward(x) {
        return x.conv_transpose1d(
            this._parameters.weight,
            this._parameters.bias || null,
            this.stride, this.padding, this.output_padding, this.dilation, this.groups
        );
    }
}

// ==================== Conv2d (for RMVPE) ====================
class Conv2d extends Module {
    constructor(in_channels, out_channels, kernel_size, {
        stride = 1, padding = 0, dilation = 1, groups = 1, bias = true
    } = {}) {
        super();
        const kh = Array.isArray(kernel_size) ? kernel_size[0] : kernel_size;
        const kw = Array.isArray(kernel_size) ? kernel_size[1] : kernel_size;
        this.stride = Array.isArray(stride) ? stride : [stride, stride];
        this.padding = Array.isArray(padding) ? padding : [padding, padding];
        this._parameters.weight = Tensor.randn([out_channels, in_channels / groups, kh, kw]);
        if (bias) this._parameters.bias = zeros([out_channels]);
    }
    
    forward(x) {
        // TODO: implement when needed for RMVPE
        throw new Error('Conv2d forward not yet implemented in CUDA');
    }
}

// ==================== LayerNorm ====================
class LayerNorm extends Module {
    constructor(normalized_shape, eps = 1e-5) {
        super();
        this.normalized_shape = Array.isArray(normalized_shape) ? normalized_shape : [normalized_shape];
        this.eps = eps;
        this._parameters.weight = ones(this.normalized_shape);
        this._parameters.bias = zeros(this.normalized_shape);
    }
    
    forward(x) {
        const mean = x.mean(-1, true);
        const variance = x.sub(mean).square().mean(-1, true);
        const eps_t = Tensor.fromFlat([this.eps], [1]);
        const normalized = x.sub(mean).div(variance.add(eps_t).sqrt());
        return normalized.mul(this._parameters.weight).add(this._parameters.bias);
    }
}

// ==================== BatchNorm ====================
class BatchNorm1d extends Module {
    constructor(num_features, eps = 1e-5, momentum = 0.1) {
        super();
        this.num_features = num_features;
        this.eps = eps;
        this._parameters.weight = ones([num_features]);
        this._parameters.bias = zeros([num_features]);
        this._buffers.running_mean = zeros([num_features]);
        this._buffers.running_var = ones([num_features]);
    }
    
    forward(x) {
        // Inference mode: use running stats
        const mean = this._buffers.running_mean;
        const var_ = this._buffers.running_var;
        const eps_t = Tensor.fromFlat([this.eps], [1]);
        const normalized = x.sub(mean).div(var_.add(eps_t).sqrt());
        return normalized.mul(this._parameters.weight).add(this._parameters.bias);
    }
}

class BatchNorm2d extends BatchNorm1d {}

// ==================== #26 Activations ====================
class LeakyReLU extends Module {
    constructor(negative_slope = 0.01) {
        super();
        this.negative_slope = negative_slope;
    }
    forward(x) { return x.leaky_relu(this.negative_slope); }
}

class ReLU extends Module { forward(x) { return x.relu(); } }
class Sigmoid extends Module { forward(x) { return x.sigmoid(); } }
class Tanh extends Module { forward(x) { return x.tanh(); } }
class GELU extends Module { forward(x) { return x.gelu(); } }
class SiLU extends Module { forward(x) { return x.silu(); } }

// ==================== #26 Dropout ====================
class Dropout extends Module {
    constructor(p = 0.5) { super(); this.p = p; }
    forward(x) { return x; } // inference: pass-through
}

// ==================== Sequential ====================
class Sequential extends Module {
    constructor(...layers) {
        super();
        layers.forEach((l, i) => { this._modules[String(i)] = l; });
    }
    forward(x) {
        for (const m of Object.values(this._modules)) x = m.forward(x);
        return x;
    }
}

// ==================== ModuleList ====================
class ModuleList extends Module {
    constructor(modules = []) {
        super();
        modules.forEach((m, i) => { this._modules[String(i)] = m; });
    }
    push(m) {
        this._modules[String(Object.keys(this._modules).length)] = m;
    }
    get length() { return Object.keys(this._modules).length; }
    [Symbol.iterator]() {
        const values = Object.values(this._modules);
        let i = 0;
        return { next: () => i < values.length ? { value: values[i++], done: false } : { done: true } };
    }
    forEach(fn) { Object.values(this._modules).forEach(fn); }
    get(i) { return this._modules[String(i)]; }
}

// ==================== #21 weight_norm ====================
function weight_norm(layer, name = 'weight') {
    const w = layer._parameters[name];
    // Decompose: w = g * (v / ||v||)
    // For inference with pre-computed weights, this is identity
    // RVC calls remove_weight_norm before inference, so weights are already merged
    return layer;
}

function remove_weight_norm(layer) {
    // No-op for inference — weights already merged in checkpoint
    return layer;
}

// ==================== Functional ====================
const F = {
    leaky_relu: (x, slope = 0.01) => x.leaky_relu(slope),
    relu: (x) => x.relu(),
    sigmoid: (x) => x.sigmoid(),
    tanh: (x) => x.tanh(),
    gelu: (x) => x.gelu(),
    silu: (x) => x.silu(),
    softmax: (x, dim = -1) => {
        const max = x.max(dim, true);
        const exp = x.sub(max).exp();
        return exp.div(exp.sum(dim, true));
    },
    interpolate: (x, { size, scale_factor, mode = 'nearest', align_corners = false } = {}) => {
        const modeInt = mode === 'nearest' ? 0 : 1;
        if (size !== undefined) return x.interpolate(size, modeInt, align_corners);
        if (scale_factor !== undefined) return x.interpolateScale(scale_factor, modeInt, align_corners);
        throw new Error('interpolate: need size or scale_factor');
    },
    pad: (x, padding, mode = 'constant', value = 0) => {
        const modeInt = mode === 'reflect' ? 1 : 0;
        return x.pad(padding, modeInt, value);
    },
};

// ==================== Export ====================
export const nn = {
    Module, Linear, Embedding, Conv1d, ConvTranspose1d, Conv2d,
    LayerNorm, BatchNorm1d, BatchNorm2d,
    ReLU, LeakyReLU, Sigmoid, Tanh, GELU, SiLU, Dropout,
    Sequential, ModuleList,
    weight_norm, remove_weight_norm,
    functional: F, F,
};
