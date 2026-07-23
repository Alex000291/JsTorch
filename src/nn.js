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

// ==================== Conv2d ====================
class Conv2d extends Module {
    constructor(in_channels, out_channels, kernel_size, {
        stride = 1, padding = 0, dilation = 1, groups = 1, bias = true
    } = {}) {
        super();
        const kh = Array.isArray(kernel_size) ? kernel_size[0] : kernel_size;
        const kw = Array.isArray(kernel_size) ? kernel_size[1] : kernel_size;
        this.stride = Array.isArray(stride) ? stride : [stride, stride];
        this.padding = Array.isArray(padding) ? padding : [padding, padding];
        this.dilation = Array.isArray(dilation) ? dilation : [dilation, dilation];
        this.groups = groups;
        this._parameters.weight = Tensor.randn([out_channels, in_channels / groups, kh, kw]);
        if (bias) this._parameters.bias = zeros([out_channels]);
    }
    
    forward(x) {
        const [sH, sW] = this.stride;
        const [pH, pW] = this.padding;
        const [dH, dW] = this.dilation;
        return x.conv2d(
            this._parameters.weight,
            this._parameters.bias || null,
            sH, sW, pH, pW, dH, dW, this.groups
        );
    }
}

// ==================== ConvTranspose2d ====================
class ConvTranspose2d extends Module {
    constructor(in_channels, out_channels, kernel_size, {
        stride = 1, padding = 0, output_padding = 0, dilation = 1, groups = 1, bias = true
    } = {}) {
        super();
        const kh = Array.isArray(kernel_size) ? kernel_size[0] : kernel_size;
        const kw = Array.isArray(kernel_size) ? kernel_size[1] : kernel_size;
        this.stride = Array.isArray(stride) ? stride : [stride, stride];
        this.padding = Array.isArray(padding) ? padding : [padding, padding];
        this.output_padding = Array.isArray(output_padding) ? output_padding : [output_padding, output_padding];
        this.dilation = Array.isArray(dilation) ? dilation : [dilation, dilation];
        this.groups = groups;
        this._parameters.weight = Tensor.randn([in_channels, out_channels / groups, kh, kw]);
        if (bias) this._parameters.bias = zeros([out_channels]);
    }
    
    forward(x) {
        const [sH, sW] = this.stride;
        const [pH, pW] = this.padding;
        const [opH, opW] = this.output_padding;
        const [dH, dW] = this.dilation;
        return x.conv_transpose2d(
            this._parameters.weight,
            this._parameters.bias || null,
            sH, sW, pH, pW, opH, opW, dH, dW, this.groups
        );
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

class BatchNorm2d extends BatchNorm1d {
    forward(x) {
        // x: [B, C, H, W] -> reshape to [B, C, H*W], apply BN1d, reshape back
        const [B, C, H, W] = x.shape;
        const flat = x.reshape([B, C, H * W]);
        const mean = this._buffers.running_mean.reshape([1, C, 1]);
        const var_ = this._buffers.running_var.reshape([1, C, 1]);
        const w = this._parameters.weight.reshape([1, C, 1]);
        const b = this._parameters.bias.reshape([1, C, 1]);
        const eps = Tensor.fromFlat([this.eps], [1]);
        const out = flat.sub(mean).div(var_.add(eps).sqrt()).mul(w).add(b);
        return out.reshape([B, C, H, W]);
    }
}

// ==================== AvgPool2d ====================
class AvgPool2d extends Module {
    constructor(kernel_size, stride, padding = 0) {
        super();
        this.kH = Array.isArray(kernel_size) ? kernel_size[0] : kernel_size;
        this.kW = Array.isArray(kernel_size) ? kernel_size[1] : kernel_size;
        if (stride === undefined) stride = kernel_size;
        this.sH = Array.isArray(stride) ? stride[0] : stride;
        this.sW = Array.isArray(stride) ? stride[1] : stride;
        this.pH = Array.isArray(padding) ? padding[0] : padding;
        this.pW = Array.isArray(padding) ? padding[1] : padding;
    }
    forward(x) {
        return x.avg_pool2d(this.kH, this.kW, this.sH, this.sW, this.pH, this.pW);
    }
}

// ==================== Upsample ====================
class Upsample extends Module {
    constructor({ scale_factor, size, mode = 'nearest' } = {}) {
        super();
        this.scale_factor = scale_factor;
        this.size = size;
        this.mode = mode;
    }
    forward(x) {
        const modeInt = this.mode === 'nearest' ? 0 : 1;
        if (this.size !== undefined) return x.interpolate(this.size, modeInt);
        const lastDim = x.shape[x.shape.length - 1];
        return x.interpolate(Math.round(lastDim * this.scale_factor), modeInt);
    }
}

// ==================== GroupNorm ====================
class GroupNorm extends Module {
    constructor(num_groups, num_channels, eps = 1e-5) {
        super();
        this.num_groups = num_groups;
        this.num_channels = num_channels;
        this.eps = eps;
        this._parameters.weight = ones([num_channels]);
        this._parameters.bias = zeros([num_channels]);
    }
    forward(x) {
        // x: [B, C, ...]
        const B = x.shape[0], C = x.shape[1];
        const spatial = x.shape.reduce((a,b) => a*b, 1) / (B * C);
        const G = this.num_groups;
        const CpG = C / G;
        // Reshape to [B, G, CpG * spatial]
        const y = x.reshape([B, G, CpG * spatial]);
        const mean = y.mean(2, true);
        const variance = y.sub(mean).pow(2).mean(2, true);
        const eps = Tensor.fromFlat([this.eps], [1]);
        const normed = y.sub(mean).div(variance.add(eps).sqrt());
        const out = normed.reshape([B, C, ...x.shape.slice(2)]);
        // Apply weight and bias per channel
        const w = this._parameters.weight.reshape([1, C, ...Array(x.shape.length - 2).fill(1)]);
        const b = this._parameters.bias.reshape([1, C, ...Array(x.shape.length - 2).fill(1)]);
        return out.mul(w).add(b);
    }
}

// ==================== GRU ====================
class GRU extends Module {
    constructor(input_size, hidden_size, { num_layers = 1, bidirectional = false, batch_first = false } = {}) {
        super();
        this.input_size = input_size;
        this.hidden_size = hidden_size;
        this.num_layers = num_layers;
        this.bidirectional = bidirectional;
        this.batch_first = batch_first;
        const num_dir = bidirectional ? 2 : 1;
        for (let l = 0; l < num_layers; l++) {
            const in_sz = l === 0 ? input_size : hidden_size * num_dir;
            // PyTorch-compatible parameter names
            this._parameters[`weight_ih_l${l}`] = Tensor.randn([3 * hidden_size, in_sz]);
            this._parameters[`weight_hh_l${l}`] = Tensor.randn([3 * hidden_size, hidden_size]);
            this._parameters[`bias_ih_l${l}`] = zeros([3 * hidden_size]);
            this._parameters[`bias_hh_l${l}`] = zeros([3 * hidden_size]);
            if (bidirectional) {
                this._parameters[`weight_ih_l${l}_reverse`] = Tensor.randn([3 * hidden_size, in_sz]);
                this._parameters[`weight_hh_l${l}_reverse`] = Tensor.randn([3 * hidden_size, hidden_size]);
                this._parameters[`bias_ih_l${l}_reverse`] = zeros([3 * hidden_size]);
                this._parameters[`bias_hh_l${l}_reverse`] = zeros([3 * hidden_size]);
            }
        }
    }
    
    _cell(x, h, wih, whh, bih, bhh) {
        // x: [batch, in], h: [batch, hidden]
        const hs = this.hidden_size;
        const gi = x.matmul(wih.transpose()).add(bih);
        const gh = h.matmul(whh.transpose()).add(bhh);
        const r = gi.slice(1, 0, hs).add(gh.slice(1, 0, hs)).sigmoid();
        const z = gi.slice(1, hs, 2*hs).add(gh.slice(1, hs, 2*hs)).sigmoid();
        const n = gi.slice(1, 2*hs, 3*hs).add(r.mul(gh.slice(1, 2*hs, 3*hs))).tanh();
        return Tensor.ones([...z.shape]).sub(z).mul(n).add(z.mul(h));
    }
    
    _run_direction(input, seqLen, batch, layerIdx, suffix = '') {
        const reverse = suffix === '_reverse';
        const wih = this._parameters[`weight_ih_l${layerIdx}${suffix}`];
        const whh = this._parameters[`weight_hh_l${layerIdx}${suffix}`];
        const bih = this._parameters[`bias_ih_l${layerIdx}${suffix}`];
        const bhh = this._parameters[`bias_hh_l${layerIdx}${suffix}`];
        let h = Tensor.zeros([batch, this.hidden_size]);
        const outputs = [];
        const start = reverse ? seqLen - 1 : 0;
        const end = reverse ? -1 : seqLen;
        const step = reverse ? -1 : 1;
        for (let t = start; t !== end; t += step) {
            const xt = input.slice(0, t, t + 1).squeeze(0);
            h = this._cell(xt, h, wih, whh, bih, bhh);
            if (reverse) outputs.unshift(h.unsqueeze(0));
            else outputs.push(h.unsqueeze(0));
        }
        return { output: Tensor.cat(outputs, 0), h };
    }
    
    forward(x, h0 = null) {
        // x: [seq, batch, features] or [batch, seq, features] if batch_first
        if (this.batch_first) x = x.transpose(0, 1);
        const seqLen = x.shape[0], batch = x.shape[1];
        let input = x;
        const hResults = [];
        
        for (let l = 0; l < this.num_layers; l++) {
            const fwd = this._run_direction(input, seqLen, batch, l);
            hResults.push(fwd.h.unsqueeze(0));
            if (this.bidirectional) {
                const rev = this._run_direction(input, seqLen, batch, l, '_reverse');
                input = Tensor.cat([fwd.output, rev.output], 2);
                hResults.push(rev.h.unsqueeze(0));
            } else {
                input = fwd.output;
            }
        }
        let output = input;
        if (this.batch_first) output = output.transpose(0, 1);
        return [output, Tensor.cat(hResults, 0)];
    }
}

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
    Module, Linear, Embedding,
    Conv1d, ConvTranspose1d, Conv2d, ConvTranspose2d,
    AvgPool2d, Upsample, GRU,
    LayerNorm, GroupNorm, BatchNorm1d, BatchNorm2d,
    ReLU, LeakyReLU, Sigmoid, Tanh, GELU, SiLU, Dropout,
    Sequential, ModuleList,
    weight_norm, remove_weight_norm,
    functional: F, F,
};
