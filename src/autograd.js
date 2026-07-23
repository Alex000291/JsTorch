/**
 * Autograd — Automatic Differentiation for JsTorch
 * 
 * GradTensor wraps native.Tensor, intercepts all ops, records to tape,
 * and supports backward() for automatic gradient computation.
 */

import { createRequire } from 'module';
const require = createRequire(import.meta.url);
const native = require('../build/jstorch.node');

// ==================== Grad context ====================
let _grad_enabled = true;

export function no_grad(fn) {
    const prev = _grad_enabled;
    _grad_enabled = false;
    try { return fn(); } finally { _grad_enabled = prev; }
}

export function enable_grad(fn) {
    const prev = _grad_enabled;
    _grad_enabled = true;
    try { return fn(); } finally { _grad_enabled = prev; }
}

export function is_grad_enabled() { return _grad_enabled; }

export function set_grad_enabled(enabled) {
    const prev = _grad_enabled;
    _grad_enabled = enabled;
    return prev;
}

// ==================== Helper: unwrap to native.Tensor ====================
function raw(x) {
    if (x instanceof GradTensor) return x.data;
    return x;
}

function needsGrad(...inputs) {
    if (!_grad_enabled) return false;
    for (const x of inputs) if (x instanceof GradTensor && x.requires_grad) return true;
    return false;
}

// Reduce grad to match input shape (sum out broadcasted dims)
function unbroadcast(grad, shape) {
    let g = grad;
    // Add leading dims if grad has more dims
    while (g.shape.length > shape.length) {
        g = g.sum(0, false);
    }
    // Sum along dims that were broadcasted (size 1 in original)
    for (let i = 0; i < shape.length; i++) {
        if (shape[i] === 1 && g.shape[i] !== 1) {
            g = g.sum(i, true);
        }
    }
    return g;
}

// ==================== GradTensor ====================
export class GradTensor {
    constructor(data, requires_grad = false, grad_fn = null, parents = [], saved = null) {
        this.data = data;              // native.Tensor
        this.requires_grad = requires_grad;
        this.grad = null;              // accumulated gradient (native.Tensor)
        this.grad_fn = grad_fn;        // function(grad) => [grad_for_parent0, ...]
        this._parents = parents;       // GradTensor[]
        this._saved = saved;           // saved tensors/metadata for backward
    }

    // ==================== Accessors ====================
    get shape() { return this.data.shape; }
    get ndim() { return this.data.ndim; }
    numel() { return this.shape.reduce((a, b) => a * b, 1); }
    size(dim) { return dim === undefined ? this.shape : this.shape[dim]; }
    dim() { return this.shape.length; }
    item() {
        const arr = this.data.toArray();
        return Array.isArray(arr) ? arr.flat(Infinity)[0] : arr;
    }
    toArray() { return this.data.toArray(); }
    toString() { return `GradTensor(shape=[${this.shape}], requires_grad=${this.requires_grad})`; }
    [Symbol.for('nodejs.util.inspect.custom')]() { return this.toString(); }

    // Detach from graph
    detach() { return new GradTensor(this.data, false); }

    // Enable grad in-place
    requires_grad_(val = true) { this.requires_grad = val; return this; }

    // Clone
    clone() {
        const out = new GradTensor(this.data.clone(), this.requires_grad);
        if (needsGrad(this)) {
            out.grad_fn = (g) => [g.clone()];
            out._parents = [this];
        }
        return out;
    }

    // ==================== backward ====================
    backward(upstream = null) {
        if (!upstream) {
            upstream = native.Tensor.full([...this.shape], 1.0);
        }
        // Topological sort
        const order = [];
        const visited = new Set();
        const topo = (node) => {
            if (visited.has(node) || !node.requires_grad) return;
            visited.add(node);
            if (node._parents) {
                for (const p of node._parents) {
                    if (p instanceof GradTensor) topo(p);
                }
            }
            order.push(node);
        };
        topo(this);

        // Backward pass
        this.grad = upstream;
        for (let i = order.length - 1; i >= 0; i--) {
            const node = order[i];
            if (!node.grad_fn || !node.grad) continue;
            const grads = node.grad_fn(node.grad);
            for (let j = 0; j < node._parents.length; j++) {
                const p = node._parents[j];
                if (!(p instanceof GradTensor) || !p.requires_grad) continue;
                if (grads[j] == null) continue;
                p.grad = p.grad ? p.grad.add(grads[j]) : grads[j];
            }
        }
    }

    // ==================== View ops (grad flows through) ====================
    reshape(shape) {
        const out_data = this.data.reshape(shape);
        if (!needsGrad(this)) return new GradTensor(out_data, false);
        const orig_shape = [...this.shape];
        const gt = new GradTensor(out_data, true, (g) => [g.reshape(orig_shape)], [this]);
        return gt;
    }
    view(...shape) {
        if (shape.length === 1 && Array.isArray(shape[0])) shape = shape[0];
        return this.reshape(shape);
    }
    squeeze(dim = -1) {
        const out_data = this.data.squeeze(dim);
        if (!needsGrad(this)) return new GradTensor(out_data, false);
        const orig_shape = [...this.shape];
        return new GradTensor(out_data, true, (g) => [g.reshape(orig_shape)], [this]);
    }
    unsqueeze(dim) {
        const out_data = this.data.unsqueeze(dim);
        if (!needsGrad(this)) return new GradTensor(out_data, false);
        const orig_shape = [...this.shape];
        return new GradTensor(out_data, true, (g) => [g.reshape(orig_shape)], [this]);
    }
    transpose(d0, d1) {
        const out_data = d0 !== undefined ? this.data.transpose(d0, d1) : this.data.transpose();
        if (!needsGrad(this)) return new GradTensor(out_data, false);
        return new GradTensor(out_data, true,
            (g) => [d0 !== undefined ? g.transpose(d0, d1) : g.transpose()], [this]);
    }
    contiguous() {
        const out_data = this.data.contiguous();
        if (!needsGrad(this)) return new GradTensor(out_data, false);
        return new GradTensor(out_data, true, (g) => [g], [this]);
    }
    flatten(start_dim, end_dim) {
        const out_data = this.data.flatten(start_dim, end_dim);
        if (!needsGrad(this)) return new GradTensor(out_data, false);
        const orig_shape = [...this.shape];
        return new GradTensor(out_data, true, (g) => [g.reshape(orig_shape)], [this]);
    }
    slice(dim, start, end) {
        const out_data = this.data.slice(dim, start, end);
        if (!needsGrad(this)) return new GradTensor(out_data, false);
        const s = [...this.shape];
        // backward: pad grad with zeros at the sliced-out positions
        return new GradTensor(out_data, true, (g) => {
            // Create zero tensor of original shape, scatter grad into it
            const z = native.Tensor.fromBuffer(new Float32Array(s.reduce((a, b) => a * b, 1)), s);
            // For simple 1D-axis slice: use pad
            const padBefore = start < 0 ? start + s[dim] : start;
            const padAfter = s[dim] - end;
            // Use cat to assemble [zeros_before, grad, zeros_after] along dim
            const parts = [];
            if (padBefore > 0) {
                const bs = [...g.shape]; bs[dim] = padBefore;
                parts.push(native.Tensor.fromBuffer(new Float32Array(bs.reduce((a, b) => a * b, 1)), bs));
            }
            parts.push(g);
            if (padAfter > 0) {
                const as_ = [...g.shape]; as_[dim] = padAfter;
                parts.push(native.Tensor.fromBuffer(new Float32Array(as_.reduce((a, b) => a * b, 1)), as_));
            }
            return [native.Tensor.cat(parts, dim)];
        }, [this]);
    }
    expand(shape) {
        const out_data = this.data.add(native.Tensor.fromBuffer(new Float32Array(shape.reduce((a, b) => a * b, 1)), shape));
        if (!needsGrad(this)) return new GradTensor(out_data, false);
        const orig_shape = [...this.shape];
        return new GradTensor(out_data, true, (g) => [unbroadcast(g, orig_shape)], [this]);
    }
    permute(...dims) {
        if (dims.length === 1 && Array.isArray(dims[0])) dims = dims[0];
        let t = this;
        const perm = [...dims];
        for (let i = 0; i < perm.length; i++) {
            while (perm[i] !== i) {
                const j = perm[i];
                t = t.transpose(i, j);
                [perm[i], perm[j]] = [perm[j], perm[i]];
            }
        }
        return t;
    }
    split(dim, chunkSize) {
        const results = [];
        const total = this.shape[dim];
        for (let s = 0; s < total; s += chunkSize)
            results.push(this.slice(dim, s, Math.min(s + chunkSize, total)));
        return results;
    }
    chunk(chunks, dim = 0) {
        const size = Math.ceil(this.shape[dim] / chunks);
        return this.split(dim, size);
    }

    // ==================== Factory (static, no grad) ====================
    static zeros(shape) { return new GradTensor(native.Tensor.fromBuffer(new Float32Array(shape.reduce((a, b) => a * b, 1)), shape), false); }
    static ones(shape) { return new GradTensor(native.Tensor.full(shape, 1.0), false); }
    static randn(shape) { return new GradTensor(native.Tensor.randn(shape), false); }
    static rand(shape) {
        const size = shape.reduce((a, b) => a * b, 1);
        const d = new Float32Array(size);
        for (let i = 0; i < size; i++) d[i] = Math.random();
        return new GradTensor(native.Tensor.fromBuffer(d, shape), false);
    }
    static full(shape, value) { return new GradTensor(native.Tensor.full(shape, value), false); }
    static arange(start, end, step = 1) {
        if (end === undefined) { end = start; start = 0; }
        return new GradTensor(native.Tensor.arange(start, end, step), false);
    }
    static linspace(start, end, steps) {
        const d = new Float32Array(steps);
        for (let i = 0; i < steps; i++)
            d[i] = steps === 1 ? start : start + i * (end - start) / (steps - 1);
        return new GradTensor(native.Tensor.fromBuffer(d, [steps]), false);
    }
    static fromBuffer(data, shape) { return new GradTensor(native.Tensor.fromBuffer(data, shape), false); }
    static fromFlat(data, shape) { return new GradTensor(native.Tensor.fromBuffer(new Float32Array(data), shape), false); }
    static fromIntArray(data, shape) { return new GradTensor(native.Tensor.fromIntArray(data, shape), false); }
    static zeros_like(t) { return GradTensor.zeros([...raw(t).shape]); }
    static ones_like(t) { return GradTensor.ones([...raw(t).shape]); }
    zeros_like() { return GradTensor.zeros([...this.shape]); }
    ones_like() { return GradTensor.ones([...this.shape]); }

    static cat(tensors, dim = 0) {
        const raws = tensors.map(t => raw(t));
        const out_data = native.Tensor.cat(raws, dim);
        if (!needsGrad(...tensors)) return new GradTensor(out_data, false);
        const sizes = tensors.map(t => t.shape[dim]);
        return new GradTensor(out_data, true, (g) => {
            // Split grad back along dim
            const grads = [];
            let offset = 0;
            for (const sz of sizes) {
                grads.push(g.slice(dim, offset, offset + sz));
                offset += sz;
            }
            return grads;
        }, tensors.filter(t => t instanceof GradTensor));
    }
    static stack(tensors, dim = 0) {
        return GradTensor.cat(tensors.map(t => t.unsqueeze(dim)), dim);
    }
    static where(cond, x, y) {
        const out_data = native.Tensor.where(raw(cond), raw(x), raw(y));
        if (!needsGrad(x, y)) return new GradTensor(out_data, false);
        const c = raw(cond);
        const parents = [x, y].filter(t => t instanceof GradTensor);
        return new GradTensor(out_data, true, (g) => {
            const z = native.Tensor.fromBuffer(new Float32Array(g.shape.reduce((a, b) => a * b, 1)), [...g.shape]);
            const gx = (x instanceof GradTensor && x.requires_grad) ? native.Tensor.where(c, g, z) : null;
            const gy = (y instanceof GradTensor && y.requires_grad) ? native.Tensor.where(c, z, g) : null;
            const grads = [];
            if (x instanceof GradTensor) grads.push(gx);
            if (y instanceof GradTensor) grads.push(gy);
            return grads;
        }, parents);
    }

    // Compatibility stubs
    to() { return this; }
    half() { return this; }
    float() { return this; }
    masked_fill(mask, value) {
        const val = native.Tensor.full([...this.shape], value);
        return GradTensor.where(
            mask instanceof GradTensor ? mask : new GradTensor(mask, false),
            new GradTensor(val, false),
            this
        );
    }
    randn_like() { return GradTensor.randn([...this.shape]); }
    interpolateScale(scaleFactor, mode = 1, alignCorners = false) {
        const lastDim = this.shape[this.shape.length - 1];
        return this.interpolate(Math.round(lastDim * scaleFactor), mode, alignCorners);
    }
}

// ==================== Backward registry ====================
// Each entry: (grad_output, saved) => [grad_input0, grad_input1, ...]
// 'saved' contains tensors/metadata stashed during forward

const B = {}; // backward functions

// --- Unary ops ---
B.neg =        (g, s) => [g.neg()];
B.abs =        (g, s) => [g.mul(s.x.sign())];
B.exp =        (g, s) => [g.mul(s.out)];
B.log =        (g, s) => [g.div(s.x)];
B.log1p =      (g, s) => [g.div(s.x.add(1))];
B.sqrt =       (g, s) => [g.div(s.out.mul(2))];
B.square =     (g, s) => [g.mul(s.x).mul(2)];
B.sin =        (g, s) => [g.mul(s.x.cos())];
B.cos =        (g, s) => [g.mul(s.x.sin().neg())];
B.sigmoid =    (g, s) => [g.mul(s.out).mul(s.out.neg().add(1))];
B.tanh =       (g, s) => [g.mul(s.out.square().neg().add(1))];
B.relu =       (g, s) => [g.mul(s.x.gt(0))];
B.leaky_relu = (g, s) => {
    const mask = s.x.gt(0);
    const neg_mask = s.x.le(0).mul(s.slope);
    return [g.mul(mask.add(neg_mask))];
};
B.silu =       (g, s) => {
    const sig = s.x.sigmoid();
    return [g.mul(sig.add(s.x.mul(sig).mul(sig.neg().add(1))))];
};
B.gelu =       (g, s) => {
    const sig = s.x.mul(1.702).sigmoid();
    return [g.mul(sig.add(s.x.mul(1.702).mul(sig).mul(sig.neg().add(1))))];
};
B.softplus =   (g, s) => [g.mul(s.out.neg().exp().neg().add(1))]; // 1 - exp(-softplus(x)) = sigmoid(x)
B.reciprocal = (g, s) => [g.neg().mul(s.out.square())];
B.sign =       (g, s) => [g.mul(0)]; // sign has zero gradient everywhere

// Parameterized unary
B.clamp =      (g, s) => [g.mul(s.x.ge(s.lo).mul(s.x.le(s.hi)))];
B.clamp_min =  (g, s) => [g.mul(s.x.ge(s.lo))];
B.clamp_max =  (g, s) => [g.mul(s.x.le(s.hi))];
B.fmod =       (g, s) => [g]; // d/dx fmod(x,d) = 1
B.pow_scalar = (g, s) => [g.mul(s.e).mul(s.x.pow(s.e - 1))];
B.mul_scalar = (g, s) => [g.mul(s.s)];
B.add_scalar = (g, s) => [g];

// --- Binary ops ---
B.add = (g, s) => [unbroadcast(g, s.a_shape), unbroadcast(g, s.b_shape)];
B.sub = (g, s) => [unbroadcast(g, s.a_shape), unbroadcast(g.neg(), s.b_shape)];
B.mul = (g, s) => [unbroadcast(g.mul(s.b), s.a_shape), unbroadcast(g.mul(s.a), s.b_shape)];
B.div = (g, s) => [unbroadcast(g.div(s.b), s.a_shape), unbroadcast(g.mul(s.a).neg().div(s.b.square()), s.b_shape)];
B.pow = (g, s) => [
    unbroadcast(g.mul(s.b).mul(s.a.pow(s.b.sub(1))), s.a_shape),
    unbroadcast(g.mul(s.out).mul(s.a.log()), s.b_shape)
];
B.maximum = (g, s) => [unbroadcast(g.mul(s.a.ge(s.b)), s.a_shape), unbroadcast(g.mul(s.b.gt(s.a)), s.b_shape)];
B.minimum = (g, s) => [unbroadcast(g.mul(s.a.le(s.b)), s.a_shape), unbroadcast(g.mul(s.b.lt(s.a)), s.b_shape)];
// Comparison ops: no gradient (returns 0/1)
B.gt = B.lt = B.ge = B.le = B.eq = B.ne = (g, s) => [g.mul(0), g.mul(0)];

// --- Matmul ---
B.matmul = (g, s) => {
    // a: [..., M, K], b: [..., K, N] => out: [..., M, N]
    // da = g @ b^T, db = a^T @ g
    return [g.matmul(s.b.transpose()), s.a.transpose().matmul(g)];
};

// --- Reduce ---
B.sum = (g, s) => {
    // Reshape grad to match input: unsqueeze reduced dim, then broadcast
    let ge = g;
    if (!s.keepdim) {
        // Reshape to put back the reduced dim as size-1
        const out_shape = [...s.x_shape];
        out_shape[s.dim] = 1;
        ge = g.reshape(out_shape);
    }
    // Broadcast to original shape
    return [ge.add(native.Tensor.fromBuffer(new Float32Array(s.x_shape.reduce((a, b) => a * b, 1)), s.x_shape))];
};
B.mean = (g, s) => {
    const n = s.dim >= 0 ? s.x_shape[s.dim] : s.x_shape.reduce((a, b) => a * b, 1);
    let ge = g.mul(1.0 / n);
    if (!s.keepdim) {
        const out_shape = [...s.x_shape];
        out_shape[s.dim] = 1;
        ge = ge.reshape(out_shape);
    }
    return [ge.add(native.Tensor.fromBuffer(new Float32Array(s.x_shape.reduce((a, b) => a * b, 1)), s.x_shape))];
};

// --- Conv ---
B.conv1d = (g, s) => {
    const dx = g.conv_transpose1d(s.weight, null, s.stride, s.padding, 0, s.dilation, s.groups);
    const C_in_g = s.x_shape[1] / s.groups;
    const K = s.weight.shape[2];
    const dw = native.Tensor.conv1dBackwardWeight(s.x, g, C_in_g, K, s.stride, s.padding, s.dilation, s.groups);
    const db = s.has_bias ? _reduce_bias_grad(g, [0, 2]) : null;
    return [dx, dw, db];
};
B.conv_transpose1d = (g, s) => {
    const dx = g.conv1d(s.weight, null, s.stride, s.padding, s.dilation, s.groups);
    const C_out_g = s.weight.shape[1];
    const K = s.weight.shape[2];
    const dw = native.Tensor.convTranspose1dBackwardWeight(s.x, g, C_out_g, K, s.stride, s.padding, s.dilation, s.groups);
    const db = s.has_bias ? _reduce_bias_grad(g, [0, 2]) : null;
    return [dx, dw, db];
};
B.conv2d = (g, s) => {
    const dx = g.conv_transpose2d(s.weight, null, s.sH, s.sW, s.pH, s.pW, 0, 0, s.dH, s.dW, s.groups);
    const Ci_g = s.x_shape[1] / s.groups;
    const kH = s.weight.shape[2], kW = s.weight.shape[3];
    const dw = native.Tensor.conv2dBackwardWeight(s.x, g, Ci_g, kH, kW, s.sH, s.sW, s.pH, s.pW, s.dH, s.dW, s.groups);
    const db = s.has_bias ? _reduce_bias_grad(g, [0, 2, 3]) : null;
    return [dx, dw, db];
};
B.conv_transpose2d = (g, s) => {
    const dx = g.conv2d(s.weight, null, s.sH, s.sW, s.pH, s.pW, s.dH, s.dW, s.groups);
    const Co_g = s.weight.shape[1];
    const kH = s.weight.shape[2], kW = s.weight.shape[3];
    const dw = native.Tensor.convTranspose2dBackwardWeight(s.x, g, Co_g, kH, kW, s.sH, s.sW, s.pH, s.pW, s.dH, s.dW, s.groups);
    const db = s.has_bias ? _reduce_bias_grad(g, [0, 2, 3]) : null;
    return [dx, dw, db];
};

// bias grad helper: sum over batch and spatial dims
function _reduce_bias_grad(g, dims) {
    let r = g;
    // sum dims in reverse order to keep indices stable
    for (let i = dims.length - 1; i >= 0; i--) {
        r = r.sum(dims[i], false);
    }
    return r;
}

// --- Misc ---
B.pad = (g, s) => {
    // Reverse pad: slice out the padded portion
    // padding is [left, right] for 1D, [left, right, top, bottom] for 2D etc.
    // We need to slice from padding[0] to shape[dim]-padding[1]
    const p = s.padding;
    let r = g;
    const ndim = g.shape.length;
    for (let i = 0; i < p.length; i += 2) {
        const dim = ndim - 1 - Math.floor(i / 2);
        const left = p[i], right = p[i + 1];
        r = r.slice(dim, left, r.shape[dim] - right);
    }
    return [r];
};
B.cumsum = (g, s) => {
    // backward of cumsum is reverse cumsum (flip + cumsum + flip)
    return [g.flip(s.dim).cumsum(s.dim).flip(s.dim)];
};
B.flip = (g, s) => [g.flip(s.dim)];
B.embedding = (g, s) => {
    // scatter_add: accumulate grad at index positions
    const vocab_size = s.weight_shape[0];
    // Flatten grad to [num_indices, embed_dim] if needed
    const embed_dim = s.weight_shape[1];
    const num_indices = s.indices.shape.reduce((a, b) => a * b, 1);
    const g_flat = g.reshape([num_indices, embed_dim]);
    // Use native scatter_add
    const dummy = native.Tensor.fromBuffer(new Float32Array(1), [1]);
    const gw = dummy.scatter_add(g_flat, s.indices, vocab_size);
    return [gw];
};
B.interpolate = (g, s) => {
    return [g.interp1d_backward(s.orig_size, s.mode, s.align_corners)];
};
B.avg_pool2d = (g, s) => {
    return [g.avgpool2d_backward(s.H, s.W, s.kH, s.kW, s.sH, s.sW, s.pH, s.pW, s.cip)];
};

// ==================== Generate op methods via metaprogramming ====================

// Simple unary ops: output = x.op()
const SIMPLE_UNARY = ['abs', 'sqrt', 'square', 'exp', 'log', 'sin', 'cos', 'neg',
    'sigmoid', 'tanh', 'relu', 'silu', 'gelu', 'softplus', 'log1p', 'reciprocal', 'sign'];

for (const op of SIMPLE_UNARY) {
    GradTensor.prototype[op] = function () {
        const out_data = this.data[op]();
        if (!needsGrad(this)) return new GradTensor(out_data, false);
        const saved = { x: this.data, out: out_data };
        return new GradTensor(out_data, true, (g) => B[op](g, saved), [this], saved);
    };
}

// Parameterized unary
GradTensor.prototype.leaky_relu = function (slope = 0.01) {
    const out_data = this.data.leaky_relu(slope);
    if (!needsGrad(this)) return new GradTensor(out_data, false);
    const saved = { x: this.data, slope };
    return new GradTensor(out_data, true, (g) => B.leaky_relu(g, saved), [this], saved);
};
GradTensor.prototype.clamp = function (lo, hi) {
    const out_data = this.data.clamp(lo, hi);
    if (!needsGrad(this)) return new GradTensor(out_data, false);
    const saved = { x: this.data, lo, hi };
    return new GradTensor(out_data, true, (g) => B.clamp(g, saved), [this], saved);
};
GradTensor.prototype.clamp_min = function (lo) {
    const out_data = this.data.clamp_min(lo);
    if (!needsGrad(this)) return new GradTensor(out_data, false);
    const saved = { x: this.data, lo };
    return new GradTensor(out_data, true, (g) => B.clamp_min(g, saved), [this], saved);
};
GradTensor.prototype.clamp_max = function (hi) {
    const out_data = this.data.clamp_max(hi);
    if (!needsGrad(this)) return new GradTensor(out_data, false);
    const saved = { x: this.data, hi };
    return new GradTensor(out_data, true, (g) => B.clamp_max(g, saved), [this], saved);
};
GradTensor.prototype.fmod = function (d) {
    const out_data = this.data.fmod(d);
    if (!needsGrad(this)) return new GradTensor(out_data, false);
    return new GradTensor(out_data, true, (g) => B.fmod(g, {}), [this]);
};
GradTensor.prototype.pow_scalar = function (e) {
    const out_data = this.data.pow_scalar(e);
    if (!needsGrad(this)) return new GradTensor(out_data, false);
    const saved = { x: this.data, e };
    return new GradTensor(out_data, true, (g) => B.pow_scalar(g, saved), [this], saved);
};
GradTensor.prototype.mul_scalar = function (s) {
    const out_data = this.data.mul_scalar(s);
    if (!needsGrad(this)) return new GradTensor(out_data, false);
    return new GradTensor(out_data, true, (g) => B.mul_scalar(g, { s }), [this]);
};
GradTensor.prototype.add_scalar = function (s) {
    const out_data = this.data.add_scalar(s);
    if (!needsGrad(this)) return new GradTensor(out_data, false);
    return new GradTensor(out_data, true, (g) => B.add_scalar(g, {}), [this]);
};

// Binary ops: output = a.op(b), where b can be scalar or GradTensor
const BINARY_OPS = ['add', 'sub', 'mul', 'div', 'pow', 'maximum', 'minimum', 'gt', 'lt', 'ge', 'le', 'eq', 'ne'];

for (const op of BINARY_OPS) {
    GradTensor.prototype[op] = function (other) {
        // Handle scalar: delegate to native scalar ops
        if (typeof other === 'number') {
            const out_data = this.data[op](other);
            if (!needsGrad(this)) return new GradTensor(out_data, false);
            // Scalar backward
            const scalarB = {
                add: (g) => [g],
                sub: (g) => [g],
                mul: (g) => [g.mul(other)],
                div: (g) => [g.mul(1.0 / other)],
                pow: (g) => [g.mul(other).mul(this.data.pow(other - 1))],
                gt: (g) => [g.mul(0)], lt: (g) => [g.mul(0)],
                ge: (g) => [g.mul(0)], le: (g) => [g.mul(0)],
                eq: (g) => [g.mul(0)], ne: (g) => [g.mul(0)],
                maximum: (g) => [g.mul(this.data.gt(other))],
                minimum: (g) => [g.mul(this.data.lt(other))],
            };
            return new GradTensor(out_data, true, scalarB[op], [this]);
        }
        const b_data = raw(other);
        const out_data = this.data[op](b_data);
        if (!needsGrad(this, other)) return new GradTensor(out_data, false);
        const saved = { a: this.data, b: b_data, out: out_data, a_shape: [...this.shape], b_shape: [...b_data.shape] };
        const parents = [];
        if (this instanceof GradTensor) parents.push(this);
        if (other instanceof GradTensor) parents.push(other);
        return new GradTensor(out_data, true, (g) => {
            const all_grads = B[op](g, saved);
            // Map grads to parents
            const result = [];
            let gi = 0;
            if (this instanceof GradTensor) result.push(all_grads[gi++]);
            if (other instanceof GradTensor) result.push(all_grads[gi++]);
            return result;
        }, parents, saved);
    };
}

// Matmul
GradTensor.prototype.matmul = function (other) {
    const b_data = raw(other);
    const out_data = this.data.matmul(b_data);
    if (!needsGrad(this, other)) return new GradTensor(out_data, false);
    const saved = { a: this.data, b: b_data };
    const parents = [];
    if (this instanceof GradTensor) parents.push(this);
    if (other instanceof GradTensor) parents.push(other);
    return new GradTensor(out_data, true, (g) => {
        const all_grads = B.matmul(g, saved);
        const result = [];
        let gi = 0;
        if (this instanceof GradTensor) result.push(all_grads[gi++]);
        if (other instanceof GradTensor) result.push(all_grads[gi++]);
        return result;
    }, parents, saved);
};

// Reduce ops
for (const op of ['sum', 'mean']) {
    GradTensor.prototype[op] = function (dim = -1, keepdim = false) {
        const out_data = this.data[op](dim, keepdim);
        if (!needsGrad(this)) return new GradTensor(out_data, false);
        // Normalize dim for backward
        const ndim = this.shape.length;
        const norm_dim = dim < 0 ? dim + ndim : dim;
        const saved = { x_shape: [...this.shape], dim: norm_dim, keepdim };
        return new GradTensor(out_data, true, (g) => B[op](g, saved), [this], saved);
    };
}
for (const op of ['max', 'min']) {
    const argOp = op === 'max' ? 'argmax' : 'argmin';
    GradTensor.prototype[op] = function (dim, keepdim = false) {
        const out_data = this.data[op](dim, keepdim);
        if (!needsGrad(this)) return new GradTensor(out_data, false);
        const x_data = this.data;
        const ndim = this.shape.length;
        const norm_dim = dim < 0 ? dim + ndim : dim;
        const saved = { dim: norm_dim, keepdim, x_shape: [...this.shape] };
        return new GradTensor(out_data, true, (g) => {
            // One-hot mask at argmax/argmin positions
            const indices = x_data[argOp](saved.dim); // shape with dim removed
            const idx_unsq = indices.unsqueeze(saved.dim); // [..., 1, ...]
            // Create range along the reduced dim
            const dim_size = saved.x_shape[saved.dim];
            const range = native.Tensor.arange(0, dim_size, 1);
            // Reshape range for broadcasting: [1,...,1, dim_size, 1,...,1]
            const bcast_shape = saved.x_shape.map((_, i) => i === saved.dim ? dim_size : 1);
            const range_bc = range.reshape(bcast_shape);
            // mask = (range == argmax_indices_unsqueezed)
            const mask = range_bc.eq(idx_unsq); // [x_shape], 1.0 at max pos, 0 elsewhere
            // Expand grad to input shape
            let g_exp = saved.keepdim ? g : g.unsqueeze(saved.dim);
            // broadcast to x_shape
            const zeros = native.Tensor.fromBuffer(
                new Float32Array(saved.x_shape.reduce((a, b) => a * b, 1)), saved.x_shape);
            g_exp = g_exp.add(zeros);
            return [g_exp.mul(mask)];
        }, [this], saved);
    };
}
GradTensor.prototype.argmax = function (dim) { return new GradTensor(this.data.argmax(dim), false); };
GradTensor.prototype.argmin = function (dim) { return new GradTensor(this.data.argmin(dim), false); };

// Tensor ops
GradTensor.prototype.flip = function (dim) {
    const out_data = this.data.flip(dim);
    if (!needsGrad(this)) return new GradTensor(out_data, false);
    return new GradTensor(out_data, true, (g) => B.flip(g, { dim }), [this]);
};
GradTensor.prototype.pad = function (padding, mode = 0, value = 0) {
    const out_data = this.data.pad(padding, mode, value);
    if (!needsGrad(this)) return new GradTensor(out_data, false);
    return new GradTensor(out_data, true, (g) => B.pad(g, { padding }), [this]);
};
GradTensor.prototype.cumsum = function (dim) {
    const out_data = this.data.cumsum(dim);
    if (!needsGrad(this)) return new GradTensor(out_data, false);
    return new GradTensor(out_data, true, (g) => B.cumsum(g, { dim }), [this]);
};
GradTensor.prototype.embedding = function (indices) {
    const idx_data = raw(indices);
    const out_data = this.data.embedding(idx_data);
    if (!needsGrad(this)) return new GradTensor(out_data, false);
    const saved = { indices: idx_data, weight_shape: [...this.shape] };
    return new GradTensor(out_data, true, (g) => B.embedding(g, saved), [this], saved);
};
GradTensor.prototype.randn_like = function () { return GradTensor.randn([...this.shape]); };

// Conv ops
GradTensor.prototype.conv1d = function (weight, bias, stride, padding, dilation, groups) {
    const w = raw(weight), b = bias ? raw(bias) : null;
    const out_data = this.data.conv1d(w, b, stride, padding, dilation, groups);
    if (!needsGrad(this, weight, bias)) return new GradTensor(out_data, false);
    const saved = { weight: w, x: this.data, x_shape: [...this.shape], stride, padding, dilation, groups, has_bias: !!bias };
    const parents = [this];
    if (weight instanceof GradTensor) parents.push(weight);
    if (bias instanceof GradTensor) parents.push(bias);
    return new GradTensor(out_data, true, (g) => B.conv1d(g, saved), parents, saved);
};
GradTensor.prototype.conv_transpose1d = function (weight, bias, stride, padding, output_padding, dilation, groups) {
    const w = raw(weight), b = bias ? raw(bias) : null;
    const out_data = this.data.conv_transpose1d(w, b, stride, padding, output_padding, dilation, groups);
    if (!needsGrad(this, weight, bias)) return new GradTensor(out_data, false);
    const saved = { weight: w, x: this.data, stride, padding, dilation, groups, has_bias: !!bias };
    const parents = [this];
    if (weight instanceof GradTensor) parents.push(weight);
    if (bias instanceof GradTensor) parents.push(bias);
    return new GradTensor(out_data, true, (g) => B.conv_transpose1d(g, saved), parents, saved);
};
GradTensor.prototype.conv2d = function (weight, bias, sH, sW, pH, pW, dH, dW, groups) {
    const w = raw(weight), b = bias ? raw(bias) : null;
    const out_data = this.data.conv2d(w, b, sH, sW, pH, pW, dH, dW, groups);
    if (!needsGrad(this, weight, bias)) return new GradTensor(out_data, false);
    const saved = { weight: w, x: this.data, x_shape: [...this.shape], sH, sW, pH, pW, dH, dW, groups, has_bias: !!bias };
    const parents = [this];
    if (weight instanceof GradTensor) parents.push(weight);
    if (bias instanceof GradTensor) parents.push(bias);
    return new GradTensor(out_data, true, (g) => B.conv2d(g, saved), parents, saved);
};
GradTensor.prototype.conv_transpose2d = function (weight, bias, sH, sW, pH, pW, opH, opW, dH, dW, groups) {
    const w = raw(weight), b = bias ? raw(bias) : null;
    const out_data = this.data.conv_transpose2d(w, b, sH, sW, pH, pW, opH, opW, dH, dW, groups);
    if (!needsGrad(this, weight, bias)) return new GradTensor(out_data, false);
    const saved = { weight: w, x: this.data, x_shape: [...this.shape], sH, sW, pH, pW, dH, dW, groups, has_bias: !!bias };
    const parents = [this];
    if (weight instanceof GradTensor) parents.push(weight);
    if (bias instanceof GradTensor) parents.push(bias);
    return new GradTensor(out_data, true, (g) => B.conv_transpose2d(g, saved), parents, saved);
};
GradTensor.prototype.avg_pool2d = function (kH, kW, sH, sW, pH, pW, cip = true) {
    const out_data = this.data.avg_pool2d(kH, kW, sH, sW, pH, pW, cip);
    if (!needsGrad(this)) return new GradTensor(out_data, false);
    const H = this.shape[2], W = this.shape[3];
    const saved = { H, W, kH, kW, sH, sW, pH, pW, cip };
    return new GradTensor(out_data, true, (g) => B.avg_pool2d(g, saved), [this], saved);
};
GradTensor.prototype.interpolate = function (target_size, mode = 0, align_corners = false) {
    const orig_size = this.shape[this.shape.length - 1];
    const out_data = this.data.interpolate(target_size, mode, align_corners);
    if (!needsGrad(this)) return new GradTensor(out_data, false);
    return new GradTensor(out_data, true, (g) => B.interpolate(g, { orig_size, mode, align_corners }), [this]);
};
// interp1d_backward — pass through to native
GradTensor.prototype.interp1d_backward = function (in_len, mode, align_corners) {
    return this.data.interp1d_backward(in_len, mode, align_corners);
};
