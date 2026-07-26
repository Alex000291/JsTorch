#pragma once
#include "tensor.hpp"
#include "allocator.hpp"
#include <memory>
#include <vector>
#include <functional>
#include <unordered_set>
#include <algorithm>

namespace jstorch {

class GradTensor;
using GradPtr = std::shared_ptr<GradTensor>;

// ==================== GradTensor ====================
// Wraps a Tensor with autograd metadata. Analogous to PyTorch's Variable.

class GradTensor : public std::enable_shared_from_this<GradTensor> {
public:
    Tensor data;
    bool requires_grad = false;
    std::shared_ptr<Tensor> grad_;  // accumulated gradient (heap to avoid needing default ctor)
    bool has_grad = false;

    // Backward graph: grad_fn computes gradients for parents
    using GradFn = std::function<std::vector<Tensor>(const Tensor&)>;
    GradFn grad_fn;
    std::vector<GradPtr> parents;

    // Constructors
    GradTensor() = default;
    explicit GradTensor(Tensor d, bool rg = false)
        : data(std::move(d)), requires_grad(rg) {}

    const Shape& shape() const { return data.shape(); }
    int ndim() const { return data.ndim(); }
    int size() const { return data.size(); }

    // ==================== Backward engine ====================
    static void backward(GradPtr root, Tensor upstream) {
        // Topological sort
        std::vector<GradTensor*> order;
        std::unordered_set<GradTensor*> visited;
        topo_sort(root.get(), order, visited);

        root->grad_ = std::make_shared<Tensor>(std::move(upstream));
        root->has_grad = true;

        for (int i = (int)order.size() - 1; i >= 0; i--) {
            GradTensor* node = order[i];
            if (!node->grad_fn || !node->has_grad) continue;
            auto grads = node->grad_fn(*node->grad_);
            for (int j = 0; j < (int)node->parents.size(); j++) {
                auto& p = node->parents[j];
                if (!p || !p->requires_grad) continue;
                if (j >= (int)grads.size()) continue;
                if (p->has_grad) {
                    *p->grad_ = p->grad_->add(grads[j]);
                } else {
                    p->grad_ = std::make_shared<Tensor>(std::move(grads[j]));
                    p->has_grad = true;
                }
            }
        }
    }

    // ==================== Factory ====================
    static GradPtr make(Tensor d, bool rg = false) {
        return std::make_shared<GradTensor>(std::move(d), rg);
    }

    // Make result with grad_fn recorded
    static GradPtr make_with_grad(Tensor d, GradFn fn, std::vector<GradPtr> parents) {
        auto r = std::make_shared<GradTensor>(std::move(d), true);
        r->grad_fn = std::move(fn);
        r->parents = std::move(parents);
        return r;
    }

    // Check if any input needs grad
    static bool any_requires_grad(const std::vector<GradPtr>& inputs) {
        for (auto& t : inputs)
            if (t && t->requires_grad) return true;
        return false;
    }

    // ==================== Ops (forward + backward recording) ====================

    // --- Unary ops ---
    GradPtr exp_() {
        Tensor out = data.exp();
        if (!requires_grad) return make(std::move(out));
        Tensor saved = out; // exp backward needs output
        auto self = shared_from_this();
        return make_with_grad(std::move(out), [saved](const Tensor& g) {
            return std::vector<Tensor>{g.mul(saved)};
        }, {self});
    }

    GradPtr neg_() {
        Tensor out = data.neg();
        if (!requires_grad) return make(std::move(out));
        return make_with_grad(std::move(out), [](const Tensor& g) {
            return std::vector<Tensor>{g.neg()};
        }, {shared_from_this()});
    }

    GradPtr sigmoid_() {
        Tensor out = data.sigmoid();
        if (!requires_grad) return make(std::move(out));
        Tensor saved = out;
        return make_with_grad(std::move(out), [saved](const Tensor& g) {
            // sig' = sig * (1 - sig)
            return std::vector<Tensor>{g.mul(saved).mul(saved.neg().add_scalar(1.0f))};
        }, {shared_from_this()});
    }

    GradPtr tanh_() {
        Tensor out = data.tanh();
        if (!requires_grad) return make(std::move(out));
        Tensor saved = out;
        return make_with_grad(std::move(out), [saved](const Tensor& g) {
            return std::vector<Tensor>{g.mul(saved.square().neg().add_scalar(1.0f))};
        }, {shared_from_this()});
    }

    GradPtr relu_() {
        Tensor out = data.relu();
        if (!requires_grad) return make(std::move(out));
        Tensor saved_x = data;
        return make_with_grad(std::move(out), [saved_x](const Tensor& g) {
            return std::vector<Tensor>{g.mul(saved_x.gt(Tensor::full(saved_x.shape(), 0.0f)))};
        }, {shared_from_this()});
    }

    GradPtr log_() {
        Tensor out = data.log();
        if (!requires_grad) return make(std::move(out));
        Tensor saved_x = data;
        return make_with_grad(std::move(out), [saved_x](const Tensor& g) {
            return std::vector<Tensor>{g.div(saved_x)};
        }, {shared_from_this()});
    }

    GradPtr sqrt_() {
        Tensor out = data.sqrt();
        if (!requires_grad) return make(std::move(out));
        Tensor saved_out = out;
        return make_with_grad(std::move(out), [saved_out](const Tensor& g) {
            return std::vector<Tensor>{g.div(saved_out.mul_scalar(2.0f))};
        }, {shared_from_this()});
    }

    GradPtr square_() {
        Tensor out = data.square();
        if (!requires_grad) return make(std::move(out));
        Tensor saved_x = data;
        return make_with_grad(std::move(out), [saved_x](const Tensor& g) {
            return std::vector<Tensor>{g.mul(saved_x).mul_scalar(2.0f)};
        }, {shared_from_this()});
    }

    GradPtr silu_() {
        Tensor out = data.silu();
        if (!requires_grad) return make(std::move(out));
        Tensor saved_x = data;
        return make_with_grad(std::move(out), [saved_x](const Tensor& g) {
            Tensor sig = saved_x.sigmoid();
            return std::vector<Tensor>{g.mul(sig.add(saved_x.mul(sig).mul(sig.neg().add_scalar(1.0f))))};
        }, {shared_from_this()});
    }

    GradPtr gelu_() {
        Tensor out = data.gelu();
        if (!requires_grad) return make(std::move(out));
        Tensor saved_x = data;
        return make_with_grad(std::move(out), [saved_x](const Tensor& g) {
            Tensor sig = saved_x.mul_scalar(1.702f).sigmoid();
            return std::vector<Tensor>{g.mul(sig.add(saved_x.mul_scalar(1.702f).mul(sig).mul(sig.neg().add_scalar(1.0f))))};
        }, {shared_from_this()});
    }

    GradPtr abs_() {
        Tensor out = data.abs();
        if (!requires_grad) return make(std::move(out));
        Tensor saved_x = data;
        return make_with_grad(std::move(out), [saved_x](const Tensor& g) {
            return std::vector<Tensor>{g.mul(saved_x.sign())};
        }, {shared_from_this()});
    }

    GradPtr sin_() {
        Tensor out = data.sin();
        if (!requires_grad) return make(std::move(out));
        Tensor saved_x = data;
        return make_with_grad(std::move(out), [saved_x](const Tensor& g) {
            return std::vector<Tensor>{g.mul(saved_x.cos())};
        }, {shared_from_this()});
    }

    GradPtr cos_() {
        Tensor out = data.cos();
        if (!requires_grad) return make(std::move(out));
        Tensor saved_x = data;
        return make_with_grad(std::move(out), [saved_x](const Tensor& g) {
            return std::vector<Tensor>{g.mul(saved_x.sin().neg())};
        }, {shared_from_this()});
    }

    GradPtr reciprocal_() {
        Tensor out = data.reciprocal();
        if (!requires_grad) return make(std::move(out));
        Tensor saved_out = out;
        return make_with_grad(std::move(out), [saved_out](const Tensor& g) {
            return std::vector<Tensor>{g.neg().mul(saved_out.square())};
        }, {shared_from_this()});
    }

    GradPtr softplus_() {
        Tensor out = data.softplus();
        if (!requires_grad) return make(std::move(out));
        Tensor saved_out = out;
        return make_with_grad(std::move(out), [saved_out](const Tensor& g) {
            // softplus'(x) = sigmoid(x) = 1 - exp(-softplus(x))
            return std::vector<Tensor>{g.mul(saved_out.neg().exp().neg().add_scalar(1.0f))};
        }, {shared_from_this()});
    }

    GradPtr sign_() {
        Tensor out = data.sign();
        if (!requires_grad) return make(std::move(out));
        return make_with_grad(std::move(out), [s=data.shape()](const Tensor& g) {
            return std::vector<Tensor>{Tensor::full(s, 0.0f)};
        }, {shared_from_this()});
    }

    GradPtr log1p_() {
        Tensor out = data.log1p();
        if (!requires_grad) return make(std::move(out));
        Tensor saved_x = data;
        return make_with_grad(std::move(out), [saved_x](const Tensor& g) {
            return std::vector<Tensor>{g.div(saved_x.add_scalar(1.0f))};
        }, {shared_from_this()});
    }

    // Parameterized unary
    GradPtr pow_scalar_(float e) {
        Tensor out = data.pow_scalar(e);
        if (!requires_grad) return make(std::move(out));
        Tensor saved_x = data;
        return make_with_grad(std::move(out), [saved_x, e](const Tensor& g) {
            return std::vector<Tensor>{g.mul_scalar(e).mul(saved_x.pow_scalar(e - 1.0f))};
        }, {shared_from_this()});
    }

    GradPtr mul_scalar_(float s) {
        Tensor out = data.mul_scalar(s);
        if (!requires_grad) return make(std::move(out));
        return make_with_grad(std::move(out), [s](const Tensor& g) {
            return std::vector<Tensor>{g.mul_scalar(s)};
        }, {shared_from_this()});
    }

    GradPtr add_scalar_(float s) {
        Tensor out = data.add_scalar(s);
        if (!requires_grad) return make(std::move(out));
        return make_with_grad(std::move(out), [](const Tensor& g) {
            return std::vector<Tensor>{g};
        }, {shared_from_this()});
    }

    GradPtr leaky_relu_(float slope = 0.01f) {
        Tensor out = data.leaky_relu(slope);
        if (!requires_grad) return make(std::move(out));
        Tensor sx = data;
        return make_with_grad(std::move(out),
            [sx, slope](const Tensor& g) {
                Tensor mask = sx.gt(Tensor::full(sx.shape(), 0.f));
                Tensor slope_mask = sx.le(Tensor::full(sx.shape(), 0.f)).mul_scalar(slope);
                return std::vector<Tensor>{g.mul(mask.add(slope_mask))};
            }, {shared_from_this()});
    }

    GradPtr clamp_(float lo, float hi) {
        Tensor out = data.clamp(lo, hi);
        if (!requires_grad) return make(std::move(out));
        Tensor sx = data;
        return make_with_grad(std::move(out),
            [sx, lo, hi](const Tensor& g) {
                return std::vector<Tensor>{g.mul(sx.ge(Tensor::full(sx.shape(), lo)).mul(sx.le(Tensor::full(sx.shape(), hi))))};
            }, {shared_from_this()});
    }

    GradPtr clamp_min_(float lo) {
        Tensor out = data.clamp_min(lo);
        if (!requires_grad) return make(std::move(out));
        Tensor sx = data;
        return make_with_grad(std::move(out),
            [sx, lo](const Tensor& g) {
                return std::vector<Tensor>{g.mul(sx.ge(Tensor::full(sx.shape(), lo)))};
            }, {shared_from_this()});
    }

    GradPtr clamp_max_(float hi) {
        Tensor out = data.clamp_max(hi);
        if (!requires_grad) return make(std::move(out));
        Tensor sx = data;
        return make_with_grad(std::move(out),
            [sx, hi](const Tensor& g) {
                return std::vector<Tensor>{g.mul(sx.le(Tensor::full(sx.shape(), hi)))};
            }, {shared_from_this()});
    }

    GradPtr fmod_(float d) {
        Tensor out = data.fmod(d);
        if (!requires_grad) return make(std::move(out));
        return make_with_grad(std::move(out),
            [](const Tensor& g) { return std::vector<Tensor>{g}; },
            {shared_from_this()});
    }

    // --- Binary ops (unbroadcast is public for tape backward) ---
    public:
    static Tensor unbroadcast(const Tensor& grad, const Shape& target) {
        Tensor g = grad;
        while ((int)g.shape().size() > (int)target.size())
            g = g.sum(0, false);
        for (int i = 0; i < (int)target.size(); i++)
            if (target[i] == 1 && g.shape()[i] != 1)
                g = g.sum(i, true);
        return g;
    }

    GradPtr add_(GradPtr other) {
        Tensor out = data.add(other->data);
        if (!requires_grad && !other->requires_grad) return make(std::move(out));
        Shape sa = data.shape(), sb = other->data.shape();
        auto self = shared_from_this();
        return make_with_grad(std::move(out), [sa, sb](const Tensor& g) {
            return std::vector<Tensor>{unbroadcast(g, sa), unbroadcast(g, sb)};
        }, {self, other});
    }

    GradPtr sub_(GradPtr other) {
        Tensor out = data.sub(other->data);
        if (!requires_grad && !other->requires_grad) return make(std::move(out));
        Shape sa = data.shape(), sb = other->data.shape();
        auto self = shared_from_this();
        return make_with_grad(std::move(out), [sa, sb](const Tensor& g) {
            return std::vector<Tensor>{unbroadcast(g, sa), unbroadcast(g.neg(), sb)};
        }, {self, other});
    }

    GradPtr mul_(GradPtr other) {
        Tensor out = data.mul(other->data);
        if (!requires_grad && !other->requires_grad) return make(std::move(out));
        Tensor sa_data = data, sb_data = other->data;
        Shape sa = data.shape(), sb = other->data.shape();
        auto self = shared_from_this();
        return make_with_grad(std::move(out), [sa_data, sb_data, sa, sb](const Tensor& g) {
            return std::vector<Tensor>{
                unbroadcast(g.mul(sb_data), sa),
                unbroadcast(g.mul(sa_data), sb)
            };
        }, {self, other});
    }

    GradPtr div_(GradPtr other) {
        Tensor out = data.div(other->data);
        if (!requires_grad && !other->requires_grad) return make(std::move(out));
        Tensor sa_data = data, sb_data = other->data;
        Shape sa = data.shape(), sb = other->data.shape();
        auto self = shared_from_this();
        return make_with_grad(std::move(out), [sa_data, sb_data, sa, sb](const Tensor& g) {
            return std::vector<Tensor>{
                unbroadcast(g.div(sb_data), sa),
                unbroadcast(g.mul(sa_data).neg().div(sb_data.square()), sb)
            };
        }, {self, other});
    }

    GradPtr pow_(GradPtr other) {
        Tensor out = data.pow(other->data);
        if (!requires_grad && !other->requires_grad) return make(std::move(out));
        Tensor a = data, b = other->data, o = out;
        Shape sa = data.shape(), sb = other->data.shape();
        auto self = shared_from_this();
        return make_with_grad(std::move(out), [a, b, o, sa, sb](const Tensor& g) {
            return std::vector<Tensor>{
                unbroadcast(g.mul(b).mul(a.pow(b.add_scalar(-1.0f))), sa),
                unbroadcast(g.mul(o).mul(a.log()), sb)
            };
        }, {self, other});
    }

    // Comparison ops — no gradient
    GradPtr gt_(GradPtr other) { return make(data.gt(other->data)); }
    GradPtr lt_(GradPtr other) { return make(data.lt(other->data)); }
    GradPtr ge_(GradPtr other) { return make(data.ge(other->data)); }
    GradPtr le_(GradPtr other) { return make(data.le(other->data)); }
    GradPtr eq_(GradPtr other) { return make(data.eq(other->data)); }
    GradPtr ne_(GradPtr other) { return make(data.ne(other->data)); }

    GradPtr maximum_(GradPtr other) {
        Tensor out = data.maximum(other->data);
        if (!requires_grad && !other->requires_grad) return make(std::move(out));
        Tensor a = data, b = other->data;
        Shape sa = a.shape(), sb = b.shape();
        return make_with_grad(std::move(out),
            [a, b, sa, sb](const Tensor& g) {
                return std::vector<Tensor>{
                    unbroadcast(g.mul(a.ge(b)), sa),
                    unbroadcast(g.mul(b.gt(a)), sb)
                };
            }, {shared_from_this(), other});
    }

    GradPtr minimum_(GradPtr other) {
        Tensor out = data.minimum(other->data);
        if (!requires_grad && !other->requires_grad) return make(std::move(out));
        Tensor a = data, b = other->data;
        Shape sa = a.shape(), sb = b.shape();
        return make_with_grad(std::move(out),
            [a, b, sa, sb](const Tensor& g) {
                return std::vector<Tensor>{
                    unbroadcast(g.mul(a.le(b)), sa),
                    unbroadcast(g.mul(b.lt(a)), sb)
                };
            }, {shared_from_this(), other});
    }

    // --- Matmul ---
    GradPtr matmul_(GradPtr other) {
        Tensor out = data.matmul(other->data);
        if (!requires_grad && !other->requires_grad) return make(std::move(out));
        Tensor a = data, b = other->data;
        auto self = shared_from_this();
        return make_with_grad(std::move(out), [a, b](const Tensor& g) {
            return std::vector<Tensor>{g.matmul(b.transpose()), a.transpose().matmul(g)};
        }, {self, other});
    }

    // --- Reduce ---
    GradPtr sum_(int dim, bool keepdim) {
        Tensor out = data.sum(dim, keepdim);
        if (!requires_grad) return make(std::move(out));
        Shape x_shape = data.shape();
        int ndim = data.ndim();
        int norm_dim = dim < 0 ? dim + ndim : dim;
        auto self = shared_from_this();
        return make_with_grad(std::move(out), [x_shape, norm_dim, keepdim](const Tensor& g) {
            Tensor ge = g;
            if (!keepdim) {
                Shape s = x_shape; s[norm_dim] = 1;
                ge = ge.reshape(s);
            }
            return std::vector<Tensor>{ge.add(Tensor::full(x_shape, 0.0f))};
        }, {self});
    }

    GradPtr mean_(int dim, bool keepdim) {
        Tensor out = data.mean(dim, keepdim);
        if (!requires_grad) return make(std::move(out));
        Shape x_shape = data.shape();
        int ndim = data.ndim();
        int norm_dim = dim < 0 ? dim + ndim : dim;
        float n = (float)(norm_dim >= 0 ? x_shape[norm_dim] : data.size());
        auto self = shared_from_this();
        return make_with_grad(std::move(out), [x_shape, norm_dim, keepdim, n](const Tensor& g) {
            Tensor ge = g.mul_scalar(1.0f / n);
            if (!keepdim) {
                Shape s = x_shape; s[norm_dim] = 1;
                ge = ge.reshape(s);
            }
            return std::vector<Tensor>{ge.add(Tensor::full(x_shape, 0.0f))};
        }, {self});
    }

    // --- View ops (grad flows through) ---
    GradPtr slice_(int dim, int start, int end) {
        Tensor out = data.slice(dim, start, end);
        if (!requires_grad) return make(std::move(out));
        Shape orig = data.shape();
        auto self = shared_from_this();
        return make_with_grad(std::move(out), [dim, start, end, orig](const Tensor& g) {
            // Backward: pad gradient back to original shape using cat of zeros
            std::vector<Tensor> parts;
            if (start > 0) {
                Shape s = orig; s[dim] = start;
                Tensor z(s);
                cudaMemsetAsync(z.data<float>(), 0, z.size() * sizeof(float), 0);
                parts.push_back(std::move(z));
            }
            parts.push_back(g.contiguous());
            int after = orig[dim] - end;
            if (after > 0) {
                Shape s = orig; s[dim] = after;
                Tensor z(s);
                cudaMemsetAsync(z.data<float>(), 0, z.size() * sizeof(float), 0);
                parts.push_back(std::move(z));
            }
            if (parts.size() == 1) return std::vector<Tensor>{parts[0]};
            return std::vector<Tensor>{Tensor::cat(parts, dim)};
        }, {self});
    }

    GradPtr squeeze_(int dim) {
        Tensor out = data.squeeze(dim);
        if (!requires_grad) return make(std::move(out));
        return make_with_grad(std::move(out), [dim](const Tensor& g) {
            return std::vector<Tensor>{g.unsqueeze(dim)};
        }, {shared_from_this()});
    }

    GradPtr reshape_(const Shape& shape) {
        Tensor out = data.reshape(shape);
        if (!requires_grad) return make(std::move(out));
        Shape orig = data.shape();
        return make_with_grad(std::move(out), [orig](const Tensor& g) {
            return std::vector<Tensor>{g.reshape(orig)};
        }, {shared_from_this()});
    }

    GradPtr transpose_(int d0, int d1) {
        Tensor out = data.transpose(d0, d1);
        if (!requires_grad) return make(std::move(out));
        return make_with_grad(std::move(out), [d0, d1](const Tensor& g) {
            return std::vector<Tensor>{g.transpose(d0, d1)};
        }, {shared_from_this()});
    }

    GradPtr transpose_() {
        Tensor out = data.transpose();
        if (!requires_grad) return make(std::move(out));
        return make_with_grad(std::move(out), [](const Tensor& g) {
            return std::vector<Tensor>{g.transpose()};
        }, {shared_from_this()});
    }

    // --- Conv2d ---
    GradPtr conv2d_(GradPtr weight, GradPtr bias,
                    int sH, int sW, int pH, int pW, int dH, int dW, int groups) {
        const Tensor* b_ptr = bias ? &bias->data : nullptr;
        Tensor out = data.conv2d(weight->data, b_ptr, sH, sW, pH, pW, dH, dW, groups);
        bool need = requires_grad || weight->requires_grad || (bias && bias->requires_grad);
        if (!need) return make(std::move(out));

        Tensor saved_x = data, saved_w = weight->data;
        Shape x_shape = data.shape();
        Shape w_shape = weight->data.shape();
        bool has_bias = (bias != nullptr);
        auto self = shared_from_this();
        std::vector<GradPtr> par = {self, weight};
        if (bias) par.push_back(bias);

        return make_with_grad(std::move(out),
            [saved_x, saved_w, x_shape, w_shape, sH, sW, pH, pW, dH, dW, groups, has_bias](const Tensor& g) {
                Tensor dx(x_shape), dw(w_shape);
                if (has_bias) {
                    int Co = w_shape[0];
                    Tensor db({Co});
                    Tensor::conv2d_backward_cudnn(saved_x, saved_w, g, dx, dw, &db, sH, sW, pH, pW, dH, dW, groups);
                    return std::vector<Tensor>{dx, dw, db};
                } else {
                    Tensor::conv2d_backward_cudnn(saved_x, saved_w, g, dx, dw, nullptr, sH, sW, pH, pW, dH, dW, groups);
                    return std::vector<Tensor>{dx, dw};
                }
            }, par);
    }

    // --- AvgPool2d ---
    GradPtr avg_pool2d_(int kH, int kW, int sH, int sW, int pH, int pW, bool cip) {
        Tensor out = data.avg_pool2d(kH, kW, sH, sW, pH, pW, cip);
        if (!requires_grad) return make(std::move(out));
        int H = data.shape()[2], W = data.shape()[3];
        return make_with_grad(std::move(out),
            [H, W, kH, kW, sH, sW, pH, pW, cip](const Tensor& g) {
                return std::vector<Tensor>{g.avgpool2d_backward(H, W, kH, kW, sH, sW, pH, pW, cip)};
            }, {shared_from_this()});
    }

    GradPtr max_pool2d_(int kH, int kW, int sH, int sW, int pH, int pW) {
        auto [out, indices] = data.max_pool2d(kH, kW, sH, sW, pH, pW);
        if (!requires_grad) return make(std::move(out));
        int B = data.shape()[0], C = data.shape()[1], H = data.shape()[2], W = data.shape()[3];
        auto idx_ptr = std::make_shared<Tensor>(std::move(indices));
        return make_with_grad(std::move(out),
            [B, C, H, W, idx_ptr](const Tensor& g) {
                return std::vector<Tensor>{Tensor::maxpool2d_backward(g, *idx_ptr, B, C, H, W)};
            }, {shared_from_this()});
    }

    // --- Conv1d ---
    GradPtr conv1d_(GradPtr weight, GradPtr bias,
                    int stride, int padding, int dilation, int groups) {
        const Tensor* b_ptr = bias ? &bias->data : nullptr;
        Tensor out = data.conv1d(weight->data, b_ptr, stride, padding, dilation, groups);
        bool need = requires_grad || weight->requires_grad || (bias && bias->requires_grad);
        if (!need) return make(std::move(out));
        Tensor saved_x = data, saved_w = weight->data;
        Shape x_shape = data.shape(), w_shape = weight->data.shape();
        bool has_bias = (bias != nullptr);
        auto self = shared_from_this();
        std::vector<GradPtr> par = {self, weight};
        if (bias) par.push_back(bias);
        return make_with_grad(std::move(out),
            [saved_x, saved_w, x_shape, w_shape, stride, padding, dilation, groups, has_bias](const Tensor& g) {
                Tensor dx = g.conv_transpose1d(saved_w, nullptr, stride, padding, 0, dilation, groups);
                int Ci_g = w_shape[1], K = w_shape[2];
                Tensor dw = Tensor::conv1d_backward_weight(saved_x, g, Ci_g, K, stride, padding, dilation, groups);
                if (has_bias) {
                    Tensor db = g.sum(0, false).sum(-1, false);
                    return std::vector<Tensor>{dx, dw, db};
                }
                return std::vector<Tensor>{dx, dw};
            }, par);
    }

    // --- ConvTranspose1d ---
    GradPtr conv_transpose1d_(GradPtr weight, GradPtr bias,
                               int stride, int padding, int output_padding, int dilation, int groups) {
        const Tensor* b_ptr = bias ? &bias->data : nullptr;
        Tensor out = data.conv_transpose1d(weight->data, b_ptr, stride, padding, output_padding, dilation, groups);
        bool need = requires_grad || weight->requires_grad || (bias && bias->requires_grad);
        if (!need) return make(std::move(out));
        Tensor saved_x = data, saved_w = weight->data;
        Shape x_shape = data.shape(), w_shape = weight->data.shape();
        bool has_bias = (bias != nullptr);
        auto self = shared_from_this();
        std::vector<GradPtr> par = {self, weight};
        if (bias) par.push_back(bias);
        return make_with_grad(std::move(out),
            [saved_x, saved_w, x_shape, w_shape, stride, padding, dilation, groups, has_bias](const Tensor& g) {
                Tensor dx = g.conv1d(saved_w, nullptr, stride, padding, dilation, groups);
                int Co_g = w_shape[1], K = w_shape[2];
                Tensor dw = Tensor::conv_transpose1d_backward_weight(saved_x, g, Co_g, K, stride, padding, dilation, groups);
                if (has_bias) {
                    Tensor db = g.sum(0, false).sum(-1, false);
                    return std::vector<Tensor>{dx, dw, db};
                }
                return std::vector<Tensor>{dx, dw};
            }, par);
    }

    // --- ConvTranspose2d ---
    GradPtr conv_transpose2d_(GradPtr weight, GradPtr bias,
                               int sH, int sW, int pH, int pW, int opH, int opW,
                               int dH, int dW, int groups) {
        const Tensor* b_ptr = bias ? &bias->data : nullptr;
        Tensor out = data.conv_transpose2d(weight->data, b_ptr, sH, sW, pH, pW, opH, opW, dH, dW, groups);
        bool need = requires_grad || weight->requires_grad || (bias && bias->requires_grad);
        if (!need) return make(std::move(out));
        Tensor saved_x = data, saved_w = weight->data;
        Shape x_shape = data.shape(), w_shape = weight->data.shape();
        bool has_bias = (bias != nullptr);
        auto self = shared_from_this();
        std::vector<GradPtr> par = {self, weight};
        if (bias) par.push_back(bias);
        return make_with_grad(std::move(out),
            [saved_x, saved_w, x_shape, w_shape, sH, sW, pH, pW, dH, dW, groups, has_bias](const Tensor& g) {
                Tensor dx = g.conv2d(saved_w, nullptr, sH, sW, pH, pW, dH, dW, groups);
                int Co_g = w_shape[1], kH = w_shape[2], kW = w_shape[3];
                Tensor dw = Tensor::conv_transpose2d_backward_weight(saved_x, g, Co_g, kH, kW, sH, sW, pH, pW, dH, dW, groups);
                if (has_bias) {
                    Tensor db = g.sum(0, false).sum(-1, false).sum(-1, false);
                    return std::vector<Tensor>{dx, dw, db};
                }
                return std::vector<Tensor>{dx, dw};
            }, par);
    }

    // --- Embedding ---
    GradPtr embedding_(const Tensor& indices) {
        Tensor out = data.embedding(indices);
        if (!requires_grad) return make(std::move(out));
        int vocab = data.shape()[0];
        auto idx = std::make_shared<Tensor>(indices);
        return make_with_grad(std::move(out),
            [idx, vocab](const Tensor& g) {
                return std::vector<Tensor>{g.scatter_add(g, *idx, vocab)};
            }, {shared_from_this()});
    }

    // --- Flip ---
    GradPtr flip_(int dim) {
        Tensor out = data.flip(dim);
        if (!requires_grad) return make(std::move(out));
        return make_with_grad(std::move(out),
            [dim](const Tensor& g) { return std::vector<Tensor>{g.flip(dim)}; },
            {shared_from_this()});
    }

    // --- Pad ---
    GradPtr pad_(const std::vector<int>& padding, int mode = 0, float value = 0.0f) {
        Tensor out = data.pad(padding, mode, value);
        if (!requires_grad) return make(std::move(out));
        Shape orig = data.shape();
        auto pad_copy = padding;
        return make_with_grad(std::move(out),
            [orig, pad_copy](const Tensor& g) {
                Tensor dx = g;
                int nd = (int)orig.size();
                for (int i = 0; i < (int)pad_copy.size(); i += 2) {
                    int dim = nd - 1 - i / 2;
                    int left = pad_copy[i];
                    int orig_size = orig[dim];
                    dx = dx.slice(dim, left, left + orig_size);
                }
                return std::vector<Tensor>{dx.contiguous()};
            }, {shared_from_this()});
    }

    // --- Interpolate ---
    GradPtr interpolate_(int target, int mode, bool align = false) {
        Tensor out = data.interpolate(target, mode, align);
        if (!requires_grad) return make(std::move(out));
        int in_len = data.shape().back();
        return make_with_grad(std::move(out),
            [in_len, mode, align](const Tensor& g) {
                return std::vector<Tensor>{g.interp1d_backward(in_len, mode, align)};
            }, {shared_from_this()});
    }

    // --- Cumsum ---
    GradPtr cumsum_(int dim) {
        Tensor out = data.cumsum(dim);
        if (!requires_grad) return make(std::move(out));
        return make_with_grad(std::move(out),
            [dim](const Tensor& g) {
                return std::vector<Tensor>{g.flip(dim).cumsum(dim).flip(dim)};
            }, {shared_from_this()});
    }

    // --- Unsqueeze ---
    GradPtr unsqueeze_(int dim) {
        Tensor out = data.unsqueeze(dim);
        if (!requires_grad) return make(std::move(out));
        Shape orig = data.shape();
        return make_with_grad(std::move(out),
            [orig](const Tensor& g) { return std::vector<Tensor>{g.reshape(orig)}; },
            {shared_from_this()});
    }

    // --- Flatten ---
    GradPtr flatten_(int start_dim = 0, int end_dim = -1) {
        Tensor out = data.flatten(start_dim, end_dim);
        if (!requires_grad) return make(std::move(out));
        Shape orig = data.shape();
        return make_with_grad(std::move(out),
            [orig](const Tensor& g) { return std::vector<Tensor>{g.reshape(orig)}; },
            {shared_from_this()});
    }

    // --- Contiguous ---
    GradPtr contiguous_() {
        Tensor out = data.contiguous();
        if (!requires_grad) return make(std::move(out));
        return make_with_grad(std::move(out),
            [](const Tensor& g) { return std::vector<Tensor>{g}; },
            {shared_from_this()});
    }

    // --- Clone ---
    GradPtr clone_() {
        Tensor out = data.clone();
        if (!requires_grad) return make(std::move(out));
        return make_with_grad(std::move(out),
            [](const Tensor& g) { return std::vector<Tensor>{g}; },
            {shared_from_this()});
    }

    // --- Max reduce ---
    GradPtr max_(int dim, bool keepdim = false) {
        Tensor out = data.max(dim, keepdim);
        if (!requires_grad) return make(std::move(out));
        Tensor sx = data;
        Tensor sout = out;
        return make_with_grad(std::move(out),
            [sx, sout, dim, keepdim](const Tensor& g) {
                Tensor g_expand = g;
                Tensor o_expand = sout;
                if (!keepdim) {
                    Shape s = sx.shape(); s[dim] = 1;
                    g_expand = g.reshape(s);
                    o_expand = sout.reshape(s);
                }
                Tensor mask = sx.eq(o_expand.add(Tensor::full(sx.shape(), 0.f)));
                return std::vector<Tensor>{g_expand.mul(mask)};
            }, {shared_from_this()});
    }

    // --- Min reduce ---
    GradPtr min_(int dim, bool keepdim = false) {
        Tensor out = data.min(dim, keepdim);
        if (!requires_grad) return make(std::move(out));
        Tensor sx = data;
        Tensor sout = out;
        return make_with_grad(std::move(out),
            [sx, sout, dim, keepdim](const Tensor& g) {
                Tensor g_expand = g;
                Tensor o_expand = sout;
                if (!keepdim) {
                    Shape s = sx.shape(); s[dim] = 1;
                    g_expand = g.reshape(s);
                    o_expand = sout.reshape(s);
                }
                Tensor mask = sx.eq(o_expand.add(Tensor::full(sx.shape(), 0.f)));
                return std::vector<Tensor>{g_expand.mul(mask)};
            }, {shared_from_this()});
    }

    // --- Static cat ---
    static GradPtr cat_(const std::vector<GradPtr>& tensors, int dim) {
        std::vector<Tensor> data_vec;
        for (auto& t : tensors) data_vec.push_back(t->data);
        Tensor out = Tensor::cat(data_vec, dim);

        bool need = false;
        for (auto& t : tensors) if (t->requires_grad) { need = true; break; }
        if (!need) return make(std::move(out));

        std::vector<int> sizes;
        for (auto& t : tensors) {
            int d = dim < 0 ? dim + t->ndim() : dim;
            sizes.push_back(t->data.shape()[d]);
        }

        std::vector<GradPtr> parents(tensors.begin(), tensors.end());
        return make_with_grad(std::move(out),
            [sizes, dim](const Tensor& g) {
                std::vector<Tensor> grads;
                int offset = 0;
                int d = dim < 0 ? dim + g.ndim() : dim;
                for (int s : sizes) {
                    grads.push_back(g.slice(d, offset, offset + s).contiguous());
                    offset += s;
                }
                return grads;
            }, parents);
    }

private:
    static void topo_sort(GradTensor* node, std::vector<GradTensor*>& order,
                          std::unordered_set<GradTensor*>& visited) {
        if (!node || visited.count(node) || !node->requires_grad) return;
        visited.insert(node);
        for (auto& p : node->parents)
            if (p) topo_sort(p.get(), order, visited);
        order.push_back(node);
    }
};

// ==================== CompiledGraph ====================
// Declarative graph: build op sequence once, replay forward+backward in C++ with 0 N-API calls.

enum class OpType : uint8_t {
    // Unary
    RELU, EXP, LOG, SQRT, SQUARE, NEG, ABS, SIGMOID, TANH, SILU, GELU,
    SOFTPLUS, RECIPROCAL, SIGN, LOG1P, SIN, COS,
    // Parameterized unary
    POW_SCALAR, MUL_SCALAR, ADD_SCALAR,
    // Binary
    ADD, SUB, MUL, DIV, POW,
    // Matmul
    MATMUL,
    // Reduce
    SUM, MEAN,
    // View
    TRANSPOSE, TRANSPOSE2, RESHAPE,
    // CNN
    CONV2D, MAX_POOL2D, AVG_POOL2D, BATCH_NORM2D, FLATTEN,
    // Misc
    CAT, SLICE, UNSQUEEZE, SQUEEZE,
    // Loop
    LOOP_BEGIN, LOOP_SLICE, LOOP_END,
    // Fused RNN (cuDNN)
    RNN_CUDNN,
};

struct TracedOp {
    OpType type;
    int16_t out, in0, in1, in2;   // in2 for conv bias / bn weight
    int16_t in3;                   // bn bias
    float fparam;
    int iparams[8];                // flexible int params (stride/pad/etc + shape)
};

class CompiledGraph {
    std::vector<TracedOp> ops_;
    int num_slots_ = 0;
    int input_slot_ = -1, target_slot_ = -1, output_slot_ = -1;
    std::vector<int> param_slots_;
    std::vector<int> buffer_slots_;
    std::vector<std::pair<int,int>> carry_pairs_;  // {from_slot, to_slot} at end of each iteration
    int loop_iter_ = 0;  // current loop iteration index

    // Tape cache (built once, reused across run_tape calls)
    struct TapeEntry { OpType type; int16_t out, in0, in1, in2, in3; float fparam; int iparams[8]; };
    std::vector<TapeEntry> tape_cache_;
    bool tape_built_ = false;
    // Pre-allocated buffers for run_tape (avoid per-call allocation)
    std::vector<Tensor> S_, G_;
    std::vector<bool> has_grad_;
    std::vector<Tensor> saved_;
    bool slots_allocated_ = false;

    // === Static Memory + CUDA Graph ===
    struct SlotMeta { Shape shape; int ndim; int total; size_t bytes; size_t offset; };
    std::vector<SlotMeta> fwd_meta_;   // forward slot metadata
    std::vector<SlotMeta> grad_meta_;  // grad slot metadata
    std::vector<SlotMeta> save_meta_;  // saved tensor metadata
    
    void* arena_ = nullptr;
    size_t arena_fwd_end_ = 0;    // end of forward region
    size_t arena_grad_end_ = 0;   // end of grad region  
    size_t arena_save_end_ = 0;   // end of saved region
    size_t arena_size_ = 0;
    
    // param device pointers (baked during profile)
    std::vector<float*> param_ptrs_;
    
    // TODO: CUDA Graph (requires all ops to use configurable stream)
    
    enum RunPhase { PHASE_PROFILE, PHASE_ARENA } phase_ = PHASE_PROFILE;

    // RNN_CUDNN op descriptor
    struct RNNCudnnOp {
        int x_slot, hx_slot, cx_slot;  // inputs (-1 = null)
        int y_slot, hy_slot, cy_slot;  // outputs (-1 = null)
        std::vector<int> wih_slots, whh_slots, bih_slots, bhh_slots; // weight slots
        int mode, hidden_size, num_layers, bidirectional;
    };
    std::vector<RNNCudnnOp> rnn_ops_;

public:
    int alloc() { return num_slots_++; }

    int input()  { input_slot_  = alloc(); return input_slot_;  }
    int target() { target_slot_ = alloc(); return target_slot_; }
    int param() {
        int s = alloc();
        param_slots_.push_back(s);
        return s;
    }
    // Buffer: non-trainable tensor (e.g. BN running stats)
    int buffer() {
        int s = alloc();
        buffer_slots_.push_back(s);
        return s;
    }
    void set_output(int s) { output_slot_ = s; }

    // Op builders — return output slot
    int op1(OpType t, int a, float fp = 0, int ip0 = 0, int ip1 = 0) {
        int o = alloc();
        TracedOp op{t, (int16_t)o, (int16_t)a, -1, -1, -1, fp, {}};
        op.iparams[0] = ip0; op.iparams[1] = ip1;
        ops_.push_back(op);
        return o;
    }
    int op2(OpType t, int a, int b) {
        int o = alloc();
        TracedOp op{t, (int16_t)o, (int16_t)a, (int16_t)b, -1, -1, 0, {}};
        ops_.push_back(op);
        return o;
    }
    // Conv2d: input, weight, bias(-1 if none), sH, sW, pH, pW, dH, dW, groups
    int conv2d(int input, int weight, int bias, int sH, int sW, int pH, int pW, int dH, int dW, int groups) {
        int o = alloc();
        TracedOp op{OpType::CONV2D, (int16_t)o, (int16_t)input, (int16_t)weight, (int16_t)bias, -1, 0, {}};
        op.iparams[0]=sH; op.iparams[1]=sW; op.iparams[2]=pH; op.iparams[3]=pW;
        op.iparams[4]=dH; op.iparams[5]=dW; op.iparams[6]=groups;
        ops_.push_back(op);
        return o;
    }
    // MaxPool2d: kH, kW, sH, sW, pH, pW
    int max_pool2d(int input, int kH, int kW, int sH, int sW, int pH, int pW) {
        int o = alloc();
        TracedOp op{OpType::MAX_POOL2D, (int16_t)o, (int16_t)input, -1, -1, -1, 0, {}};
        op.iparams[0]=kH; op.iparams[1]=kW; op.iparams[2]=sH; op.iparams[3]=sW;
        op.iparams[4]=pH; op.iparams[5]=pW;
        ops_.push_back(op);
        return o;
    }
    // AvgPool2d: kH, kW, sH, sW, pH, pW, count_include_pad
    int avg_pool2d(int input, int kH, int kW, int sH, int sW, int pH, int pW, bool cip = true) {
        int o = alloc();
        TracedOp op{OpType::AVG_POOL2D, (int16_t)o, (int16_t)input, -1, -1, -1, 0, {}};
        op.iparams[0]=kH; op.iparams[1]=kW; op.iparams[2]=sH; op.iparams[3]=sW;
        op.iparams[4]=pH; op.iparams[5]=pW; op.iparams[6]=cip?1:0;
        ops_.push_back(op);
        return o;
    }
    // BatchNorm2d: input, weight(gamma), bias(beta), running_mean, running_var
    int batch_norm2d(int input, int weight, int bias, int rmean, int rvar, float eps = 1e-5f) {
        int o = alloc();
        TracedOp op{OpType::BATCH_NORM2D, (int16_t)o, (int16_t)input, (int16_t)weight, (int16_t)bias, (int16_t)rmean, eps, {}};
        op.iparams[0] = rvar;  // store running_var slot in iparams[0]
        ops_.push_back(op);
        return o;
    }
    // --- Loop support ---
    // loop_begin(num_iters): marks start of loop body
    void loop_begin(int num_iters) {
        TracedOp op{OpType::LOOP_BEGIN, -1, -1, -1, -1, -1, 0, {}};
        op.iparams[0] = num_iters;
        ops_.push_back(op);
    }
    // loop_slice(seq, dim): slice seq at current iteration along dim, then squeeze
    int loop_slice(int seq_slot, int dim) {
        int o = alloc();
        TracedOp op{OpType::LOOP_SLICE, (int16_t)o, (int16_t)seq_slot, -1, -1, -1, 0, {}};
        op.iparams[0] = dim;
        ops_.push_back(op);
        return o;
    }
    // loop_carry(from, to): at end of each iteration, slots[to] = slots[from]
    void loop_carry(int from, int to) {
        carry_pairs_.push_back({from, to});
    }
    // loop_end(): marks end of loop body
    void loop_end() {
        ops_.push_back({OpType::LOOP_END, -1, -1, -1, -1, -1, 0, {}});
    }

    // Flatten from start_dim (default 1) — [B, C, H, W] -> [B, C*H*W]
    int flatten(int input, int start_dim = 1) {
        int o = alloc();
        TracedOp op{OpType::FLATTEN, (int16_t)o, (int16_t)input, -1, -1, -1, 0, {}};
        op.iparams[0] = start_dim;
        ops_.push_back(op);
        return o;
    }
    // Reshape: input, shape (up to 6 dims, -1 terminated)
    int reshape(int input, const std::vector<int>& shape) {
        int o = alloc();
        TracedOp op{OpType::RESHAPE, (int16_t)o, (int16_t)input, -1, -1, -1, 0, {}};
        for (int i = 0; i < (int)shape.size() && i < 8; i++) op.iparams[i] = shape[i];
        if ((int)shape.size() < 8) op.iparams[shape.size()] = INT_MIN; // sentinel
        ops_.push_back(op);
        return o;
    }

    // Cat: concatenate multiple slots along dim. slots stored in iparams, count in fparam
    int cat(const std::vector<int>& inputs, int dim) {
        int o = alloc();
        TracedOp op{OpType::CAT, (int16_t)o, -1, -1, -1, -1, (float)dim, {}};
        for (int i = 0; i < (int)inputs.size() && i < 8; i++) op.iparams[i] = inputs[i];
        // Store count in in0
        op.in0 = (int16_t)inputs.size();
        ops_.push_back(op);
        return o;
    }
    // Slice: input[dim, start:end]
    int slice(int input, int dim, int start, int end) {
        int o = alloc();
        TracedOp op{OpType::SLICE, (int16_t)o, (int16_t)input, -1, -1, -1, 0, {}};
        op.iparams[0] = dim; op.iparams[1] = start; op.iparams[2] = end;
        ops_.push_back(op);
        return o;
    }
    int unsqueeze(int input, int dim) {
        return op1(OpType::UNSQUEEZE, input, 0, dim);
    }
    int squeeze(int input, int dim) {
        return op1(OpType::SQUEEZE, input, 0, dim);
    }

    // RNN_CUDNN: returns {y_slot, hy_slot, cy_slot}
    std::vector<int> rnn_cudnn(int x_slot, int hx_slot, int cx_slot,
                               const std::vector<int>& wih, const std::vector<int>& whh,
                               const std::vector<int>& bih, const std::vector<int>& bhh,
                               int mode, int hidden_size, int num_layers, int bidirectional) {
        int y = alloc(), hy = alloc(), cy = alloc();
        int rnn_idx = (int)rnn_ops_.size();
        rnn_ops_.push_back({x_slot, hx_slot, cx_slot, y, hy, cy,
                            wih, whh, bih, bhh, mode, hidden_size, num_layers, bidirectional});
        // Store a TracedOp placeholder — iparams[0] = rnn_ops_ index
        TracedOp op{OpType::RNN_CUDNN, (int16_t)y, (int16_t)x_slot, -1, -1, -1, 0, {}};
        op.iparams[0] = rnn_idx;
        ops_.push_back(op);
        return {y, hy, cy};
    }

    int num_params() const { return (int)param_slots_.size(); }

    int num_buffers() const { return (int)buffer_slots_.size(); }

    // Execute forward + backward, write gradients back into param tensors
    void run(const Tensor& input, const Tensor& target,
             std::vector<Tensor>& params,       // in: param data, modified in-place by adam
             std::vector<Tensor>& m_states,      // adam m
             std::vector<Tensor>& v_states,      // adam v
             float lr, float beta1, float beta2, float eps, float bc1, float bc2, float wd,
             std::vector<Tensor>* buffers = nullptr) {

        int np = (int)params.size();
        std::vector<GradPtr> slots(num_slots_);

        // Set up slots
        slots[input_slot_] = GradTensor::make(input, false);
        if (target_slot_ >= 0)
            slots[target_slot_] = GradTensor::make(target, false);
        for (int i = 0; i < np; i++)
            slots[param_slots_[i]] = GradTensor::make(params[i], true);
        if (buffers) {
            for (int i = 0; i < (int)buffer_slots_.size() && i < (int)buffers->size(); i++)
                slots[buffer_slots_[i]] = GradTensor::make((*buffers)[i], false);
        }

        // Forward (with loop support)
        for (int ip = 0; ip < (int)ops_.size(); ip++) {
            auto& op = ops_[ip];
            auto& a = slots[op.in0];
            switch (op.type) {
                // Unary
                case OpType::RELU:       slots[op.out] = a->relu_(); break;
                case OpType::EXP:        slots[op.out] = a->exp_(); break;
                case OpType::LOG:        slots[op.out] = a->log_(); break;
                case OpType::SQRT:       slots[op.out] = a->sqrt_(); break;
                case OpType::SQUARE:     slots[op.out] = a->square_(); break;
                case OpType::NEG:        slots[op.out] = a->neg_(); break;
                case OpType::ABS:        slots[op.out] = a->abs_(); break;
                case OpType::SIGMOID:    slots[op.out] = a->sigmoid_(); break;
                case OpType::TANH:       slots[op.out] = a->tanh_(); break;
                case OpType::SILU:       slots[op.out] = a->silu_(); break;
                case OpType::GELU:       slots[op.out] = a->gelu_(); break;
                case OpType::SOFTPLUS:   slots[op.out] = a->softplus_(); break;
                case OpType::RECIPROCAL: slots[op.out] = a->reciprocal_(); break;
                case OpType::SIGN:       slots[op.out] = a->sign_(); break;
                case OpType::LOG1P:      slots[op.out] = a->log1p_(); break;
                case OpType::SIN:        slots[op.out] = a->sin_(); break;
                case OpType::COS:        slots[op.out] = a->cos_(); break;
                // Parameterized unary
                case OpType::POW_SCALAR: slots[op.out] = a->pow_scalar_(op.fparam); break;
                case OpType::MUL_SCALAR: slots[op.out] = a->mul_scalar_(op.fparam); break;
                case OpType::ADD_SCALAR: slots[op.out] = a->add_scalar_(op.fparam); break;
                // Binary
                case OpType::ADD:    slots[op.out] = a->add_(slots[op.in1]); break;
                case OpType::SUB:    slots[op.out] = a->sub_(slots[op.in1]); break;
                case OpType::MUL:    slots[op.out] = a->mul_(slots[op.in1]); break;
                case OpType::DIV:    slots[op.out] = a->div_(slots[op.in1]); break;
                case OpType::POW:    slots[op.out] = a->pow_(slots[op.in1]); break;
                // Matmul
                case OpType::MATMUL: slots[op.out] = a->matmul_(slots[op.in1]); break;
                // Reduce
                case OpType::SUM:    slots[op.out] = a->sum_(op.iparams[0], op.iparams[1]); break;
                case OpType::MEAN:   slots[op.out] = a->mean_(op.iparams[0], op.iparams[1]); break;
                // View
                case OpType::TRANSPOSE:  slots[op.out] = a->transpose_(); break;
                case OpType::TRANSPOSE2: slots[op.out] = a->transpose_(op.iparams[0], op.iparams[1]); break;
                case OpType::RESHAPE: {
                    Shape s;
                    for (int i = 0; i < 8 && op.iparams[i] != INT_MIN; i++) s.push_back(op.iparams[i]);
                    // Handle -1 dim: infer from total size
                    int total = a->data.size(), known = 1, neg_idx = -1;
                    for (int i = 0; i < (int)s.size(); i++) {
                        if (s[i] == -1) neg_idx = i;
                        else known *= s[i];
                    }
                    if (neg_idx >= 0) s[neg_idx] = total / known;
                    slots[op.out] = a->reshape_(s);
                    break;
                }
                case OpType::FLATTEN: {
                    int start = op.iparams[0];
                    Shape orig = a->data.shape();
                    Shape ns;
                    for (int i = 0; i < start; i++) ns.push_back(orig[i]);
                    int flat = 1;
                    for (int i = start; i < (int)orig.size(); i++) flat *= orig[i];
                    ns.push_back(flat);
                    slots[op.out] = a->reshape_(ns);
                    break;
                }
                // Misc
                case OpType::CAT: {
                    int count = op.in0;
                    std::vector<GradPtr> tensors;
                    for (int i = 0; i < count; i++) tensors.push_back(slots[op.iparams[i]]);
                    int dim = (int)op.fparam;
                    slots[op.out] = GradTensor::cat_(tensors, dim);
                    break;
                }
                case OpType::SLICE:
                    slots[op.out] = a->slice_(op.iparams[0], op.iparams[1], op.iparams[2]);
                    break;
                case OpType::UNSQUEEZE:
                    slots[op.out] = a->unsqueeze_(op.iparams[0]);
                    break;
                case OpType::SQUEEZE:
                    slots[op.out] = a->squeeze_(op.iparams[0]);
                    break;
                // CNN
                case OpType::CONV2D: {
                    auto bias = op.in2 >= 0 ? slots[op.in2] : nullptr;
                    slots[op.out] = a->conv2d_(slots[op.in1], bias,
                        op.iparams[0], op.iparams[1], op.iparams[2], op.iparams[3],
                        op.iparams[4], op.iparams[5], op.iparams[6]);
                    break;
                }
                case OpType::MAX_POOL2D:
                    slots[op.out] = a->max_pool2d_(op.iparams[0], op.iparams[1],
                        op.iparams[2], op.iparams[3], op.iparams[4], op.iparams[5]);
                    break;
                case OpType::AVG_POOL2D:
                    slots[op.out] = a->avg_pool2d_(op.iparams[0], op.iparams[1],
                        op.iparams[2], op.iparams[3], op.iparams[4], op.iparams[5], op.iparams[6] != 0);
                    break;
                case OpType::BATCH_NORM2D: {
                    // Inference-only BN: (x - mean) / sqrt(var + eps) * weight + bias
                    auto& weight = slots[op.in2];   // gamma
                    auto& bias = slots[op.in1];     // beta — reuse in1 for bias since input is in0
                    // Actually: in0=input, in1=weight(gamma), in2=bias(beta), in3=rmean, iparams[0]=rvar slot
                    auto& x = a;                    // in0
                    auto& gamma = slots[op.in1];    // weight
                    auto& beta = slots[op.in2];     // bias
                    auto& rmean = slots[op.in3];    // running mean
                    auto& rvar = slots[op.iparams[0]]; // running var
                    float eps = op.fparam;
                    // BN: out = gamma * (x - mean) / sqrt(var + eps) + beta
                    // Broadcast: mean/var/gamma/beta are [C], x is [B,C,H,W]
                    // Reshape to [1,C,1,1] for broadcast
                    int C = gamma->data.shape()[0];
                    Shape bc = {1, C, 1, 1};
                    auto m = rmean->reshape_(bc);
                    auto v = rvar->reshape_(bc);
                    auto g = gamma->reshape_(bc);
                    auto b = beta->reshape_(bc);
                    auto centered = x->sub_(m);
                    auto vstd = v->add_scalar_(eps)->sqrt_()->reciprocal_();
                    slots[op.out] = centered->mul_(vstd)->mul_(g)->add_(b);
                    break;
                }
                // Loop
                case OpType::LOOP_BEGIN: {
                    int num_iters = op.iparams[0];
                    int loop_start = ip + 1;
                    // Find matching LOOP_END
                    int loop_end_ip = loop_start;
                    while (loop_end_ip < (int)ops_.size() && ops_[loop_end_ip].type != OpType::LOOP_END)
                        loop_end_ip++;
                    // Execute loop body num_iters times
                    for (int iter = 0; iter < num_iters; iter++) {
                        loop_iter_ = iter;
                        for (int j = loop_start; j < loop_end_ip; j++) {
                            auto& lop = ops_[j];
                            auto& la = slots[lop.in0];
                            switch (lop.type) {
                                case OpType::LOOP_SLICE: {
                                    int dim = lop.iparams[0];
                                    slots[lop.out] = la->slice_(dim, iter, iter + 1)->squeeze_(dim);
                                    break;
                                }
                                // All other ops: same as outer switch
                                case OpType::RELU: slots[lop.out] = la->relu_(); break;
                                case OpType::SIGMOID: slots[lop.out] = la->sigmoid_(); break;
                                case OpType::TANH: slots[lop.out] = la->tanh_(); break;
                                case OpType::ADD: slots[lop.out] = la->add_(slots[lop.in1]); break;
                                case OpType::SUB: slots[lop.out] = la->sub_(slots[lop.in1]); break;
                                case OpType::MUL: slots[lop.out] = la->mul_(slots[lop.in1]); break;
                                case OpType::MATMUL: slots[lop.out] = la->matmul_(slots[lop.in1]); break;
                                case OpType::TRANSPOSE: slots[lop.out] = la->transpose_(); break;
                                case OpType::TRANSPOSE2: slots[lop.out] = la->transpose_(lop.iparams[0], lop.iparams[1]); break;
                                case OpType::MUL_SCALAR: slots[lop.out] = la->mul_scalar_(lop.fparam); break;
                                case OpType::ADD_SCALAR: slots[lop.out] = la->add_scalar_(lop.fparam); break;
                                case OpType::SUM: slots[lop.out] = la->sum_(lop.iparams[0], lop.iparams[1]); break;
                                case OpType::MEAN: slots[lop.out] = la->mean_(lop.iparams[0], lop.iparams[1]); break;
                                case OpType::RESHAPE: {
                                    Shape s;
                                    for (int i = 0; i < 8 && lop.iparams[i] != INT_MIN; i++) s.push_back(lop.iparams[i]);
                                    int total = la->data.size(), known = 1, neg_idx = -1;
                                    for (int i = 0; i < (int)s.size(); i++) {
                                        if (s[i] == -1) neg_idx = i; else known *= s[i];
                                    }
                                    if (neg_idx >= 0) s[neg_idx] = total / known;
                                    slots[lop.out] = la->reshape_(s);
                                    break;
                                }
                                case OpType::FLATTEN: {
                                    int start = lop.iparams[0];
                                    Shape orig = la->data.shape();
                                    Shape ns;
                                    for (int i = 0; i < start; i++) ns.push_back(orig[i]);
                                    int flat = 1;
                                    for (int i = start; i < (int)orig.size(); i++) flat *= orig[i];
                                    ns.push_back(flat);
                                    slots[lop.out] = la->reshape_(ns);
                                    break;
                                }
                                case OpType::CONV2D: {
                                    auto bias = lop.in2 >= 0 ? slots[lop.in2] : nullptr;
                                    slots[lop.out] = la->conv2d_(slots[lop.in1], bias,
                                        lop.iparams[0], lop.iparams[1], lop.iparams[2], lop.iparams[3],
                                        lop.iparams[4], lop.iparams[5], lop.iparams[6]);
                                    break;
                                }
                                case OpType::CAT: {
                                    int count = lop.in0;
                                    std::vector<GradPtr> tensors;
                                    for (int i = 0; i < count; i++) tensors.push_back(slots[lop.iparams[i]]);
                                    int dim = (int)lop.fparam;
                                    slots[lop.out] = GradTensor::cat_(tensors, dim);
                                    break;
                                }
                                case OpType::SLICE:
                                    slots[lop.out] = la->slice_(lop.iparams[0], lop.iparams[1], lop.iparams[2]);
                                    break;
                                case OpType::UNSQUEEZE:
                                    slots[lop.out] = la->unsqueeze_(lop.iparams[0]);
                                    break;
                                case OpType::SQUEEZE:
                                    slots[lop.out] = la->squeeze_(lop.iparams[0]);
                                    break;
                                default: break;
                            }
                        }
                        // Apply carry: copy from→to slots
                        for (auto& [from, to] : carry_pairs_) {
                            slots[to] = slots[from];
                        }
                    }
                    ip = loop_end_ip; // skip past loop body
                    break;
                }
                case OpType::LOOP_SLICE: break; // handled inside LOOP_BEGIN
                case OpType::LOOP_END: break;   // handled inside LOOP_BEGIN
            }
        }

        // Backward
        GradTensor::backward(slots[output_slot_],
                             Tensor::full(slots[output_slot_]->shape(), 1.0f));

        // Adam step for each param (fused: no N-API per param)
        for (int i = 0; i < np; i++) {
            auto& s = slots[param_slots_[i]];
            if (!s->has_grad) continue;
            Tensor::adam_step(params[i], *s->grad_, m_states[i], v_states[i],
                              lr, beta1, beta2, eps, bc1, bc2, wd);
        }
    }

    ArenaAllocator arena_alloc_;

    // ==================== Tape-based forward/backward ====================
    // Zero GradTensor allocation — uses raw Tensor ops + analytical gradients.
    // Phase 1 (first call): profile sizes, allocate arena
    // Phase 2+: bump-allocate from arena (O(1) alloc, no mutex, no free list)
    void run_tape(const Tensor& input, const Tensor& target,
                  std::vector<Tensor>& params,
                  std::vector<Tensor>& m_states, std::vector<Tensor>& v_states,
                  float lr, float beta1, float beta2, float eps,
                  float bc1, float bc2, float wd,
                  std::vector<Tensor>* buffers = nullptr) {

        int np = (int)params.size();
        if (!slots_allocated_) {
            Tensor dummy({1});
            S_.assign(num_slots_, dummy);
            G_.assign(num_slots_, dummy);
            has_grad_.assign(num_slots_, false);
            slots_allocated_ = true;
        }
        auto& S = S_; auto& G = G_; auto& has_grad = has_grad_;
        std::fill(has_grad.begin(), has_grad.end(), false);
        auto& saved = saved_;

        // Arena mode: bump-allocate all intermediates, reset each call
        arena_alloc_.reset();
        active_arena = &arena_alloc_;

        // Accumulate gradient into slot
        auto accum = [&](int slot, Tensor g) {
            if (has_grad[slot]) {
                S[slot]; // ensure exists
                G[slot] = G[slot].add(g);
            } else {
                G[slot] = std::move(g);
                has_grad[slot] = true;
            }
        };

        // Setup
        S[input_slot_] = input;
        if (target_slot_ >= 0) S[target_slot_] = target;
        for (int i = 0; i < np; i++) S[param_slots_[i]] = params[i];
        if (buffers) {
            for (int i = 0; i < (int)buffer_slots_.size() && i < (int)buffers->size(); i++)
                S[buffer_slots_[i]] = (*buffers)[i];
        }

        // ---- Build tape once (cached) ----
        if (!tape_built_) {
            tape_cache_.clear();
            auto push_op = [](std::vector<TapeEntry>& tape, const TracedOp& op) {
                TapeEntry e;
                e.type = op.type; e.out = op.out; e.in0 = op.in0; e.in1 = op.in1;
                e.in2 = op.in2; e.in3 = op.in3; e.fparam = op.fparam;
                std::memcpy(e.iparams, op.iparams, sizeof(e.iparams));
                tape.push_back(e);
            };
            for (int ip = 0; ip < (int)ops_.size(); ip++) {
                auto& op = ops_[ip];
                if (op.type == OpType::LOOP_BEGIN) {
                    int num_iters = op.iparams[0];
                    int loop_start = ip + 1;
                    int loop_end_ip = loop_start;
                    while (loop_end_ip < (int)ops_.size() && ops_[loop_end_ip].type != OpType::LOOP_END)
                        loop_end_ip++;
                    for (int iter = 0; iter < num_iters; iter++) {
                        for (int j = loop_start; j < loop_end_ip; j++) {
                            auto& lop = ops_[j];
                            if (lop.type == OpType::LOOP_SLICE) {
                                TapeEntry e;
                                e.type = OpType::LOOP_SLICE; e.out = lop.out; e.in0 = lop.in0;
                                e.in1 = -1; e.in2 = -1; e.in3 = -1; e.fparam = 0;
                                std::memset(e.iparams, 0, sizeof(e.iparams));
                                e.iparams[0] = lop.iparams[0];
                                e.iparams[1] = iter;
                                tape_cache_.push_back(e);
                            } else {
                                push_op(tape_cache_, lop);
                            }
                        }
                        for (auto& [from, to] : carry_pairs_) {
                            TapeEntry e;
                            e.type = OpType::LOOP_END; e.out = (int16_t)to; e.in0 = (int16_t)from;
                            e.in1 = -1; e.in2 = -1; e.in3 = -1; e.fparam = 0;
                            std::memset(e.iparams, 0, sizeof(e.iparams));
                            tape_cache_.push_back(e);
                        }
                    }
                    ip = loop_end_ip;
                } else if (op.type == OpType::LOOP_SLICE || op.type == OpType::LOOP_END) {
                } else {
                    push_op(tape_cache_, op);
                }
            }
            tape_built_ = true;
        }
        auto& tape = tape_cache_;

        // Execute forward tape
        saved.clear();
        saved.reserve(tape.size()); // upper bound
        for (auto& e : tape) {
            auto& a = S[e.in0];
            switch (e.type) {
                case OpType::RELU:
                    S[e.out] = a.relu();
                    saved.push_back(a); // need input for backward
                    break;
                case OpType::EXP:
                    S[e.out] = a.exp();
                    saved.push_back(S[e.out]); // need output
                    break;
                case OpType::LOG:
                    S[e.out] = a.log();
                    saved.push_back(a);
                    break;
                case OpType::SQRT:
                    S[e.out] = a.sqrt();
                    saved.push_back(S[e.out]);
                    break;
                case OpType::SQUARE:
                    S[e.out] = a.square();
                    saved.push_back(a);
                    break;
                case OpType::NEG:
                    S[e.out] = a.neg();
                    break;
                case OpType::ABS:
                    S[e.out] = a.abs();
                    saved.push_back(a);
                    break;
                case OpType::SIGMOID:
                    S[e.out] = a.sigmoid();
                    saved.push_back(S[e.out]);
                    break;
                case OpType::TANH:
                    S[e.out] = a.tanh();
                    saved.push_back(S[e.out]);
                    break;
                case OpType::SILU:
                    S[e.out] = a.silu();
                    saved.push_back(a);
                    break;
                case OpType::GELU:
                    S[e.out] = a.gelu();
                    saved.push_back(a);
                    break;
                case OpType::SOFTPLUS:
                    S[e.out] = a.softplus();
                    saved.push_back(S[e.out]);
                    break;
                case OpType::RECIPROCAL:
                    S[e.out] = a.reciprocal();
                    saved.push_back(S[e.out]);
                    break;
                case OpType::SIGN:
                    S[e.out] = a.sign();
                    break;
                case OpType::LOG1P:
                    S[e.out] = a.log1p();
                    saved.push_back(a);
                    break;
                case OpType::SIN:
                    S[e.out] = a.sin();
                    saved.push_back(a);
                    break;
                case OpType::COS:
                    S[e.out] = a.cos();
                    saved.push_back(a);
                    break;
                case OpType::POW_SCALAR:
                    S[e.out] = a.pow_scalar(e.fparam);
                    saved.push_back(a);
                    break;
                case OpType::MUL_SCALAR:
                    S[e.out] = a.mul_scalar(e.fparam);
                    break;
                case OpType::ADD_SCALAR:
                    S[e.out] = a.add_scalar(e.fparam);
                    break;
                case OpType::ADD:
                    S[e.out] = a.add(S[e.in1]);
                    break;
                case OpType::SUB:
                    S[e.out] = a.sub(S[e.in1]);
                    break;
                case OpType::MUL:
                    S[e.out] = a.mul(S[e.in1]);
                    saved.push_back(a);
                    saved.push_back(S[e.in1]);
                    break;
                case OpType::DIV:
                    S[e.out] = a.div(S[e.in1]);
                    saved.push_back(a);
                    saved.push_back(S[e.in1]);
                    break;
                case OpType::POW:
                    S[e.out] = a.pow(S[e.in1]);
                    saved.push_back(a);
                    saved.push_back(S[e.in1]);
                    saved.push_back(S[e.out]);
                    break;
                case OpType::MATMUL:
                    S[e.out] = a.matmul(S[e.in1]);
                    saved.push_back(a);
                    saved.push_back(S[e.in1]);
                    break;
                case OpType::SUM: {
                    int dim = e.iparams[0], kd = e.iparams[1];
                    S[e.out] = a.sum(dim, kd);
                    break;
                }
                case OpType::MEAN: {
                    int dim = e.iparams[0], kd = e.iparams[1];
                    S[e.out] = a.mean(dim, kd);
                    break;
                }
                case OpType::TRANSPOSE:
                    S[e.out] = a.transpose();
                    break;
                case OpType::TRANSPOSE2:
                    S[e.out] = a.transpose(e.iparams[0], e.iparams[1]);
                    break;
                case OpType::RESHAPE: {
                    Shape s;
                    for (int i = 0; i < 8 && e.iparams[i] != INT_MIN; i++) s.push_back(e.iparams[i]);
                    int total = a.size(), known = 1, neg_idx = -1;
                    for (int i = 0; i < (int)s.size(); i++) {
                        if (s[i] == -1) neg_idx = i; else known *= s[i];
                    }
                    if (neg_idx >= 0) s[neg_idx] = total / known;
                    S[e.out] = a.reshape(s);
                    break;
                }
                case OpType::FLATTEN: {
                    int start = e.iparams[0];
                    Shape orig = a.shape();
                    Shape ns;
                    for (int i = 0; i < start; i++) ns.push_back(orig[i]);
                    int flat = 1;
                    for (int i = start; i < (int)orig.size(); i++) flat *= orig[i];
                    ns.push_back(flat);
                    S[e.out] = a.reshape(ns);
                    break;
                }
                case OpType::CONV2D: {
                    const Tensor* bias = e.in2 >= 0 ? &S[e.in2] : nullptr;
                    S[e.out] = a.conv2d(S[e.in1], bias,
                        e.iparams[0], e.iparams[1], e.iparams[2], e.iparams[3],
                        e.iparams[4], e.iparams[5], e.iparams[6]);
                    saved.push_back(a);
                    saved.push_back(S[e.in1]);
                    break;
                }
                case OpType::MAX_POOL2D: {
                    auto [out, indices] = a.max_pool2d(e.iparams[0], e.iparams[1],
                        e.iparams[2], e.iparams[3], e.iparams[4], e.iparams[5]);
                    S[e.out] = std::move(out);
                    saved.push_back(indices);
                    break;
                }
                case OpType::AVG_POOL2D: {
                    S[e.out] = a.avg_pool2d(e.iparams[0], e.iparams[1],
                        e.iparams[2], e.iparams[3], e.iparams[4], e.iparams[5], e.iparams[6]);
                    break;
                }
                case OpType::BATCH_NORM2D: {
                    // BN: (x - mean) / sqrt(var + eps) * gamma + beta
                    auto& rmean = S[e.in1]; auto& rvar = S[e.in2];
                    auto& gamma = S[e.in3]; auto& beta  = S[(int)e.fparam]; // fparam holds beta slot cast
                    // Actually, let's check how BN is stored...
                    // For now, just do forward like the GradTensor version
                    // TODO: tape BN backward
                    float bneps = *(float*)&e.iparams[0]; // eps stored in iparams
                    // This is complex, skip BN in tape mode for now
                    break;
                }
                case OpType::LOOP_SLICE: {
                    int dim = e.iparams[0], iter = e.iparams[1];
                    S[e.out] = a.slice(dim, iter, iter + 1).squeeze(dim);
                    break;
                }
                case OpType::LOOP_END: {
                    // CARRY: copy from→to
                    S[e.out] = S[e.in0];
                    break;
                }
                case OpType::RNN_CUDNN: {
                    auto& rop = rnn_ops_[e.iparams[0]];
                    auto& x = S[rop.x_slot];
                    const Tensor* hx = rop.hx_slot >= 0 ? &S[rop.hx_slot] : nullptr;
                    const Tensor* cx = rop.cx_slot >= 0 ? &S[rop.cx_slot] : nullptr;
                    std::vector<Tensor> wih, whh, bih, bhh;
                    for (int s : rop.wih_slots) wih.push_back(S[s]);
                    for (int s : rop.whh_slots) whh.push_back(S[s]);
                    for (int s : rop.bih_slots) bih.push_back(S[s]);
                    for (int s : rop.bhh_slots) bhh.push_back(S[s]);
                    auto result = Tensor::rnn_forward(x, hx, cx, wih, whh, bih, bhh,
                        rop.mode, rop.hidden_size, rop.num_layers, rop.bidirectional);
                    S[rop.y_slot] = std::move(result[0]);
                    S[rop.hy_slot] = std::move(result[1]);
                    if (result.size() > 2) S[rop.cy_slot] = std::move(result[2]);
                    // Save for backward: x, hx, y, weights
                    saved.push_back(x);
                    saved.push_back(hx ? *hx : Tensor({1}));
                    saved.push_back(cx ? *cx : Tensor({1}));
                    saved.push_back(S[rop.y_slot]);
                    break;
                }
                default: break;
            }
        }

        // ---- Backward ----
        has_grad[output_slot_] = true;
        G[output_slot_] = Tensor::full(S[output_slot_].shape(), 1.0f);

        int si = (int)saved.size(); // saved index, read backward

        for (int i = (int)tape.size() - 1; i >= 0; i--) {
            auto& e = tape[i];
            if (!has_grad[e.out]) continue;
            auto& g = G[e.out];

            switch (e.type) {
                case OpType::RELU: {
                    auto& sx = saved[--si];
                    accum(e.in0, g.mul(sx.gt(Tensor::full(sx.shape(), 0.f))));
                    break;
                }
                case OpType::EXP: {
                    auto& so = saved[--si];
                    accum(e.in0, g.mul(so));
                    break;
                }
                case OpType::LOG: {
                    auto& sx = saved[--si];
                    accum(e.in0, g.div(sx));
                    break;
                }
                case OpType::SQRT: {
                    auto& so = saved[--si];
                    accum(e.in0, g.div(so.mul_scalar(2.f)));
                    break;
                }
                case OpType::SQUARE: {
                    auto& sx = saved[--si];
                    accum(e.in0, g.mul(sx).mul_scalar(2.f));
                    break;
                }
                case OpType::NEG:
                    accum(e.in0, g.neg());
                    break;
                case OpType::ABS: {
                    auto& sx = saved[--si];
                    accum(e.in0, g.mul(sx.sign()));
                    break;
                }
                case OpType::SIGMOID: {
                    auto& so = saved[--si];
                    accum(e.in0, g.mul(so).mul(so.neg().add_scalar(1.f)));
                    break;
                }
                case OpType::TANH: {
                    auto& so = saved[--si];
                    accum(e.in0, g.mul(so.square().neg().add_scalar(1.f)));
                    break;
                }
                case OpType::SILU: {
                    auto& sx = saved[--si];
                    Tensor sig = sx.sigmoid();
                    accum(e.in0, g.mul(sig.add(sx.mul(sig).mul(sig.neg().add_scalar(1.f)))));
                    break;
                }
                case OpType::GELU: {
                    auto& sx = saved[--si];
                    Tensor sig = sx.mul_scalar(1.702f).sigmoid();
                    accum(e.in0, g.mul(sig.add(sx.mul_scalar(1.702f).mul(sig).mul(sig.neg().add_scalar(1.f)))));
                    break;
                }
                case OpType::SOFTPLUS: {
                    auto& so = saved[--si];
                    accum(e.in0, g.mul(so.neg().exp().neg().add_scalar(1.f)));
                    break;
                }
                case OpType::RECIPROCAL: {
                    auto& so = saved[--si];
                    accum(e.in0, g.neg().mul(so.square()));
                    break;
                }
                case OpType::SIGN:
                    // grad is zero
                    break;
                case OpType::LOG1P: {
                    auto& sx = saved[--si];
                    accum(e.in0, g.div(sx.add_scalar(1.f)));
                    break;
                }
                case OpType::SIN: {
                    auto& sx = saved[--si];
                    accum(e.in0, g.mul(sx.cos()));
                    break;
                }
                case OpType::COS: {
                    auto& sx = saved[--si];
                    accum(e.in0, g.mul(sx.sin().neg()));
                    break;
                }
                case OpType::POW_SCALAR: {
                    auto& sx = saved[--si];
                    float ee = e.fparam;
                    accum(e.in0, g.mul_scalar(ee).mul(sx.pow_scalar(ee - 1.f)));
                    break;
                }
                case OpType::MUL_SCALAR:
                    accum(e.in0, g.mul_scalar(e.fparam));
                    break;
                case OpType::ADD_SCALAR:
                    accum(e.in0, g);
                    break;
                case OpType::ADD: {
                    Shape sa = S[e.in0].shape(), sb = S[e.in1].shape();
                    accum(e.in0, GradTensor::unbroadcast(g, sa));
                    accum(e.in1, GradTensor::unbroadcast(g, sb));
                    break;
                }
                case OpType::SUB: {
                    Shape sa = S[e.in0].shape(), sb = S[e.in1].shape();
                    accum(e.in0, GradTensor::unbroadcast(g, sa));
                    accum(e.in1, GradTensor::unbroadcast(g.neg(), sb));
                    break;
                }
                case OpType::MUL: {
                    auto& sb = saved[--si]; auto& sa = saved[--si];
                    Shape sha = sa.shape(), shb = sb.shape();
                    accum(e.in0, GradTensor::unbroadcast(g.mul(sb), sha));
                    accum(e.in1, GradTensor::unbroadcast(g.mul(sa), shb));
                    break;
                }
                case OpType::DIV: {
                    auto& sb = saved[--si]; auto& sa = saved[--si];
                    Shape sha = sa.shape(), shb = sb.shape();
                    accum(e.in0, GradTensor::unbroadcast(g.div(sb), sha));
                    accum(e.in1, GradTensor::unbroadcast(g.mul(sa).neg().div(sb.square()), shb));
                    break;
                }
                case OpType::POW: {
                    auto& so = saved[--si]; auto& sb = saved[--si]; auto& sa = saved[--si];
                    Shape sha = sa.shape(), shb = sb.shape();
                    accum(e.in0, GradTensor::unbroadcast(g.mul(sb).mul(sa.pow(sb.add_scalar(-1.f))), sha));
                    accum(e.in1, GradTensor::unbroadcast(g.mul(so).mul(sa.log()), shb));
                    break;
                }
                case OpType::MATMUL: {
                    auto& b = saved[--si]; auto& a = saved[--si];
                    accum(e.in0, g.matmul(b.transpose()));
                    accum(e.in1, a.transpose().matmul(g));
                    break;
                }
                case OpType::SUM: {
                    int dim = e.iparams[0], kd = e.iparams[1];
                    Shape x_shape = S[e.in0].shape();
                    int ndim = (int)x_shape.size();
                    int nd = dim < 0 ? dim + ndim : dim;
                    Tensor ge = g;
                    if (!kd) { Shape s = x_shape; s[nd] = 1; ge = ge.reshape(s); }
                    accum(e.in0, ge.add(Tensor::full(x_shape, 0.f)));
                    break;
                }
                case OpType::MEAN: {
                    int dim = e.iparams[0], kd = e.iparams[1];
                    Shape x_shape = S[e.in0].shape();
                    int ndim = (int)x_shape.size();
                    int nd = dim < 0 ? dim + ndim : dim;
                    int n = nd >= 0 ? x_shape[nd] : 1;
                    for (auto d : x_shape) if (nd < 0) n = 1; // all-reduce handled by sum
                    if (nd >= 0) n = x_shape[nd];
                    Tensor ge = g.mul_scalar(1.f / n);
                    if (!kd) { Shape s = x_shape; s[nd] = 1; ge = ge.reshape(s); }
                    accum(e.in0, ge.add(Tensor::full(x_shape, 0.f)));
                    break;
                }
                case OpType::TRANSPOSE:
                    accum(e.in0, g.transpose());
                    break;
                case OpType::TRANSPOSE2:
                    accum(e.in0, g.transpose(e.iparams[0], e.iparams[1]));
                    break;
                case OpType::RESHAPE:
                case OpType::FLATTEN:
                    accum(e.in0, g.reshape(S[e.in0].shape()));
                    break;
                case OpType::CONV2D: {
                    auto& w = saved[--si]; auto& x = saved[--si];
                    Shape xs = x.shape(), ws = w.shape();
                    Tensor dx(xs), dw(ws);
                    bool hb = e.in2 >= 0;
                    if (hb) {
                        Tensor db({ws[0]});
                        Tensor::conv2d_backward_cudnn(x, w, g, dx, dw, &db,
                            e.iparams[0], e.iparams[1], e.iparams[2], e.iparams[3],
                            e.iparams[4], e.iparams[5], e.iparams[6]);
                        accum(e.in2, db);
                    } else {
                        Tensor::conv2d_backward_cudnn(x, w, g, dx, dw, nullptr,
                            e.iparams[0], e.iparams[1], e.iparams[2], e.iparams[3],
                            e.iparams[4], e.iparams[5], e.iparams[6]);
                    }
                    accum(e.in0, dx);
                    accum(e.in1, dw);
                    break;
                }
                case OpType::MAX_POOL2D: {
                    auto& indices = saved[--si];
                    int B = S[e.in0].shape()[0], C = S[e.in0].shape()[1];
                    int H = S[e.in0].shape()[2], W = S[e.in0].shape()[3];
                    accum(e.in0, Tensor::maxpool2d_backward(g, indices, B, C, H, W));
                    break;
                }
                case OpType::AVG_POOL2D: {
                    int H = S[e.in0].shape()[2], W = S[e.in0].shape()[3];
                    accum(e.in0, g.avgpool2d_backward(H, W,
                        e.iparams[0], e.iparams[1], e.iparams[2], e.iparams[3],
                        e.iparams[4], e.iparams[5], e.iparams[6]));
                    break;
                }
                case OpType::LOOP_SLICE: {
                    // backward of slice+squeeze: unsqueeze + pad with zeros via cat
                    int dim = e.iparams[0], iter = e.iparams[1];
                    Shape orig = S[e.in0].shape();
                    Tensor gu = g.unsqueeze(dim); // undo squeeze
                    std::vector<Tensor> parts;
                    if (iter > 0) {
                        Shape zs = orig; zs[dim] = iter;
                        Tensor z(zs); cudaMemsetAsync(z.data<float>(), 0, z.size() * sizeof(float), 0);
                        parts.push_back(std::move(z));
                    }
                    parts.push_back(gu.contiguous());
                    int after = orig[dim] - iter - 1;
                    if (after > 0) {
                        Shape zs = orig; zs[dim] = after;
                        Tensor z(zs); cudaMemsetAsync(z.data<float>(), 0, z.size() * sizeof(float), 0);
                        parts.push_back(std::move(z));
                    }
                    Tensor dx = parts.size() == 1 ? parts[0] : Tensor::cat(parts, dim);
                    accum(e.in0, dx);
                    break;
                }
                case OpType::LOOP_END: {
                    // CARRY backward: grad flows from to→from
                    accum(e.in0, g);
                    break;
                }
                case OpType::RNN_CUDNN: {
                    auto& rop = rnn_ops_[e.iparams[0]];
                    // Restore saved
                    auto& saved_y = saved[--si];
                    auto& saved_cx = saved[--si];
                    auto& saved_hx = saved[--si];
                    auto& saved_x = saved[--si];
                    const Tensor* hx = rop.hx_slot >= 0 ? &saved_hx : nullptr;
                    const Tensor* cx = rop.cx_slot >= 0 ? &saved_cx : nullptr;
                    Tensor dy = has_grad[rop.y_slot] ? G[rop.y_slot] : Tensor::full(saved_y.shape(), 0.f);
                    const Tensor* dhy = has_grad[rop.hy_slot] ? &G[rop.hy_slot] : nullptr;
                    const Tensor* dcy = has_grad[rop.cy_slot] ? &G[rop.cy_slot] : nullptr;

                    std::vector<Tensor> wih, whh, bih, bhh;
                    for (int s : rop.wih_slots) wih.push_back(S[s]);
                    for (int s : rop.whh_slots) whh.push_back(S[s]);
                    for (int s : rop.bih_slots) bih.push_back(S[s]);
                    for (int s : rop.bhh_slots) bhh.push_back(S[s]);

                    auto result = Tensor::rnn_backward(saved_x, hx, cx, saved_y, dy, dhy, dcy,
                        wih, whh, bih, bhh, rop.mode, rop.hidden_size, rop.num_layers, rop.bidirectional);
                    // result: [dx, dhx, dcx, dwih_0, dwhh_0, dbih_0, dbhh_0, ...]
                    accum(rop.x_slot, result[0]);
                    if (rop.hx_slot >= 0) accum(rop.hx_slot, result[1]);
                    if (rop.cx_slot >= 0) accum(rop.cx_slot, result[2]);
                    int nl = rop.num_layers * (rop.bidirectional ? 2 : 1);
                    for (int l = 0; l < nl; l++) {
                        accum(rop.wih_slots[l], result[3 + l * 4]);
                        accum(rop.whh_slots[l], result[4 + l * 4]);
                        accum(rop.bih_slots[l], result[5 + l * 4]);
                        accum(rop.bhh_slots[l], result[6 + l * 4]);
                    }
                    break;
                }
                default: break;
            }
        }

        // ---- Adam step (outside arena — params are external Tensors) ----
        active_arena = nullptr;
        for (int i = 0; i < np; i++) {
            if (!has_grad[param_slots_[i]]) continue;
            Tensor::adam_step(params[i], G[param_slots_[i]], m_states[i], v_states[i],
                              lr, beta1, beta2, eps, bc1, bc2, wd);
        }

    }
};

} // namespace jstorch
