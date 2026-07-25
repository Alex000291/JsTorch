#pragma once
#include "tensor.hpp"
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

    // --- Binary ops ---
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

        // Forward
        for (auto& op : ops_) {
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
};

} // namespace jstorch
