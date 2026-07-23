// index.d.ts - 完美架构类型定义
export class Tensor {
    readonly shape: number[];
    
    constructor(data: any);
    
    // 静态构造
    static zeros(shape: number[]): Tensor;
    static ones(shape: number[]): Tensor;
    static randn(shape: number[]): Tensor;
    static rand(shape: number[]): Tensor;
    
    // 二元操作
    add(other: Tensor): Tensor;
    sub(other: Tensor): Tensor;
    mul(other: Tensor): Tensor;
    div(other: Tensor): Tensor;
    
    // 一元操作
    abs(): Tensor;
    sqrt(): Tensor;
    square(): Tensor;
    exp(): Tensor;
    log(): Tensor;
    sin(): Tensor;
    cos(): Tensor;
    sigmoid(): Tensor;
    tanh(): Tensor;
    relu(): Tensor;
    
    // 归约
    sum(dim?: number): Tensor;
    mean(dim?: number): Tensor;
    
    // Shape操作
    reshape(shape: number[]): Tensor;
    transpose(): Tensor;
    
    // In-place
    add_(other: Tensor): this;
    sub_(other: Tensor): this;
    mul_(scalar: number): this;
    
    // 工具
    toArray(): any;
    toString(): string;
    size(dim?: number): number | number[];
    dim(): number;
    numel(): number;
}

// 便利函数
export function zeros(shape: number[]): Tensor;
export function ones(shape: number[]): Tensor;
export function randn(shape: number[]): Tensor;
export function rand(shape: number[]): Tensor;
export function tensor(data: any): Tensor;

// 神经网络
export namespace nn {
    class Module {
        training: boolean;
        parameters(): Tensor[];
        train(): void;
        eval(): void;
        forward(x: Tensor): Tensor;
    }
    
    class Linear extends Module {
        constructor(in_features: number, out_features: number);
    }
    
    class Conv1d extends Module {
        constructor(in_channels: number, out_channels: number, kernel_size: number);
    }
    
    class LayerNorm extends Module {
        constructor(normalized_shape: number | number[], eps?: number);
    }
    
    class ReLU extends Module {}
    class Sigmoid extends Module {}
    class Tanh extends Module {}
    
    class Sequential extends Module {
        constructor(...layers: Module[]);
    }
}

// 优化器
export namespace optim {
    class Optimizer {
        constructor(parameters: Tensor[], defaults: any);
        zero_grad(): void;
        step(): void;
    }
    
    class SGD extends Optimizer {
        constructor(parameters: Tensor[], options?: {
            lr?: number;
            momentum?: number;
            weight_decay?: number;
        });
    }
    
    class Adam extends Optimizer {
        constructor(parameters: Tensor[], options?: {
            lr?: number;
            betas?: [number, number];
            eps?: number;
            weight_decay?: number;
        });
    }
    
    class AdamW extends Adam {}
}

// 自动求导
export function no_grad<T>(fn: () => T): T;
export function enable_grad<T>(fn: () => T): T;
export function is_grad_enabled(): boolean;
export function set_grad_enabled(enabled: boolean): <T>(fn: () => T) => T;
