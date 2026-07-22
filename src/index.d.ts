/**
 * JsTorch - PyTorch-like Tensor Library for Node.js
 * TypeScript Type Definitions
 */

// ==================== Core Types ====================

export type Device = 'cpu' | 'cuda';
export type DType = 'float32' | 'float64' | 'int32' | 'int64';
export type Shape = number[];

// ==================== Tensor ====================

export class Tensor {
  constructor(data: number | number[] | number[][] | number[][][], requires_grad?: boolean);
  
  // Properties
  readonly shape: Shape;
  readonly size: number;
  requires_grad: boolean;
  grad: Tensor | null;
  
  // Binary operations
  add(other: Tensor): Tensor;
  add_(other: Tensor): Tensor;
  sub(other: Tensor): Tensor;
  mul(other: Tensor): Tensor;
  div(other: Tensor): Tensor;
  matmul(other: Tensor): Tensor;
  
  // Unary operations
  relu(): Tensor;
  relu_(): Tensor;
  exp(): Tensor;
  log(): Tensor;
  neg(): Tensor;
  
  // Scalar operations
  mulScalar(scalar: number): Tensor;
  
  // Reduction operations
  sum(): Tensor;
  mean(): Tensor;
  
  // Autograd
  backward(gradient?: Tensor): void;
  zero_grad(): void;
  
  // Utilities
  toArray(): number | number[] | number[][];
  item(): number;
}

// ==================== Tensor Creation ====================

export function tensor(
  data: number | number[] | number[][] | number[][][],
  requires_grad?: boolean
): Tensor;

export function zeros(shape: Shape): Tensor;
export function ones(shape: Shape): Tensor;

// ==================== Autograd ====================

export function no_grad<T>(fn: () => T): T;
export function enable_grad<T>(fn: () => T): T;
export function is_grad_enabled(): boolean;
export function set_grad_enabled(enabled: boolean): <T>(fn: () => T) => T;

// ==================== Neural Network ====================

export namespace nn {
  // Module interface
  interface Module {
    forward(input: Tensor): Tensor;
    parameters(): Tensor[];
    train?(): void;
    eval?(): void;
  }
  
  // Layers
  export function linear(in_features: number, out_features: number, bias?: boolean): Module & {
    weight: Tensor;
    bias: Tensor | null;
  };
  
  export function relu(): Module;
  export function flatten(start_dim?: number, end_dim?: number): Module;
  
  // Containers
  export function sequential(layers: Module[]): Module;
  
  // Loss functions
  export function cross_entropy_loss(): (output: Tensor, target: Tensor) => Tensor;
  export function mse_loss(): (output: Tensor, target: Tensor) => Tensor;
  
  // Activations
  export function softmax(input: Tensor, dim?: number): Tensor;
  export function log_softmax(input: Tensor, dim?: number): Tensor;
}

// ==================== Optimizers ====================

export namespace optim {
  interface Optimizer {
    zero_grad(): void;
    step(): void;
  }
  
  export function sgd(
    params: Tensor[],
    options: {
      lr: number;
      momentum?: number;
    }
  ): Optimizer;
  
  export function adam(
    params: Tensor[],
    options: {
      lr: number;
      betas?: [number, number];
      eps?: number;
    }
  ): Optimizer;
}

// ==================== CUDA ====================

export namespace cuda {
  export function is_available(): boolean;
  export function device_count(): number;
  export function synchronize(): void;
}

// ==================== Default Export ====================

declare const jstorch: {
  // Core
  tensor: typeof tensor;
  zeros: typeof zeros;
  ones: typeof ones;
  Tensor: typeof Tensor;
  
  // Autograd
  no_grad: typeof no_grad;
  enable_grad: typeof enable_grad;
  is_grad_enabled: typeof is_grad_enabled;
  set_grad_enabled: typeof set_grad_enabled;
  
  // Modules
  nn: typeof nn;
  optim: typeof optim;
  cuda: typeof cuda;
};

export default jstorch;
