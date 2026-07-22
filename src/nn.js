/**
 * Neural Network Module - Pure JavaScript Implementation
 */

import { tensor, zeros, ones } from './tensor.js';

// ==================== Layers ====================

export function linear(in_features, out_features, bias = true) {
  // Xavier initialization
  const limit = Math.sqrt(6 / (in_features + out_features));
  
  const weight_data = Array.from({ length: out_features }, () =>
    Array.from({ length: in_features }, () => (Math.random() * 2 - 1) * limit)
  );
  
  const weight = tensor(weight_data, true);
  const bias_tensor = bias ? tensor(Array(out_features).fill(0), true) : null;
  
  return {
    forward: (x) => {
      // y = x @ W^T + b
      const out = x.matmul(weight);
      return bias_tensor ? out.add(bias_tensor) : out;
    },
    
    parameters: () => bias_tensor ? [weight, bias_tensor] : [weight],
    
    weight,
    bias: bias_tensor
  };
}

export function relu() {
  return {
    forward: (x) => x.relu(),
    parameters: () => []
  };
}

export function flatten(start_dim = 0, end_dim = -1) {
  return {
    forward: (x) => {
      // For now, assume already flat for MLP
      return x;
    },
    parameters: () => []
  };
}

export function sequential(layers) {
  return {
    forward: (x) => {
      return layers.reduce((input, layer) => layer.forward(input), x);
    },
    
    parameters: () => {
      return layers.flatMap(layer => layer.parameters());
    },
    
    train: () => {},
    eval: () => {}
  };
}

// ==================== Loss Functions ====================

export function cross_entropy_loss() {
  return (output, target) => {
    // output: [batch, num_classes] raw logits
    // target: [batch, num_classes] one-hot or probabilities
    
    // Softmax: exp(x_i) / sum(exp(x_j))
    // Log-softmax: log(exp(x_i) / sum(exp(x_j))) = x_i - log(sum(exp(x_j)))
    // Cross entropy: -sum(target * log_softmax(output))
    
    // Step 1: Compute max for numerical stability
    // output_max = max(output, dim=1)
    // output_shifted = output - output_max
    
    // Step 2: Compute exp
    const exp_output = output.exp();
    
    // Step 3: Compute sum of exp
    const sum_exp = exp_output.sum();  // TODO: should be sum along dim=1
    
    // Step 4: Compute log(sum(exp))
    const log_sum_exp = sum_exp.log();
    
    // Step 5: Log-softmax = output - log_sum_exp
    // For simplicity, we'll compute a mean loss
    
    // Step 6: Cross entropy = -mean(target * log_softmax)
    // Simplified: just return a scalar loss for now
    
    // TODO: Implement proper cross entropy with target class indices
    // For now, return MSE-like loss as placeholder
    const diff = output.sub(target);
    const squared = diff.mul(diff);
    const loss = squared.mean();
    
    return loss;
  };
}

export function mse_loss() {
  return (output, target) => {
    const diff = output.sub(target);
    const squared = diff.mul(diff);
    return squared.mean();
  };
}

// ==================== Softmax ====================

export function softmax(input, dim = 1) {
  // softmax(x_i) = exp(x_i) / sum(exp(x_j))
  const exp_input = input.exp();
  const sum_exp = exp_input.sum();  // TODO: sum along specific dim
  return exp_input.div(sum_exp);
}

export function log_softmax(input, dim = 1) {
  // log_softmax(x_i) = x_i - log(sum(exp(x_j)))
  const exp_input = input.exp();
  const sum_exp = exp_input.sum();
  const log_sum = sum_exp.log();
  
  // Broadcast subtraction (simplified)
  return input.sub(log_sum);
}
