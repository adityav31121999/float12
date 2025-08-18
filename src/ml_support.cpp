#include "float12.hpp"
#include <vector>
#include <algorithm>
#include <cmath>

// LLM Training Support Functions for float12
namespace float12_llm {
    
    // ===== ACTIVATION FUNCTIONS FOR TRANSFORMERS =====
    
    // GELU activation (Gaussian Error Linear Unit) - critical for transformers
    float12 gelu(const float12& x) {
        // Approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
        float12 x_cubed = x * x * x;
        float12 inner = x + float12(0.044715f) * x_cubed;
        float12 tanh_arg = float12(0.7978845608f) * inner; // sqrt(2/π)
        return float12(0.5f) * x * (float12(1.0f) + tanh(tanh_arg));
    }
    
    // Swish/SiLU activation (x * sigmoid(x))
    float12 swish(const float12& x) {
        return x * float12_math::sigmoid(x);
    }
    
    // Mish activation
    float12 mish(const float12& x) {
        return x * tanh(float12_math::softplus(x));
    }
    
    // ===== ATTENTION MECHANISM SUPPORT =====
    
    // Scaled dot-product attention softmax (numerically stable)
    void softmax_inplace(float12* logits, size_t length) {
        // Find max for numerical stability
        float12 max_val = *std::max_element(logits, logits + length);
        
        // Subtract max and compute exp
        float12 sum(0.0f);
        for (size_t i = 0; i < length; ++i) {
            logits[i] = exp(logits[i] - max_val);
            sum += logits[i];
        }
        
        // Normalize
        for (size_t i = 0; i < length; ++i) {
            logits[i] /= sum;
        }
    }
    
    // Temperature scaling for generation
    float12 apply_temperature(const float12& logit, float temperature) {
        return logit / float12(temperature);
    }
    
    // ===== GRADIENT OPERATIONS =====
    
    // Gradient clipping by global norm
    float12 clip_gradients_by_norm(float12* gradients, size_t count, float12 max_norm) {
        // Calculate global norm
        float12 global_norm_sq(0.0f);
        for (size_t i = 0; i < count; ++i) {
            global_norm_sq += gradients[i] * gradients[i];
        }
        float12 global_norm = sqrt(global_norm_sq);
        
        // Clip if necessary
        if (global_norm > max_norm) {
            float12 scale_factor = max_norm / global_norm;
            for (size_t i = 0; i < count; ++i) {
                gradients[i] *= scale_factor;
            }
        }
        
        return global_norm;
    }
    
    // Gradient accumulation with overflow detection
    bool accumulate_gradient(float12& accumulated, const float12& new_grad, int step_count) {
        float12 old_val = accumulated;
        accumulated += new_grad;
        
        // Check for overflow/underflow
        if (accumulated.isInfinite() || accumulated.isNaN()) {
            accumulated = old_val; // Revert
            return false; // Overflow detected
        }
        return true;
    }
    
    // ===== LOSS SCALING FOR MIXED PRECISION =====
    
    LossScaler::LossScaler(float initial_scale, int interval, 
                          float growth, float backoff)
        : current_scale(initial_scale), growth_interval(interval), 
          growth_counter(0), growth_factor(growth), backoff_factor(backoff) {}
    
    float12 LossScaler::scale_loss(const float12& loss) {
        return loss * float12(current_scale);
    }
    
    float12 LossScaler::unscale_gradient(const float12& gradient) {
        return gradient / float12(current_scale);
    }
    
    bool LossScaler::update_scale(bool overflow_detected) {
        if (overflow_detected) {
            current_scale *= backoff_factor;
            growth_counter = 0;
            return false; // Skip optimizer step
        } else {
            growth_counter++;
            if (growth_counter >= growth_interval) {
                current_scale *= growth_factor;
                growth_counter = 0;
            }
            return true; // Proceed with optimizer step
        }
    }
    
    float LossScaler::get_scale() const { return current_scale; }
    
    // ===== QUANTIZATION FOR MODEL COMPRESSION =====
    
    // Symmetric quantization (zero-point = 0)
    float12 quantize_symmetric(float value, float scale) {
        float quantized = std::round(value / scale);
        // Clamp to float12 range
        quantized = std::max(-2048.0f, std::min(2047.0f, quantized));
        return float12(quantized * scale);
    }
    
    // Calculate optimal scale for symmetric quantization
    float calculate_symmetric_scale(const float* data, size_t count) {
        float max_abs = 0.0f;
        for (size_t i = 0; i < count; ++i) {
            max_abs = std::max(max_abs, std::abs(data[i]));
        }
        return max_abs / 2047.0f; // Use full range of float12
    }
    
    // ===== OPTIMIZER SUPPORT =====
    
    AdamState::AdamState() : m(0.0f), v(0.0f), step(0) {}
    
    float12 adam_update(AdamState& state, const float12& gradient, 
                       float learning_rate, float beta1, 
                       float beta2, float epsilon) {
        state.step++;
        
        // Update biased first moment estimate
        state.m = float12(beta1) * state.m + float12(1.0f - beta1) * gradient;
        
        // Update biased second raw moment estimate  
        state.v = float12(beta2) * state.v + float12(1.0f - beta2) * gradient * gradient;
        
        // Compute bias-corrected first moment estimate
        float12 m_hat = state.m / float12(1.0f - std::pow(beta1, static_cast<float>(state.step)));
        
        // Compute bias-corrected second raw moment estimate
        float12 v_hat = state.v / float12(1.0f - std::pow(beta2, static_cast<float>(state.step)));
        
        // Update parameters
        return float12(learning_rate) * m_hat / (sqrt(v_hat) + float12(epsilon));
    }
    
    // ===== NUMERICAL STABILITY UTILITIES =====
    
    // Log-sum-exp for numerical stability
    float12 log_sum_exp(const float12* values, size_t count) {
        float12 max_val = *std::max_element(values, values + count);
        float12 sum(0.0f);
        
        for (size_t i = 0; i < count; ++i) {
            sum += exp(values[i] - max_val);
        }
        
        return max_val + log(sum);
    }
    
    // Safe division with gradient flow
    float12 safe_divide_with_gradient(const float12& numerator, const float12& denominator, 
                                     float12 epsilon) {
        float12 safe_denom = denominator + (denominator >= float12(0.0f) ? epsilon : -epsilon);
        return numerator / safe_denom;
    }
    
    // ===== MEMORY EFFICIENT OPERATIONS =====
    
    // In-place layer normalization
    void layer_norm_inplace(float12* data, size_t length, const float12& gamma, const float12& beta) {
        // Calculate mean
        float12 mean(0.0f);
        for (size_t i = 0; i < length; ++i) {
            mean += data[i];
        }
        mean /= float12(static_cast<float>(length));
        
        // Calculate variance
        float12 variance(0.0f);
        for (size_t i = 0; i < length; ++i) {
            float12 diff = data[i] - mean;
            variance += diff * diff;
        }
        variance /= float12(static_cast<float>(length));
        
        // Normalize
        float12 std_dev = sqrt(variance + float12(1e-5f));
        for (size_t i = 0; i < length; ++i) {
            data[i] = gamma * (data[i] - mean) / std_dev + beta;
        }
    }
    
    // ===== DEBUGGING AND MONITORING =====
    
    TensorStats analyze_tensor(const float12* data, size_t count) {
        TensorStats stats = {};
        stats.min_val = float12(std::numeric_limits<float>::max());
        stats.max_val = float12(std::numeric_limits<float>::lowest());
        
        float12 sum(0.0f);
        size_t valid_count = 0;
        
        for (size_t i = 0; i < count; ++i) {
            if (data[i].isNaN()) {
                stats.nan_count++;
            } else if (data[i].isInfinite()) {
                stats.inf_count++;
            } else {
                if (data[i].isZero()) stats.zero_count++;
                
                stats.min_val = fmin(stats.min_val, data[i]);
                stats.max_val = fmax(stats.max_val, data[i]);
                sum += data[i];
                valid_count++;
            }
        }
        
        if (valid_count > 0) {
            stats.mean = sum / float12(static_cast<float>(valid_count));
            
            // Calculate standard deviation
            float12 var_sum(0.0f);
            for (size_t i = 0; i < count; ++i) {
                if (!data[i].isNaN() && !data[i].isInfinite()) {
                    float12 diff = data[i] - stats.mean;
                    var_sum += diff * diff;
                }
            }
            stats.std_dev = sqrt(var_sum / float12(static_cast<float>(valid_count)));
        }
        
        return stats;
    }
}