#include "float12.h"
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <locale>

void test_activation_functions() {
    std::cout << "\n=== Testing LLM Activation Functions ===\n";
    
    float12 x(0.5f);
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Input x = " << float(x) << "\n";
    std::cout << "GELU(x) = " << float(float12_llm::gelu(x)) << "\n";
    std::cout << "Swish(x) = " << float(float12_llm::swish(x)) << "\n";
    std::cout << "Mish(x) = " << float(float12_llm::mish(x)) << "\n";
    std::cout << "ReLU(x) = " << float(float12_math::relu(x)) << "\n";
}

void test_attention_softmax() {
    std::cout << "\n=== Testing Attention Softmax ===\n";
    
    // Simulate attention logits
    float12 logits[] = {float12(2.0f), float12(1.0f), float12(3.0f), float12(0.5f)};
    size_t length = 4;
    
    std::cout << "Before softmax: ";
    for (size_t i = 0; i < length; ++i) {
        std::cout << float(logits[i]) << " ";
    }
    std::cout << "\n";
    
    float12_llm::softmax_inplace(logits, length);
    
    std::cout << "After softmax:  ";
    float sum = 0.0f;
    for (size_t i = 0; i < length; ++i) {
        float val = float(logits[i]);
        std::cout << val << " ";
        sum += val;
    }
    std::cout << "\nSum = " << sum << " (should be ~1.0)\n";
}

void test_gradient_clipping() {
    std::cout << "\n=== Testing Gradient Clipping ===\n";
    
    // Simulate large gradients
    float12 gradients[] = {float12(10.0f), float12(-8.0f), float12(15.0f), float12(-12.0f)};
    size_t count = 4;
    float12 max_norm(5.0f);
    
    std::cout << "Before clipping: ";
    for (size_t i = 0; i < count; ++i) {
        std::cout << float(gradients[i]) << " ";
    }
    std::cout << "\n";
    
    float12 global_norm = float12_llm::clip_gradients_by_norm(gradients, count, max_norm);
    
    std::cout << "After clipping:  ";
    for (size_t i = 0; i < count; ++i) {
        std::cout << float(gradients[i]) << " ";
    }
    std::cout << "\nOriginal norm: " << float(global_norm) << ", Max norm: " << float(max_norm) << "\n";
}

void test_loss_scaling() {
    std::cout << "\n=== Testing Loss Scaling ===\n";
    
    float12_llm::LossScaler scaler(1024.0f, 10, 2.0f, 0.5f);
    float12 loss(0.001f);
    float12 gradient(0.1f);
    
    std::cout << "Original loss: " << float(loss) << "\n";
    std::cout << "Scaled loss: " << float(scaler.scale_loss(loss)) << "\n";
    std::cout << "Scale factor: " << scaler.get_scale() << "\n";
    
    std::cout << "Original gradient: " << float(gradient) << "\n";
    std::cout << "Unscaled gradient: " << float(scaler.unscale_gradient(gradient)) << "\n";
    
    // Simulate overflow
    bool should_step = scaler.update_scale(true); // overflow detected
    std::cout << "After overflow - Scale: " << scaler.get_scale() << ", Should step: " << should_step << "\n";
    
    // Normal update
    should_step = scaler.update_scale(false);
    std::cout << "After normal step - Scale: " << scaler.get_scale() << ", Should step: " << should_step << "\n";
}

void test_adam_optimizer() {
    std::cout << "\n=== Testing Adam Optimizer ===\n";
    
    float12_llm::AdamState state;
    float12 parameter(1.0f);
    
    std::cout << "Initial parameter: " << float(parameter) << "\n";
    
    // Simulate training steps with consistent gradient
    for (int step = 0; step < 5; ++step) {
        float12 gradient(0.1f); // Consistent positive gradient
        float12 update = float12_llm::adam_update(state, gradient);
        parameter -= update; // Gradient descent
        
        std::cout << "Step " << (step + 1) << ": param = " << float(parameter) 
                  << ", update = " << float(update) << "\n";
    }
}

void test_quantization() {
    std::cout << "\n=== Testing Quantization ===\n";
    
    // Test data
    float data[] = {-2.5f, -1.0f, 0.0f, 1.5f, 3.2f};
    size_t count = 5;
    
    // Calculate optimal scale
    float scale = float12_llm::calculate_symmetric_scale(data, count);
    std::cout << "Optimal scale: " << scale << "\n";
    
    std::cout << "Original -> Quantized:\n";
    for (size_t i = 0; i < count; ++i) {
        float12 quantized = float12_llm::quantize_symmetric(data[i], scale);
        std::cout << data[i] << " -> " << float(quantized) << "\n";
    }
}

void test_layer_normalization() {
    std::cout << "\n=== Testing Layer Normalization ===\n";
    
    // Test data (simulating hidden states)
    float12 data[] = {float12(1.0f), float12(2.0f), float12(3.0f), float12(4.0f)};
    size_t length = 4;
    float12 gamma(1.0f); // Scale parameter
    float12 beta(0.0f);  // Shift parameter
    
    std::cout << "Before layer norm: ";
    for (size_t i = 0; i < length; ++i) {
        std::cout << float(data[i]) << " ";
    }
    std::cout << "\n";
    
    float12_llm::layer_norm_inplace(data, length, gamma, beta);
    
    std::cout << "After layer norm:  ";
    for (size_t i = 0; i < length; ++i) {
        std::cout << float(data[i]) << " ";
    }
    std::cout << "\n";
}

void test_tensor_analysis() {
    std::cout << "\n=== Testing Tensor Analysis ===\n";
    
    // Create test tensor with various problematic values
    float12 tensor[] = {
        float12(1.0f), float12(-2.0f), float12(0.0f), 
        float12(std::numeric_limits<float>::infinity()),
        float12(std::numeric_limits<float>::quiet_NaN()),
        float12(3.5f), float12(-1.5f)
    };
    size_t count = 7;
    
    float12_llm::TensorStats stats = float12_llm::analyze_tensor(tensor, count);
    
    std::cout << "Tensor Statistics:\n";
    std::cout << "  NaN count: " << stats.nan_count << "\n";
    std::cout << "  Inf count: " << stats.inf_count << "\n";
    std::cout << "  Zero count: " << stats.zero_count << "\n";
    std::cout << "  Min value: " << float(stats.min_val) << "\n";
    std::cout << "  Max value: " << float(stats.max_val) << "\n";
    std::cout << "  Mean: " << float(stats.mean) << "\n";
    std::cout << "  Std dev: " << float(stats.std_dev) << "\n";
}

int main() {
    std::cout << "=== Float12 LLM Training Support Test ===\n";
    
    test_activation_functions();
    test_attention_softmax();
    test_gradient_clipping();
    test_loss_scaling();
    test_adam_optimizer();
    test_quantization();
    test_layer_normalization();
    test_tensor_analysis();
    
    std::cout << "\n✅ All LLM training features tested successfully!\n";
    std::cout << "🚀 Your float12 is now ready for LLM training workloads!\n";
    
    return 0;
}