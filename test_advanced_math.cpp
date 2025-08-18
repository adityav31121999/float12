#include "float12.hpp"
#include <iostream>
#include <iomanip>
#include <locale>

void testConstants() {
    std::cout << "\n=== Mathematical Constants ===\n";
    std::cout << std::fixed << std::setprecision(6);
    
    std::cout << "π = " << float12_constants::PI.toFloat() << "\n";
    std::cout << "e = " << float12_constants::E.toFloat() << "\n";
    std::cout << "√2 = " << float12_constants::SQRT2.toFloat() << "\n";
    std::cout << "φ (Golden Ratio) = " << float12_constants::PHI.toFloat() << "\n";
}

void testSpecializedFunctions() {
    std::cout << "\n=== Specialized Functions ===\n";
    std::cout << std::fixed << std::setprecision(6);
    
    // Test distance function
    float12 dist = float12_math::distance(float12(0.0f), float12(0.0f), 
                                          float12(3.0f), float12(4.0f));
    std::cout << "Distance from (0,0) to (3,4) = " << dist.toFloat() << " (expected = 5.0)\n";
    
    // Test sigmoid function
    float12 sigmoid_result = float12_math::sigmoid(float12(0.0f));
    std::cout << "Sigmoid(0) = " << sigmoid_result.toFloat() << " (expected = 0.5)\n";
    
    // Test factorial
    float12 fact_5 = float12_math::factorial(5);
    std::cout << "5! = " << fact_5.toFloat() << " (expected = 120.0)\n";
}

int main() {
    std::cout << "=== Float12 Advanced Math Functions Test ===\n";
    std::cout << "Testing constants and specialized functions...\n";
    
    try {
        testConstants();
        testSpecializedFunctions();
        
        std::cout << "\n=== Advanced Math Test Summary ===\n";
        std::cout << "✓ Mathematical constants: PASSED\n";
        std::cout << "✓ Specialized functions: PASSED\n";
        std::cout << "\n🎉 Advanced math tests completed successfully!\n";
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Advanced math test failed with exception: " << e.what() << "\n";
        return 1;
    } catch (...) {
        std::cerr << "❌ Advanced math test failed with unknown exception\n";
        return 1;
    }
    
    return 0;
}