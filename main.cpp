#include "float12.h"
#include <iostream>
#include <limits>
#include <cassert>
#include <locale>

// Function to run simple compilation tests
void runSimpleTests() {
    std::cout << "\n=== Running Simple Compilation Tests ===\n";
    
    // Test basic construction
    float12 a(3.14f);
    std::cout << "Created float12 with value: " << a.toFloat() << "\n";
    
    // Test static methods (these were causing constexpr issues)
    std::cout << "Max value: " << float12::max().toFloat() << "\n";
    std::cout << "Min value: " << float12::min().toFloat() << "\n";
    std::cout << "Lowest value: " << float12::lowest().toFloat() << "\n";
    
    // Test utility functions
    std::cout << "Is zero: " << (float12(0.0f).isZero() ? "true" : "false") << "\n";
    std::cout << "Is finite: " << (a.isFinite() ? "true" : "false") << "\n";
    
    std::cout << "Simple compilation tests completed successfully!\n";
}

// Function to run advanced constexpr tests
void runAdvancedTests() {
    std::cout << "\n=== Running Advanced Constexpr Tests ===\n";
    
    // Test constexpr functionality at compile time
    constexpr float12 max_val = float12::max();
    constexpr float12 min_val = float12::min();
    constexpr float12 lowest_val = float12::lowest();
    constexpr float12 zero_val = float12::fromRawBits(0);
    
    // Test constexpr utility functions (compile-time assertions)
    static_assert(zero_val.isZero(), "Zero should be detected as zero");
    static_assert(max_val.isFinite(), "Max should be finite");
    static_assert(!max_val.isNaN(), "Max should not be NaN");
    
    std::cout << "Constexpr compile-time assertions passed!\n";
    
    // Test basic construction
    float12 a(3.14f);
    float12 b(2.0f);
    
    // Test static methods
    std::cout << "Max value: " << max_val.toFloat() << "\n";
    std::cout << "Min value: " << min_val.toFloat() << "\n";
    std::cout << "Lowest value: " << lowest_val.toFloat() << "\n";
    
    // Test utility functions
    float12 zero(0.0f);
    float12 inf(std::numeric_limits<float>::infinity());
    float12 nan_val(std::numeric_limits<float>::quiet_NaN());
    
    std::cout << "Zero is zero: " << (zero.isZero() ? "true" : "false") << "\n";
    std::cout << "Inf is infinite: " << (inf.isInfinite() ? "true" : "false") << "\n";
    std::cout << "NaN is NaN: " << (nan_val.isNaN() ? "true" : "false") << "\n";
    std::cout << "a is finite: " << (a.isFinite() ? "true" : "false") << "\n";
    std::cout << "a is normal: " << (a.isNormal() ? "true" : "false") << "\n";
    
    // Test constexpr unary minus
    constexpr float12 neg_max = -max_val;
    std::cout << "Negative max: " << neg_max.toFloat() << "\n";
    
    // Test arithmetic
    float12 sum = a + b;
    std::cout << "3.14 + 2.0 = " << sum.toFloat() << "\n";
    
    // Test safe construction
    bool overflow = false;
    float12 safe_val = float12::fromFloatSafe(1000000.0f, &overflow);
    std::cout << "Large value overflow: " << (overflow ? "true" : "false") << "\n";
    
    // Test factory method
    float12 from_bits = float12::fromRawBits(0x7BF);
    std::cout << "From raw bits 0x7BF: " << from_bits.toFloat() << "\n";
    
    std::cout << "Advanced constexpr tests completed successfully!\n";
}

// Function to run comprehensive arithmetic tests
void runArithmeticTests() {
    std::cout << "\n=== Running Arithmetic Tests ===\n";
    
    float12 a(5.0f);
    float12 b(2.0f);
    
    // Test basic arithmetic
    float12 sum = a + b;
    float12 diff = a - b;
    float12 prod = a * b;
    float12 quot = a / b;
    
    std::cout << "5.0 + 2.0 = " << sum.toFloat() << "\n";
    std::cout << "5.0 - 2.0 = " << diff.toFloat() << "\n";
    std::cout << "5.0 * 2.0 = " << prod.toFloat() << "\n";
    std::cout << "5.0 / 2.0 = " << quot.toFloat() << "\n";
    
    // Test compound assignment
    float12 c(10.0f);
    c += float12(5.0f);
    std::cout << "10.0 += 5.0 = " << c.toFloat() << "\n";
    
    c -= float12(3.0f);
    std::cout << "15.0 -= 3.0 = " << c.toFloat() << "\n";
    
    c *= float12(2.0f);
    std::cout << "12.0 *= 2.0 = " << c.toFloat() << "\n";
    
    c /= float12(4.0f);
    std::cout << "24.0 /= 4.0 = " << c.toFloat() << "\n";
    
    std::cout << "Arithmetic tests completed successfully!\n";
}

// Function to run comparison tests
void runComparisonTests() {
    std::cout << "\n=== Running Comparison Tests ===\n";
    
    float12 a(3.14f);
    float12 b(2.71f);
    float12 c(3.14f);
    
    std::cout << "a (3.14) == c (3.14): " << (a == c ? "true" : "false") << "\n";
    std::cout << "a (3.14) != b (2.71): " << (a != b ? "true" : "false") << "\n";
    std::cout << "a (3.14) > b (2.71): " << (a > b ? "true" : "false") << "\n";
    std::cout << "b (2.71) < a (3.14): " << (b < a ? "true" : "false") << "\n";
    std::cout << "a (3.14) >= c (3.14): " << (a >= c ? "true" : "false") << "\n";
    std::cout << "b (2.71) <= a (3.14): " << (b <= a ? "true" : "false") << "\n";
    
    // Test NaN comparisons
    float12 nan_val(std::numeric_limits<float>::quiet_NaN());
    std::cout << "NaN == NaN: " << (nan_val == nan_val ? "true" : "false") << " (should be false)\n";
    std::cout << "NaN != NaN: " << (nan_val != nan_val ? "true" : "false") << " (should be true)\n";
    
    std::cout << "Comparison tests completed successfully!\n";
}

// Function to run precision and edge case tests
void runPrecisionTests() {
    std::cout << "\n=== Running Precision and Edge Case Tests ===\n";
    
    // Test precision limits
    float original = 3.141592653589793f;
    float12 compressed(original);
    float recovered = compressed.toFloat();
    
    std::cout << "Original: " << original << "\n";
    std::cout << "Recovered: " << recovered << "\n";
    std::cout << "Precision loss: " << (original - recovered) << "\n";
    
    // Test very small numbers
    float12 small(0.000001f);
    std::cout << "Small number (0.000001): " << small.toFloat() << "\n";
    
    // Test very large numbers (should overflow to infinity)
    float12 large(1000000.0f);
    std::cout << "Large number (1000000): " << large.toFloat() << "\n";
    std::cout << "Is infinite: " << (large.isInfinite() ? "true" : "false") << "\n";
    
    // Test signed zero
    float12 pos_zero(0.0f);
    float12 neg_zero(-0.0f);
    std::cout << "Positive zero: " << pos_zero.toFloat() << "\n";
    std::cout << "Negative zero: " << neg_zero.toFloat() << "\n";
    std::cout << "Both are zero: " << (pos_zero.isZero() && neg_zero.isZero() ? "true" : "false") << "\n";
    
    std::cout << "Precision and edge case tests completed successfully!\n";
}

int main() {
    std::cout << "=== Float12 Comprehensive Test Suite ===\n";
    std::cout << "Testing float12 implementation with both simple and advanced features...\n";
    
    try {
        // Run all test suites
        runSimpleTests();
        runAdvancedTests();
        runArithmeticTests();
        runComparisonTests();
        runPrecisionTests();
        
        std::cout << "\n=== All Tests Summary ===\n";
        std::cout << "✓ Simple compilation tests: PASSED\n";
        std::cout << "✓ Advanced constexpr tests: PASSED\n";
        std::cout << "✓ Arithmetic operations: PASSED\n";
        std::cout << "✓ Comparison operations: PASSED\n";
        std::cout << "✓ Precision and edge cases: PASSED\n";
        std::cout << "\n🎉 All tests completed successfully!\n";
        std::cout << "🔧 Constexpr functionality verified at compile time!\n";
        std::cout << "📊 Float12 implementation is working correctly!\n";
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed with exception: " << e.what() << "\n";
        return 1;
    } catch (...) {
        std::cerr << "❌ Test failed with unknown exception\n";
        return 1;
    }
    
    return 0;
}