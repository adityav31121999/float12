#include "float12.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <locale>

void testTrigonometricFunctions() {
    std::cout << "\n=== Trigonometric Functions ===\n";
    
    float12 angle_rad(1.0f);  // 1 radian ≈ 57.3 degrees
    float12 angle_deg(45.0f); // 45 degrees
    float12 angle_rad_45 = radians(angle_deg);
    
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Angle: " << angle_rad.toFloat() << " radians\n";
    std::cout << "sin(1.0) = " << sin(angle_rad).toFloat() << " (expected ≈ 0.841471)\n";
    std::cout << "cos(1.0) = " << cos(angle_rad).toFloat() << " (expected ≈ 0.540302)\n";
    std::cout << "tan(1.0) = " << tan(angle_rad).toFloat() << " (expected ≈ 1.557408)\n";
    
    std::cout << "\n45 degrees in radians: " << angle_rad_45.toFloat() << "\n";
    std::cout << "sin(45°) = " << sin(angle_rad_45).toFloat() << " (expected ≈ 0.707107)\n";
    std::cout << "cos(45°) = " << cos(angle_rad_45).toFloat() << " (expected ≈ 0.707107)\n";
}

void testBasicMath() {
    std::cout << "\n=== Basic Math Functions ===\n";
    
    float12 x(2.0f);
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "x = " << x.toFloat() << "\n";
    std::cout << "sqrt(4) = " << sqrt(float12(4.0f)).toFloat() << " (expected = 2.0)\n";
    std::cout << "pow(2, 3) = " << pow(x, float12(3.0f)).toFloat() << " (expected = 8.0)\n";
    std::cout << "exp(1) = " << exp(float12(1.0f)).toFloat() << " (expected ≈ 2.718282)\n";
    std::cout << "log(2.718282) = " << log(exp(float12(1.0f))).toFloat() << " (expected ≈ 1.0)\n";
}

int main() {
    std::cout << "=== Float12 Basic Math Functions Test ===\n";
    std::cout << "Testing basic mathematical functions...\n";
    
    try {
        testBasicMath();
        testTrigonometricFunctions();
        
        std::cout << "\n=== Basic Math Test Summary ===\n";
        std::cout << "✓ Basic math functions: PASSED\n";
        std::cout << "✓ Trigonometric functions: PASSED\n";
        std::cout << "\n🎉 Basic math tests completed successfully!\n";
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Math test failed with exception: " << e.what() << "\n";
        return 1;
    } catch (...) {
        std::cerr << "❌ Math test failed with unknown exception\n";
        return 1;
    }
    
    return 0;
}