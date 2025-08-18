# Float12 Mathematical Functions Documentation

This document provides comprehensive documentation for the mathematical functions available in the float12 library.

## Overview

The float12 library now includes extensive mathematical function support, making it suitable for:
- Scientific computing
- Graphics and game development
- Machine learning and neural networks
- Signal processing
- Statistical analysis

## Core Mathematical Functions

### Trigonometric Functions
```cpp
float12 sin(const float12& x);      // Sine
float12 cos(const float12& x);      // Cosine
float12 tan(const float12& x);      // Tangent
float12 asin(const float12& x);     // Arcsine
float12 acos(const float12& x);     // Arccosine
float12 atan(const float12& x);     // Arctangent
float12 atan2(const float12& y, const float12& x); // Two-argument arctangent
```

**Example:**
```cpp
float12 angle = float12_constants::PI_4; // 45 degrees in radians
float12 sine_val = sin(angle);           // ≈ 0.707107
float12 cosine_val = cos(angle);         // ≈ 0.707107
```

### Hyperbolic Functions
```cpp
float12 sinh(const float12& x);     // Hyperbolic sine
float12 cosh(const float12& x);     // Hyperbolic cosine
float12 tanh(const float12& x);     // Hyperbolic tangent
float12 asinh(const float12& x);    // Inverse hyperbolic sine
float12 acosh(const float12& x);    // Inverse hyperbolic cosine
float12 atanh(const float12& x);    // Inverse hyperbolic tangent
```

### Exponential and Logarithmic Functions
```cpp
float12 exp(const float12& x);      // e^x
float12 exp2(const float12& x);     // 2^x
float12 expm1(const float12& x);    // e^x - 1 (more accurate for small x)
float12 log(const float12& x);      // Natural logarithm
float12 log10(const float12& x);    // Base-10 logarithm
float12 log2(const float12& x);     // Base-2 logarithm
float12 log1p(const float12& x);    // ln(1 + x) (more accurate for small x)
```

### Power Functions
```cpp
float12 pow(const float12& base, const float12& exponent); // base^exponent
float12 sqrt(const float12& x);     // Square root
float12 cbrt(const float12& x);     // Cube root
float12 hypot(const float12& x, const float12& y);         // √(x² + y²)
float12 hypot(const float12& x, const float12& y, const float12& z); // √(x² + y² + z²)
```

### Rounding and Remainder Functions
```cpp
float12 ceil(const float12& x);     // Ceiling (round up)
float12 floor(const float12& x);    // Floor (round down)
float12 trunc(const float12& x);    // Truncate (round toward zero)
float12 round(const float12& x);    // Round to nearest integer
float12 nearbyint(const float12& x); // Round using current rounding mode
float12 rint(const float12& x);     // Round to nearest integer
float12 fmod(const float12& x, const float12& y);      // Floating-point remainder
float12 remainder(const float12& x, const float12& y); // IEEE remainder
```

### Min/Max and Utility Functions
```cpp
float12 fmin(const float12& x, const float12& y);  // Minimum
float12 fmax(const float12& x, const float12& y);  // Maximum
float12 fdim(const float12& x, const float12& y);  // Positive difference
float12 abs(const float12& x);                     // Absolute value
float12 fabs(const float12& x);                    // Absolute value (alias)
float12 copysign(const float12& mag, const float12& sgn); // Copy sign
float12 fma(const float12& x, const float12& y, const float12& z); // x*y + z
```

## Mathematical Constants

The `float12_constants` namespace provides essential mathematical constants:

```cpp
namespace float12_constants {
    extern const float12 PI;         // π ≈ 3.14159
    extern const float12 E;          // e ≈ 2.71828
    extern const float12 SQRT2;      // √2 ≈ 1.41421
    extern const float12 SQRT3;      // √3 ≈ 1.73205
    extern const float12 LN2;        // ln(2) ≈ 0.69315
    extern const float12 LN10;       // ln(10) ≈ 2.30259
    extern const float12 LOG2E;      // log₂(e) ≈ 1.44270
    extern const float12 LOG10E;     // log₁₀(e) ≈ 0.43429
    extern const float12 PI_2;       // π/2
    extern const float12 PI_4;       // π/4
    extern const float12 TWO_PI;     // 2π
    extern const float12 INV_PI;     // 1/π
    extern const float12 INV_SQRT2;  // 1/√2
    extern const float12 PHI;        // Golden ratio ≈ 1.61803
    extern const float12 DEG_TO_RAD; // π/180 (degrees to radians)
    extern const float12 RAD_TO_DEG; // 180/π (radians to degrees)
}
```

**Example:**
```cpp
float12 circle_area = float12_constants::PI * radius * radius;
float12 angle_rad = degrees * float12_constants::DEG_TO_RAD;
```

## Specialized Mathematical Functions

The `float12_math` namespace provides specialized functions for various applications:

### Fast Approximations
```cpp
float12 fast_sin(const float12& x);  // Fast sine approximation
float12 fast_cos(const float12& x);  // Fast cosine approximation
```
These provide faster computation at the cost of some accuracy, useful for real-time applications.

### Distance Functions
```cpp
float12 distance(const float12& x1, const float12& y1, 
                 const float12& x2, const float12& y2);
float12 distance3d(const float12& x1, const float12& y1, const float12& z1,
                   const float12& x2, const float12& y2, const float12& z2);
```

**Example:**
```cpp
float12 dist = float12_math::distance(float12(0), float12(0), 
                                      float12(3), float12(4)); // = 5.0
```

### Angle Utilities
```cpp
float12 normalize_angle(const float12& angle);        // Normalize to [0, 2π)
float12 normalize_angle_signed(const float12& angle); // Normalize to [-π, π)
```

### Combinatorial Functions
```cpp
float12 factorial(int n);           // n!
float12 binomial(int n, int k);     // C(n,k) = n!/(k!(n-k)!)
```

### Neural Network Activation Functions
```cpp
float12 sigmoid(const float12& x);                    // 1/(1 + e^(-x))
float12 relu(const float12& x);                       // max(0, x)
float12 leaky_relu(const float12& x, const float12& alpha = 0.01f); // x > 0 ? x : α*x
float12 softplus(const float12& x);                   // ln(1 + e^x)
float12 tanh_alt(const float12& x);                   // Alternative tanh implementation
```

**Example:**
```cpp
float12 input(-2.0f);
float12 activated = float12_math::sigmoid(input); // ≈ 0.119203
```

### Statistical Functions
```cpp
float12 gaussian_pdf(const float12& x, 
                     const float12& mean = 0.0f, 
                     const float12& std_dev = 1.0f);
```

### Graphics and Interpolation Functions
```cpp
float12 lerp(const float12& a, const float12& b, const float12& t);
float12 mix(const float12& a, const float12& b, const float12& t);  // Alias for lerp
float12 clamp(const float12& value, const float12& min_val, const float12& max_val);
float12 smoothstep(const float12& edge0, const float12& edge1, const float12& x);
float12 step(const float12& edge, const float12& x);
float12 hermite(const float12& t);    // Hermite interpolation
float12 quintic(const float12& t);    // Quintic interpolation
```

### Utility Functions
```cpp
float12 map_range(const float12& value, 
                  const float12& from_min, const float12& from_max,
                  const float12& to_min, const float12& to_max);
float12 wrap(const float12& value, const float12& min_val, const float12& max_val);
float12 ping_pong(const float12& t, const float12& length);
float12 safe_divide(const float12& numerator, const float12& denominator, 
                    const float12& default_value = 0.0f);
bool approximately_equal(const float12& a, const float12& b, 
                        const float12& epsilon = 1e-5f);
```

## Classification Functions

These functions test properties of float12 values:

```cpp
bool isfinite(const float12& x);     // Is finite (not infinite or NaN)
bool isinf(const float12& x);        // Is infinite
bool isnan(const float12& x);        // Is NaN (Not a Number)
bool isnormal(const float12& x);     // Is normal (not zero, subnormal, infinite, or NaN)
bool signbit(const float12& x);      // Is negative (including -0)
```

## Comparison Functions

IEEE 754-compliant comparison functions:

```cpp
bool isgreater(const float12& x, const float12& y);
bool isgreaterequal(const float12& x, const float12& y);
bool isless(const float12& x, const float12& y);
bool islessequal(const float12& x, const float12& y);
bool islessgreater(const float12& x, const float12& y);
bool isunordered(const float12& x, const float12& y);
```

## Error and Gamma Functions

```cpp
float12 erf(const float12& x);       // Error function
float12 erfc(const float12& x);      // Complementary error function
float12 tgamma(const float12& x);    // Gamma function
float12 lgamma(const float12& x);    // Log gamma function
```

## Usage Examples

### Scientific Computing
```cpp
// Calculate the standard deviation
float12 mean = float12(5.0f);
float12 variance = float12(2.0f);
float12 std_dev = sqrt(variance);

// Normal distribution probability
float12 x = float12(6.0f);
float12 prob = float12_math::gaussian_pdf(x, mean, std_dev);
```

### Graphics Programming
```cpp
// Smooth animation curve
float12 t = float12(0.3f); // Animation parameter [0,1]
float12 smooth_t = float12_math::hermite(t);

// Color interpolation
float12 red1 = float12(1.0f), red2 = float12(0.5f);
float12 interpolated_red = lerp(red1, red2, smooth_t);

// Distance-based effects
float12 distance = float12_math::distance(player_x, player_y, enemy_x, enemy_y);
float12 effect_strength = float12(1.0f) / (float12(1.0f) + distance);
```

### Machine Learning
```cpp
// Neural network forward pass
float12 weighted_sum = float12(0.0f);
// ... calculate weighted sum ...
float12 activation = float12_math::sigmoid(weighted_sum);

// ReLU activation with leaky behavior
float12 leaky_output = float12_math::leaky_relu(input, float12(0.01f));
```

### Signal Processing
```cpp
// Generate sine wave
float12 frequency = float12(440.0f); // A4 note
float12 sample_rate = float12(44100.0f);
float12 time = float12(0.001f); // 1ms

float12 phase = float12_constants::TWO_PI * frequency * time;
float12 amplitude = sin(phase);
```

## Performance Considerations

1. **Fast Approximations**: Use `fast_sin()` and `fast_cos()` for real-time applications where speed is more important than precision.

2. **Constants**: Use predefined constants from `float12_constants` instead of computing them repeatedly.

3. **Specialized Functions**: Functions in `float12_math` namespace are optimized for specific use cases.

4. **Memory Usage**: All functions maintain the 12-bit precision of float12, providing memory efficiency while supporting complex mathematical operations.

## Building and Linking

Make sure to include all source files in your build:

```cmake
add_library(float12 STATIC
    operation.cpp
    convert.cpp
    error.cpp
    math.cpp
    constants.cpp
)
```

Include the header in your code:
```cpp
#include "src/include/float12.hpp"
```

All mathematical functions are now available for use with your float12 values, providing a complete mathematical computing environment with reduced precision requirements.