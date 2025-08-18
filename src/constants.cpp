#include "float12.hpp"

// Mathematical constants for float12
namespace float12_constants {
    // Fundamental constants
    const float12 PI = float12(3.14159265358979323846f);
    const float12 E = float12(2.71828182845904523536f);
    const float12 SQRT2 = float12(1.41421356237309504880f);
    const float12 SQRT3 = float12(1.73205080756887729352f);
    const float12 LN2 = float12(0.69314718055994530942f);
    const float12 LN10 = float12(2.30258509299404568402f);
    const float12 LOG2E = float12(1.44269504088896340736f);
    const float12 LOG10E = float12(0.43429448190325182765f);
    
    // Derived constants (computed as float literals to avoid initialization order issues)
    const float12 PI_2 = float12(1.57079632679489661923f);        // π/2
    const float12 PI_4 = float12(0.78539816339744830962f);        // π/4
    const float12 TWO_PI = float12(6.28318530717958647692f);      // 2π
    const float12 INV_PI = float12(0.31830988618379067154f);      // 1/π
    const float12 INV_SQRT2 = float12(0.70710678118654752440f);   // 1/√2
    
    // Golden ratio
    const float12 PHI = float12(1.61803398874989484820f);
    
    // Conversion factors
    const float12 DEG_TO_RAD = float12(0.01745329251994329577f);  // π/180
    const float12 RAD_TO_DEG = float12(57.29577951308232087680f); // 180/π
}

// Specialized mathematical functions
namespace float12_math {
    
    // Fast approximations for common functions (useful for graphics)
    float12 fast_sin(const float12& x) {
        // Simple polynomial approximation for sin(x) in [-π, π]
        float12 x_norm = fmod(x, float12_constants::TWO_PI);
        if (x_norm > float12_constants::PI) {
            x_norm = x_norm - float12_constants::TWO_PI;
        }
        
        // Taylor series approximation: sin(x) ≈ x - x³/6 + x⁵/120
        float12 x2 = x_norm * x_norm;
        float12 x3 = x2 * x_norm;
        float12 x5 = x3 * x2;
        
        return x_norm - x3 / float12(6.0f) + x5 / float12(120.0f);
    }
    
    float12 fast_cos(const float12& x) {
        // cos(x) = sin(π/2 - x)
        return fast_sin(float12_constants::PI_2 - x);
    }
    
    // Distance functions
    float12 distance(const float12& x1, const float12& y1, const float12& x2, const float12& y2) {
        float12 dx = x2 - x1;
        float12 dy = y2 - y1;
        return sqrt(dx * dx + dy * dy);
    }
    
    float12 distance3d(const float12& x1, const float12& y1, const float12& z1,
                       const float12& x2, const float12& y2, const float12& z2) {
        float12 dx = x2 - x1;
        float12 dy = y2 - y1;
        float12 dz = z2 - z1;
        return sqrt(dx * dx + dy * dy + dz * dz);
    }
    
    // Normalize angle to [0, 2π)
    float12 normalize_angle(const float12& angle) {
        float12 result = fmod(angle, float12_constants::TWO_PI);
        if (result < float12(0.0f)) {
            result = result + float12_constants::TWO_PI;
        }
        return result;
    }
    
    // Normalize angle to [-π, π)
    float12 normalize_angle_signed(const float12& angle) {
        float12 result = normalize_angle(angle);
        if (result > float12_constants::PI) {
            result = result - float12_constants::TWO_PI;
        }
        return result;
    }
    
    // Factorial (for small integers)
    float12 factorial(int n) {
        if (n < 0) return float12(0.0f);
        if (n == 0 || n == 1) return float12(1.0f);
        
        float12 result(1.0f);
        for (int i = 2; i <= n; ++i) {
            result *= float12(static_cast<float>(i));
        }
        return result;
    }
    
    // Binomial coefficient C(n, k)
    float12 binomial(int n, int k) {
        if (k > n || k < 0) return float12(0.0f);
        if (k == 0 || k == n) return float12(1.0f);
        
        // Use the more efficient formula: C(n,k) = n! / (k! * (n-k)!)
        // But compute it iteratively to avoid large factorials
        float12 result(1.0f);
        for (int i = 0; i < k; ++i) {
            result *= float12(static_cast<float>(n - i));
            result /= float12(static_cast<float>(i + 1));
        }
        return result;
    }
    
    // Sigmoid function (useful for neural networks)
    float12 sigmoid(const float12& x) {
        return float12(1.0f) / (float12(1.0f) + exp(-x));
    }
    
    // Hyperbolic tangent (alternative implementation)
    float12 tanh_alt(const float12& x) {
        float12 exp_2x = exp(float12(2.0f) * x);
        return (exp_2x - float12(1.0f)) / (exp_2x + float12(1.0f));
    }
    
    // ReLU activation function
    float12 relu(const float12& x) {
        return fmax(float12(0.0f), x);
    }
    
    // Leaky ReLU activation function
    float12 leaky_relu(const float12& x, const float12& alpha) {
        return x > float12(0.0f) ? x : alpha * x;
    }
    
    // Softplus activation function
    float12 softplus(const float12& x) {
        return log(float12(1.0f) + exp(x));
    }
    
    // Gaussian/Normal distribution probability density function
    float12 gaussian_pdf(const float12& x, const float12& mean, 
                         const float12& std_dev) {
        float12 variance = std_dev * std_dev;
        float12 diff = x - mean;
        float12 exponent = -(diff * diff) / (float12(2.0f) * variance);
        float12 coefficient = float12(1.0f) / (std_dev * sqrt(float12_constants::TWO_PI));
        return coefficient * exp(exponent);
    }
    
    // Linear map from one range to another
    float12 map_range(const float12& value, const float12& from_min, const float12& from_max,
                      const float12& to_min, const float12& to_max) {
        float12 from_range = from_max - from_min;
        float12 to_range = to_max - to_min;
        float12 normalized = (value - from_min) / from_range;
        return to_min + normalized * to_range;
    }
    
    // Wrap value to range [min, max)
    float12 wrap(const float12& value, const float12& min_val, const float12& max_val) {
        float12 range = max_val - min_val;
        float12 result = value - min_val;
        result = result - floor(result / range) * range;
        return result + min_val;
    }
    
    // Ping-pong between 0 and length
    float12 ping_pong(const float12& t, const float12& length) {
        float12 t_mod = fmod(t, length * float12(2.0f));
        return length - abs(t_mod - length);
    }
    
    // Inverse square root (fast approximation)
    float12 inv_sqrt(const float12& x) {
        return float12(1.0f) / sqrt(x);
    }
    
    // Safe division (returns 0 if denominator is 0)
    float12 safe_divide(const float12& numerator, const float12& denominator, 
                        const float12& default_value) {
        return abs(denominator) < float12(1e-6f) ? default_value : numerator / denominator;
    }
    
    // Check if two float12 values are approximately equal
    bool approximately_equal(const float12& a, const float12& b, 
                           const float12& epsilon) {
        return abs(a - b) < epsilon;
    }
    
    // Hermite interpolation (smooth curve between two points)
    float12 hermite(const float12& t) {
        return t * t * (float12(3.0f) - float12(2.0f) * t);
    }
    
    // Quintic interpolation (even smoother)
    float12 quintic(const float12& t) {
        return t * t * t * (t * (t * float12(6.0f) - float12(15.0f)) + float12(10.0f));
    }
}