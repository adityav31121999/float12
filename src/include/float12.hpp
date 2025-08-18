#ifndef FLOAT_12_HPP
#define FLOAT_12_HPP

#include <cstdint>
#include <cmath>
#include <limits>
#include <iostream>
#include <utility>

/**
 * @brief A definitive 12-bit float type (E5M6) with extended features.
 * @details Implements 1 sign, 5 exponent, 6 mantissa bits. Features safe
 *          type-punning, IEEE 754-style rounding, a rich API, and efficient,
 *          modern C++ idioms.
 */
class float12 {
private:
    uint16_t data;

    // --- Format Constants ---
    static constexpr int EXPONENT_BITS = 5;
    static constexpr int MANTISSA_BITS = 6;
    static constexpr int EXP_BIAS = 15;
    static constexpr uint16_t SIGN_MASK = 0x800;
    static constexpr uint16_t EXP_MASK = 0x7C0;
    static constexpr uint16_t MAN_MASK = 0x03F;
    static constexpr uint16_t INF_NAN_EXP = (1 << EXPONENT_BITS) - 1;

    // Union for safe, standard-compliant type-punning
    union Float32 { float f; uint32_t u; };

    // Core conversion logic
    static uint16_t to_float12(float f);
    float to_float() const;

    // Tag type for internal construction
    struct raw_bits_tag {};

public:
    // Internal constructor for constexpr static methods - DO NOT USE DIRECTLY
    constexpr float12(uint16_t raw_bits, raw_bits_tag) noexcept : data(raw_bits) {}

public:
    // --- Constructors ---
    constexpr float12() : data(0) {}
    explicit float12(float f) : data(to_float12(f)) {}
    explicit float12(double d) : data(to_float12(static_cast<float>(d))) {}
    explicit float12(int i) : data(to_float12(static_cast<float>(i))) {}
    explicit float12(long long int lli) : data(to_float12(static_cast<float>(lli))) {}
    float12(const float12&) = default;
    float12(float12&&) noexcept = default;

    // --- Destructor & Assignment ---
    ~float12() = default;
    float12& operator=(const float12&) = default;
    float12& operator=(float12&&) noexcept = default;
    float12& operator=(float f) { data = to_float12(f); return *this; }

    // --- Static Factory Methods ---
    static constexpr float12 fromRawBits(uint16_t bits) noexcept {
        return float12(bits & 0xFFF, raw_bits_tag{}); // Ensure only 12 bits are used
    }

    // --- Static Methods for Min/Max Values ---
    static constexpr float12 max() noexcept {
        // Sign=0, Exp=30 (11110), Mant=all 1s (111111) -> 0x7BF
        return float12(0x7BF, raw_bits_tag{});
    }
    static constexpr float12 lowest() noexcept {
        // Sign=1, Exp=30 (11110), Mant=all 1s (111111) -> 0xFBF
        return float12(0xFBF, raw_bits_tag{});
    }
    static constexpr float12 min() noexcept {
        // Sign=0, Exp=1 (00001), Mant=all 0s (000000) -> 0x040
        return float12(0x040, raw_bits_tag{});
    }

    // --- Static Utility Functions ---
    static bool isInRange(float f) {
        // Max value for E5M6 format is approximately 65504.0f
        constexpr float MAX_VALUE = 65504.0f;
        return std::abs(f) <= MAX_VALUE || f == 0.0f || std::isinf(f) || std::isnan(f);
    }

    static float12 fromFloatSafe(float f, bool* overflow = nullptr) {
        if (overflow) *overflow = false;
        if (!float12::isInRange(f) && std::isfinite(f)) {
            if (overflow) *overflow = true;
        }
        return float12(f);
    }

    // --- Conversion Operators to Other Types ---
    explicit operator float() const { return to_float(); }
    explicit operator double() const { return static_cast<double>(to_float()); }
    explicit operator int() const { return static_cast<int>(to_float()); }
    explicit operator long long int() const { return static_cast<long long int>(to_float()); }
    float toFloat() const { return to_float(); }

    // --- Unary & Compound Assignment Operators ---
    constexpr float12 operator-() const noexcept { 
        return float12(data ^ SIGN_MASK, raw_bits_tag{}); 
    }
    float12& operator+=(const float12& rhs) { *this = float12(to_float() + rhs.toFloat()); return *this; }
    float12& operator-=(const float12& rhs) { *this = float12(to_float() - rhs.toFloat()); return *this; }
    float12& operator*=(const float12& rhs) { *this = float12(to_float() * rhs.toFloat()); return *this; }
    float12& operator/=(const float12& rhs) { *this = float12(to_float() / rhs.toFloat()); return *this; }

    // --- Binary Arithmetic Operators ---
    friend float12 operator+(const float12& lhs, const float12& rhs) { 
        return float12(lhs.to_float() + rhs.to_float()); 
    }
    friend float12 operator-(const float12& lhs, const float12& rhs) { 
        return float12(lhs.to_float() - rhs.to_float()); 
    }
    friend float12 operator*(const float12& lhs, const float12& rhs) { 
        return float12(lhs.to_float() * rhs.to_float()); 
    }
    friend float12 operator/(const float12& lhs, const float12& rhs) { 
        return float12(lhs.to_float() / rhs.to_float()); 
    }

    // --- Comparison Operators ---
    friend bool operator==(const float12& lhs, const float12& rhs) noexcept {
        return lhs.to_float() == rhs.to_float();
    }
    friend bool operator!=(const float12& lhs, const float12& rhs) noexcept {
        return !(lhs == rhs);
    }
    friend bool operator<(const float12& lhs, const float12& rhs) noexcept {
        return lhs.to_float() < rhs.to_float();
    }
    friend bool operator<=(const float12& lhs, const float12& rhs) noexcept {
        return lhs.to_float() <= rhs.to_float();
    }
    friend bool operator>(const float12& lhs, const float12& rhs) noexcept {
        return lhs.to_float() > rhs.to_float();
    }
    friend bool operator>=(const float12& lhs, const float12& rhs) noexcept {
        return lhs.to_float() >= rhs.to_float();
    }

    // --- Utility Functions ---
    constexpr bool isZero() const noexcept { return (data & ~SIGN_MASK) == 0; }
    constexpr bool isNegative() const noexcept { return (data & SIGN_MASK) != 0; }
    constexpr bool isInfinite() const noexcept { return (data & EXP_MASK) == EXP_MASK && (data & MAN_MASK) == 0; }
    constexpr bool isNaN() const noexcept { return (data & EXP_MASK) == EXP_MASK && (data & MAN_MASK) != 0; }
    constexpr bool isFinite() const noexcept { return !isInfinite() && !isNaN(); }
    constexpr bool isNormal() const noexcept { 
        uint16_t exp = (data & EXP_MASK) >> MANTISSA_BITS;
        return exp != 0 && exp != INF_NAN_EXP; 
    }

    // --- Raw Bit Access ---
    constexpr uint16_t getRawBits() const noexcept { return data; }
    void setRawBits(uint16_t bits) noexcept { data = bits & 0xFFF; }
};

// --- Mathematical Functions ---
// Trigonometric functions
float12 sin(const float12& x);
float12 cos(const float12& x);
float12 tan(const float12& x);
float12 asin(const float12& x);
float12 acos(const float12& x);
float12 atan(const float12& x);
float12 atan2(const float12& y, const float12& x);

// Hyperbolic functions
float12 sinh(const float12& x);
float12 cosh(const float12& x);
float12 tanh(const float12& x);
float12 asinh(const float12& x);
float12 acosh(const float12& x);
float12 atanh(const float12& x);

// Exponential and logarithmic functions
float12 exp(const float12& x);
float12 exp2(const float12& x);
float12 expm1(const float12& x);
float12 log(const float12& x);
float12 log10(const float12& x);
float12 log2(const float12& x);
float12 log1p(const float12& x);

// Power functions
float12 pow(const float12& base, const float12& exponent);
float12 sqrt(const float12& x);
float12 cbrt(const float12& x);
float12 hypot(const float12& x, const float12& y);
float12 hypot(const float12& x, const float12& y, const float12& z);

// Rounding and remainder functions
float12 ceil(const float12& x);
float12 floor(const float12& x);
float12 trunc(const float12& x);
float12 round(const float12& x);
float12 nearbyint(const float12& x);
float12 rint(const float12& x);
float12 fmod(const float12& x, const float12& y);
float12 remainder(const float12& x, const float12& y);
float12 remquo(const float12& x, const float12& y, int* quo);

// Floating-point manipulation functions
float12 copysign(const float12& mag, const float12& sgn);
float12 nextafter(const float12& from, const float12& to);
float12 nexttoward(const float12& from, long double to);

// Minimum, maximum, difference functions
float12 fmin(const float12& x, const float12& y);
float12 fmax(const float12& x, const float12& y);
float12 fdim(const float12& x, const float12& y);

// Fused multiply-add
float12 fma(const float12& x, const float12& y, const float12& z);

// Classification functions
bool isfinite(const float12& x);
bool isinf(const float12& x);
bool isnan(const float12& x);
bool isnormal(const float12& x);
bool signbit(const float12& x);

// Comparison functions
bool isgreater(const float12& x, const float12& y);
bool isgreaterequal(const float12& x, const float12& y);
bool isless(const float12& x, const float12& y);
bool islessequal(const float12& x, const float12& y);
bool islessgreater(const float12& x, const float12& y);
bool isunordered(const float12& x, const float12& y);

// Error and gamma functions
float12 erf(const float12& x);
float12 erfc(const float12& x);
float12 tgamma(const float12& x);
float12 lgamma(const float12& x);

// Bessel functions (GNU extension)
#ifdef _GNU_SOURCE
float12 j0(const float12& x);
float12 j1(const float12& x);
float12 jn(int n, const float12& x);
float12 y0(const float12& x);
float12 y1(const float12& x);
float12 yn(int n, const float12& x);
#endif

// Additional utility functions
float12 abs(const float12& x);
float12 fabs(const float12& x);

// Degree/radian conversion utilities
float12 degrees(const float12& radians);
float12 radians(const float12& degrees);

// Graphics/interpolation functions
float12 lerp(const float12& a, const float12& b, const float12& t);
float12 clamp(const float12& value, const float12& min_val, const float12& max_val);
float12 smoothstep(const float12& edge0, const float12& edge1, const float12& x);
float12 mix(const float12& a, const float12& b, const float12& t);
float12 step(const float12& edge, const float12& x);

// --- Mathematical Constants ---
namespace float12_constants {
    extern const float12 PI;
    extern const float12 E;
    extern const float12 SQRT2;
    extern const float12 SQRT3;
    extern const float12 LN2;
    extern const float12 LN10;
    extern const float12 LOG2E;
    extern const float12 LOG10E;
    extern const float12 PI_2;      // π/2
    extern const float12 PI_4;      // π/4
    extern const float12 TWO_PI;    // 2π
    extern const float12 INV_PI;    // 1/π
    extern const float12 INV_SQRT2; // 1/√2
    extern const float12 PHI;       // Golden ratio
    extern const float12 DEG_TO_RAD;
    extern const float12 RAD_TO_DEG;
}

// --- Specialized Mathematical Functions ---
namespace float12_math {
    // Fast approximations
    float12 fast_sin(const float12& x);
    float12 fast_cos(const float12& x);
    
    // Distance functions
    float12 distance(const float12& x1, const float12& y1, const float12& x2, const float12& y2);
    float12 distance3d(const float12& x1, const float12& y1, const float12& z1,
                       const float12& x2, const float12& y2, const float12& z2);
    
    // Angle utilities
    float12 normalize_angle(const float12& angle);
    float12 normalize_angle_signed(const float12& angle);
    
    // Combinatorics
    float12 factorial(int n);
    float12 binomial(int n, int k);
    
    // Activation functions (neural networks)
    float12 sigmoid(const float12& x);
    float12 tanh_alt(const float12& x);
    float12 relu(const float12& x);
    float12 leaky_relu(const float12& x, const float12& alpha = float12(0.01f));
    float12 softplus(const float12& x);
    
    // Statistical functions
    float12 gaussian_pdf(const float12& x, const float12& mean = float12(0.0f), 
                         const float12& std_dev = float12(1.0f));
    
    // Utility functions
    float12 map_range(const float12& value, const float12& from_min, const float12& from_max,
                      const float12& to_min, const float12& to_max);
    float12 wrap(const float12& value, const float12& min_val, const float12& max_val);
    float12 ping_pong(const float12& t, const float12& length);
    float12 inv_sqrt(const float12& x);
    float12 safe_divide(const float12& numerator, const float12& denominator, 
                        const float12& default_value = float12(0.0f));
    
    // Comparison utilities
    bool approximately_equal(const float12& a, const float12& b, 
                           const float12& epsilon = float12(1e-5f));
    
    // Advanced interpolation
    float12 hermite(const float12& t);
    float12 quintic(const float12& t);
}

// --- LLM Training Support ---
namespace float12_llm {
    // Advanced activation functions for transformers
    float12 gelu(const float12& x);
    float12 swish(const float12& x);
    float12 mish(const float12& x);
    
    // Attention mechanism support
    void softmax_inplace(float12* logits, size_t length);
    float12 apply_temperature(const float12& logit, float temperature);
    
    // Gradient operations
    float12 clip_gradients_by_norm(float12* gradients, size_t count, float12 max_norm);
    bool accumulate_gradient(float12& accumulated, const float12& new_grad, int step_count);
    
    // Loss scaling for mixed precision training
    class LossScaler {
    private:
        float current_scale;
        int growth_interval;
        int growth_counter;
        float growth_factor;
        float backoff_factor;
        
    public:
        LossScaler(float initial_scale = 65536.0f, int interval = 2000, 
                  float growth = 2.0f, float backoff = 0.5f);
        
        float12 scale_loss(const float12& loss);
        float12 unscale_gradient(const float12& gradient);
        bool update_scale(bool overflow_detected);
        float get_scale() const;
    };
    
    // Quantization support
    float12 quantize_symmetric(float value, float scale);
    float calculate_symmetric_scale(const float* data, size_t count);
    
    // Optimizer support
    struct AdamState {
        float12 m; // First moment
        float12 v; // Second moment
        int step;
        AdamState();
    };
    
    float12 adam_update(AdamState& state, const float12& gradient, 
                       float learning_rate = 0.001f, float beta1 = 0.9f, 
                       float beta2 = 0.999f, float epsilon = 1e-8f);
    
    // Numerical stability utilities
    float12 log_sum_exp(const float12* values, size_t count);
    float12 safe_divide_with_gradient(const float12& numerator, const float12& denominator, 
                                     float12 epsilon = float12(1e-8f));
    
    // Memory efficient operations
    void layer_norm_inplace(float12* data, size_t length, const float12& gamma, const float12& beta);
    
    // Debugging and monitoring
    struct TensorStats {
        size_t nan_count;
        size_t inf_count;
        size_t zero_count;
        float12 min_val;
        float12 max_val;
        float12 mean;
        float12 std_dev;
    };
    
    TensorStats analyze_tensor(const float12* data, size_t count);
}

#endif // FLOAT_12_HPP