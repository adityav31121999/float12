#include "float12.hpp"
#include <cmath>

// Trigonometric functions
float12 sin(const float12& x) {
    return float12(std::sin(float(x)));
}

float12 cos(const float12& x) {
    return float12(std::cos(float(x)));
}

float12 tan(const float12& x) {
    return float12(std::tan(float(x)));
}

// Inverse trigonometric functions
float12 asin(const float12& x) {
    return float12(std::asin(float(x)));
}

float12 acos(const float12& x) {
    return float12(std::acos(float(x)));
}

float12 atan(const float12& x) {
    return float12(std::atan(float(x)));
}

float12 atan2(const float12& y, const float12& x) {
    return float12(std::atan2(float(y), float(x)));
}

// Hyperbolic functions
float12 sinh(const float12& x) {
    return float12(std::sinh(float(x)));
}

float12 cosh(const float12& x) {
    return float12(std::cosh(float(x)));
}

float12 tanh(const float12& x) {
    return float12(std::tanh(float(x)));
}

// Inverse hyperbolic functions
float12 asinh(const float12& x) {
    return float12(std::asinh(float(x)));
}

float12 acosh(const float12& x) {
    return float12(std::acosh(float(x)));
}

float12 atanh(const float12& x) {
    return float12(std::atanh(float(x)));
}

// Exponential and logarithmic functions
float12 exp(const float12& x) {
    return float12(std::exp(float(x)));
}

float12 exp2(const float12& x) {
    return float12(std::exp2(float(x)));
}

float12 expm1(const float12& x) {
    return float12(std::expm1(float(x)));
}

float12 log(const float12& x) {
    return float12(std::log(float(x)));
}

float12 log10(const float12& x) {
    return float12(std::log10(float(x)));
}

float12 log2(const float12& x) {
    return float12(std::log2(float(x)));
}

float12 log1p(const float12& x) {
    return float12(std::log1p(float(x)));
}

// Power functions
float12 pow(const float12& base, const float12& exponent) {
    return float12(std::pow(float(base), float(exponent)));
}

float12 sqrt(const float12& x) {
    return float12(std::sqrt(float(x)));
}

float12 cbrt(const float12& x) {
    return float12(std::cbrt(float(x)));
}

float12 hypot(const float12& x, const float12& y) {
    return float12(std::hypot(float(x), float(y)));
}

float12 hypot(const float12& x, const float12& y, const float12& z) {
    return float12(std::hypot(float(x), float(y), float(z)));
}

// Rounding and remainder functions
float12 ceil(const float12& x) {
    return float12(std::ceil(float(x)));
}

float12 floor(const float12& x) {
    return float12(std::floor(float(x)));
}

float12 trunc(const float12& x) {
    return float12(std::trunc(float(x)));
}

float12 round(const float12& x) {
    return float12(std::round(float(x)));
}

float12 nearbyint(const float12& x) {
    return float12(std::nearbyint(float(x)));
}

float12 rint(const float12& x) {
    return float12(std::rint(float(x)));
}

float12 fmod(const float12& x, const float12& y) {
    return float12(std::fmod(float(x), float(y)));
}

float12 remainder(const float12& x, const float12& y) {
    return float12(std::remainder(float(x), float(y)));
}

float12 remquo(const float12& x, const float12& y, int* quo) {
    return float12(std::remquo(float(x), float(y), quo));
}

// Floating-point manipulation functions
float12 copysign(const float12& mag, const float12& sgn) {
    return float12(std::copysign(float(mag), float(sgn)));
}

float12 nextafter(const float12& from, const float12& to) {
    return float12(std::nextafter(float(from), float(to)));
}

float12 nexttoward(const float12& from, long double to) {
    return float12(std::nexttoward(float(from), to));
}

// Minimum, maximum, difference functions
float12 fmin(const float12& x, const float12& y) {
    return float12(std::fmin(float(x), float(y)));
}

float12 fmax(const float12& x, const float12& y) {
    return float12(std::fmax(float(x), float(y)));
}

float12 fdim(const float12& x, const float12& y) {
    return float12(std::fdim(float(x), float(y)));
}

// Fused multiply-add
float12 fma(const float12& x, const float12& y, const float12& z) {
    return float12(std::fma(float(x), float(y), float(z)));
}

// Classification functions (return bool, not float12)
bool isfinite(const float12& x) {
    return x.isFinite();
}

bool isinf(const float12& x) {
    return x.isInfinite();
}

bool isnan(const float12& x) {
    return x.isNaN();
}

bool isnormal(const float12& x) {
    return x.isNormal();
}

bool signbit(const float12& x) {
    return x.isNegative();
}

// Comparison functions
bool isgreater(const float12& x, const float12& y) {
    return !isnan(x) && !isnan(y) && (x > y);
}

bool isgreaterequal(const float12& x, const float12& y) {
    return !isnan(x) && !isnan(y) && (x >= y);
}

bool isless(const float12& x, const float12& y) {
    return !isnan(x) && !isnan(y) && (x < y);
}

bool islessequal(const float12& x, const float12& y) {
    return !isnan(x) && !isnan(y) && (x <= y);
}

bool islessgreater(const float12& x, const float12& y) {
    return !isnan(x) && !isnan(y) && (x != y);
}

bool isunordered(const float12& x, const float12& y) {
    return isnan(x) || isnan(y);
}

// Error and gamma functions
float12 erf(const float12& x) {
    return float12(std::erf(float(x)));
}

float12 erfc(const float12& x) {
    return float12(std::erfc(float(x)));
}

float12 tgamma(const float12& x) {
    return float12(std::tgamma(float(x)));
}

float12 lgamma(const float12& x) {
    return float12(std::lgamma(float(x)));
}

// Bessel functions (if available)
#ifdef _GNU_SOURCE
float12 j0(const float12& x) {
    return float12(::j0(float(x)));
}

float12 j1(const float12& x) {
    return float12(::j1(float(x)));
}

float12 jn(int n, const float12& x) {
    return float12(::jn(n, float(x)));
}

float12 y0(const float12& x) {
    return float12(::y0(float(x)));
}

float12 y1(const float12& x) {
    return float12(::y1(float(x)));
}

float12 yn(int n, const float12& x) {
    return float12(::yn(n, float(x)));
}
#endif

// Additional utility functions
float12 abs(const float12& x) {
    return x.isNegative() ? -x : x;
}

float12 fabs(const float12& x) {
    return abs(x);
}

// Degree/radian conversion utilities
float12 degrees(const float12& radians) {
    constexpr float PI = 3.14159265358979323846f;
    return radians * float12(180.0f / PI);
}

float12 radians(const float12& degrees) {
    constexpr float PI = 3.14159265358979323846f;
    return degrees * float12(PI / 180.0f);
}

// Linear interpolation
float12 lerp(const float12& a, const float12& b, const float12& t) {
    return a + t * (b - a);
}

// Clamp function
float12 clamp(const float12& value, const float12& min_val, const float12& max_val) {
    return fmin(fmax(value, min_val), max_val);
}

// Smoothstep function (useful for graphics)
float12 smoothstep(const float12& edge0, const float12& edge1, const float12& x) {
    float12 t = clamp((x - edge0) / (edge1 - edge0), float12(0.0f), float12(1.0f));
    return t * t * (float12(3.0f) - float12(2.0f) * t);
}

// Mix function (alternative name for lerp)
float12 mix(const float12& a, const float12& b, const float12& t) {
    return lerp(a, b, t);
}

// Step function
float12 step(const float12& edge, const float12& x) {
    return x < edge ? float12(0.0f) : float12(1.0f);
}