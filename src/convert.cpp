
#include "float12.hpp"


// --- Conversion Implementation ---
uint16_t float12::to_float12(float f) {
    if (std::isnan(f)) return EXP_MASK | 1; // NaN: exponent all 1s, mantissa non-zero
    if (std::isinf(f)) return (f < 0.0f ? SIGN_MASK : 0) | EXP_MASK;
    if (f == 0.0f) return (std::signbit(f) ? SIGN_MASK : 0);

    Float32 conv{f};
    uint32_t u = conv.u;
    uint16_t sign = (u >> 16) & SIGN_MASK;
    int32_t exp = ((u >> 23) & 0xFF) - 127;
    uint32_t man = u & 0x7FFFFF;

    int32_t new_exp = exp + EXP_BIAS;

    if (new_exp >= INF_NAN_EXP) return sign | EXP_MASK; // Overflow to infinity
    if (new_exp <= 0) return sign; // Underflow and subnormals flush to zero

    // Round to nearest, ties to even
    uint16_t new_man = man >> (23 - MANTISSA_BITS);
    uint32_t frac_bits = man & ((1 << (23 - MANTISSA_BITS)) - 1);
    uint32_t halfway = 1 << (23 - MANTISSA_BITS - 1);

    if (frac_bits > halfway || (frac_bits == halfway && (new_man & 1))) {
        new_man++;
        if (new_man >= (1 << MANTISSA_BITS)) { // Handle mantissa overflow from rounding
            new_man = 0;
            new_exp++;
            if (new_exp >= INF_NAN_EXP) return sign | EXP_MASK; // Overflow to infinity
        }
    }
    
    return sign | (static_cast<uint16_t>(new_exp) << MANTISSA_BITS) | new_man;
}

float float12::to_float() const {
    uint16_t exp = (data & EXP_MASK) >> MANTISSA_BITS;
    uint16_t man = data & MAN_MASK;

    if (exp == INF_NAN_EXP) {
        if (man == 0) return (data & SIGN_MASK) ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();
        return std::numeric_limits<float>::quiet_NaN();
    }
    if (exp == 0 && man == 0) return (data & SIGN_MASK) ? -0.0f : 0.0f;

    // Normalized number (subnormals are not supported in this simplified version)
    int32_t float_exp = static_cast<int32_t>(exp) - EXP_BIAS + 127;
    uint32_t float_man = static_cast<uint32_t>(man) << (23 - MANTISSA_BITS);

    Float32 conv;
    conv.u = ((data & SIGN_MASK) << 16) | (static_cast<uint32_t>(float_exp) << 23) | float_man;
    return conv.f;
}
