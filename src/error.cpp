
#include "float12.hpp"
#include <limits>
#include <algorithm>
#include <cmath>

// Additional utility functions for float12

// Range validation - static function
bool float12_isInRange(float f) {
    return std::abs(f) <= static_cast<float>(float12::max()) || f == 0.0f || std::isinf(f) || std::isnan(f);
}

// Safe construction with overflow detection - static function
float12 float12_fromFloatSafe(float f, bool* overflow) {
    if (overflow) *overflow = false;
    if (!float12_isInRange(f) && std::isfinite(f)) {
        if (overflow) *overflow = true;
    }
    return float12(f);
}
