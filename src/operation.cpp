
#include "float12.hpp"

// Note: Binary arithmetic and comparison operators are now defined as friend functions in the header

// --- Stream Operators ---
inline std::ostream& operator<<(std::ostream& os, const float12& f12) {
    os << float(f12);
    return os;
}

inline std::istream& operator>>(std::istream& is, float12& f12) {
    float temp;
    is >> temp;
    f12 = float12(temp);
    return is;
}
