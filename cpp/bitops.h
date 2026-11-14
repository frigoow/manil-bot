#pragma once

#include <cstdint>
#include <type_traits>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace manille {

// Population count and lowest-set-bit index over a 64-bit mask.
//
// The obvious spellings, __builtin_popcountll / __builtin_ctzll, are
// GCC/Clang builtins that MSVC does not provide, and these sit in the hot
// path of the samplers, the ISMCTS search and the rule agent's play-out -- so
// a portable-but-slow fallback would be felt. C++20's <bit> would settle it,
// but the project targets C++17.
//
// Templates rather than uint64_t/int64_t overloads: callers pass int64_t,
// uint64_t and unsigned long long, which are distinct types on LP64 Linux,
// and a fixed overload set makes the third one ambiguous.
template <typename T>
inline int popcount64(T mask) {
    static_assert(std::is_integral<T>::value, "popcount64 takes an integer mask");
    const std::uint64_t m = static_cast<std::uint64_t>(mask);
#if defined(_MSC_VER)
    return static_cast<int>(__popcnt64(m));
#else
    return __builtin_popcountll(m);
#endif
}

// The mask must be non-zero.
template <typename T>
inline int ctz64(T mask) {
    static_assert(std::is_integral<T>::value, "ctz64 takes an integer mask");
    const std::uint64_t m = static_cast<std::uint64_t>(mask);
#if defined(_MSC_VER)
    unsigned long index;
    _BitScanForward64(&index, m);
    return static_cast<int>(index);
#else
    return __builtin_ctzll(m);
#endif
}

}  // namespace manille
