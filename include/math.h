//
// fastmath.h — SIMD fast-math operations (AVX2 + FMA)
// Requires: -mavx2 -mfma -O3  (or /arch:AVX2 on MSVC)
//
// simd_ops<T> provides the per-type AVX2 primitives.
// The free functions (dot, add, scalar_mult, copy) dispatch through
// simd_ops<T> so adding a new type only requires a new specialisation.
//

#pragma once

#include <cstdint>
#include <cstddef>
#include <immintrin.h>

namespace fastmath {

// simd_ops<T>  — one specialisation per scalar type
// Required members:
//   static constexpr size_t lanes   — elements per AVX register
//   using vec                       — the __m256* register type
//   static vec  set1(T x)
//   static vec  load(const T* p)    — p must be 32-byte aligned
//   static void store(T* p, vec v)  — p must be 32-byte aligned
//   static vec  add(vec a, vec b)
//   static vec  mul(vec a, vec b)   — not available for integer types (see notes)
//   static vec  fmadd(vec a, vec b, vec c)  — a*b + c  (float/double only)

template <typename T>
struct simd_ops; //Fullspec for dec
template <>
struct simd_ops<float> {
    static constexpr size_t lanes = 8;
    using vec = __m256;

    static vec  set1(float x)              { return _mm256_set1_ps(x); }
    static vec  load(const float* p)       { return _mm256_load_ps(p); }
    static void store(float* p, vec v)     { _mm256_store_ps(p, v); }
    static vec  setzero()                  { return _mm256_setzero_ps(); }

    static vec add  (vec a, vec b)         { return _mm256_add_ps(a, b); }
    static vec sub  (vec a, vec b)         { return _mm256_sub_ps(a, b); }
    static vec mul  (vec a, vec b)         { return _mm256_mul_ps(a, b); }
    static vec div  (vec a, vec b)         { return _mm256_div_ps(a, b); }
    static vec fmadd(vec a, vec b, vec c)  { return _mm256_fmadd_ps(a, b, c); } // a*b + c
};

// double (8 bytes × 4 = 256 bits)
struct simd_ops<double> {
    static constexpr size_t lanes = 4;
    using vec = __m256d;

    static vec  set1(double x)             { return _mm256_set1_pd(x); }
    static vec  load(const double* p)      { return _mm256_load_pd(p); }
    static void store(double* p, vec v)    { _mm256_store_pd(p, v); }
    static vec  setzero()                  { return _mm256_setzero_pd(); }

    static vec add  (vec a, vec b)         { return _mm256_add_pd(a, b); }
    static vec sub  (vec a, vec b)         { return _mm256_sub_pd(a, b); }
    static vec mul  (vec a, vec b)         { return _mm256_mul_pd(a, b); }
    static vec div  (vec a, vec b)         { return _mm256_div_pd(a, b); }
    static vec fmadd(vec a, vec b, vec c)  { return _mm256_fmadd_pd(a, b, c); }
};

//  int32_t (4 bytes × 8 = 256 bits) 
// Note: no hardware mul for 32-bit full-width; mullo gives low 32 bits of product.
// No FMA for integers — use scalar tail or widen to float when needed.
template <>
struct simd_ops<int32_t> {
    static constexpr size_t lanes = 8;
    using vec = __m256i;

    static vec  set1(int32_t x)             { return _mm256_set1_epi32(x); }
    static vec  load(const int32_t* p)      { return _mm256_load_si256(reinterpret_cast<const __m256i*>(p)); }
    static void store(int32_t* p, vec v)    { _mm256_store_si256(reinterpret_cast<__m256i*>(p), v); }
    static vec  setzero()                   { return _mm256_setzero_si256(); }

    static vec add(vec a, vec b)            { return _mm256_add_epi32(a, b); }
    static vec sub(vec a, vec b)            { return _mm256_sub_epi32(a, b); }
    static vec mul(vec a, vec b)            { return _mm256_mullo_epi32(a, b); } // low 32 bits
};

// --- int16_t (2 bytes × 16 = 256 bits) ----------------------------------------
// mul: _mm256_mullo_epi16 gives low 16 bits of product.
template <>
struct simd_ops<int16_t> {
    static constexpr size_t lanes = 16;
    using vec = __m256i;

    static vec  set1(int16_t x)             { return _mm256_set1_epi16(x); }
    static vec  load(const int16_t* p)      { return _mm256_load_si256(reinterpret_cast<const __m256i*>(p)); }
    static void store(int16_t* p, vec v)    { _mm256_store_si256(reinterpret_cast<__m256i*>(p), v); }
    static vec  setzero()                   { return _mm256_setzero_si256(); }

    static vec add(vec a, vec b)            { return _mm256_add_epi16(a, b); }
    static vec sub(vec a, vec b)            { return _mm256_sub_epi16(a, b); }
    static vec mul(vec a, vec b)            { return _mm256_mullo_epi16(a, b); } // low 16 bits
};

//  int8_t (1 byte × 32 = 256 bits) 
// No direct 8-bit multiply in AVX2; mul is intentionally omitted.
// Use maddubs (_mm256_maddubs_epi16) for dot-product-like ops on int8 if needed.
template <>
struct simd_ops<int8_t> {
    static constexpr size_t lanes = 32;
    using vec = __m256i;

    static vec  set1(int8_t x)              { return _mm256_set1_epi8(x); }
    static vec  load(const int8_t* p)       { return _mm256_load_si256(reinterpret_cast<const __m256i*>(p)); }
    static void store(int8_t* p, vec v)     { _mm256_store_si256(reinterpret_cast<__m256i*>(p), v); }
    static vec  setzero()                   { return _mm256_setzero_si256(); }

    static vec add(vec a, vec b)            { return _mm256_add_epi8(a, b); }
    static vec sub(vec a, vec b)            { return _mm256_sub_epi8(a, b); }
    // no mul — see note above
};

// Free-function operations — dispatch through simd_ops<T>
// All pointer arguments must be 32-byte aligned (_mm_malloc(..., 32)).


// Horizontal reduction — sums all lanes of a register.
// Specialised per register width so the scalar code doesn't have to know.
namespace detail {
    inline float hsum(__m256 v) {
        // fold 8 → 4 → 2 → 1
        __m128 lo = _mm256_castps256_ps128(v);
        __m128 hi = _mm256_extractf128_ps(v, 1);
        __m128 s  = _mm_add_ps(lo, hi);
        s = _mm_hadd_ps(s, s);
        s = _mm_hadd_ps(s, s);
        return _mm_cvtss_f32(s);
    }

    inline int32_t hsum(__m256i v) {
        // Fold 256-bit to 128-bit
        __m128i lo = _mm256_castsi256_si128(v);
        __m128i hi = _mm256_extracti128_si256(v, 1);
        __m128i s = _mm_add_epi32(lo, hi);
        
        // Fold 128-bit horizontally
        s = _mm_hadd_epi32(s, s);
        s = _mm_hadd_epi32(s, s);
        return _mm_cvtsi128_si32(s);
    }

    // Fold 16 int16 lanes down to 1 scalar
    inline int16_t hsum_epi16(__m256i v) {
        __m128i lo = _mm256_castsi256_si128(v);
        __m128i hi = _mm256_extracti128_si256(v, 1);
        __m128i s  = _mm_add_epi16(lo, hi);
        s = _mm_hadd_epi16(s, s);
        s = _mm_hadd_epi16(s, s);
        s = _mm_hadd_epi16(s, s);
        return static_cast<int16_t>(_mm_extract_epi16(s, 0));
    }

    inline double hsum(__m256d v) {
        __m128d lo = _mm256_castpd256_pd128(v);
        __m128d hi = _mm256_extractf128_pd(v, 1);
        __m128d s  = _mm_add_pd(lo, hi);
        s = _mm_hadd_pd(s, s);
        return _mm_cvtsd_f64(s);
    }
} // namespace detail

// dot — 4 independent accumulators to hide FMA latency (Haswell+: 4-cycle FMA)
//
// float/double: uses FMA.  integers: add(mul()) — no FMA in AVX2 for ints.
// After the unrolled loop the 4 accumulators are folded, then a single-vector
// tail cleans up any leftover full registers, and a scalar tail handles the rest.

// float / double (primary template — requires fmadd in simd_ops<T>)
template <typename T>
T dot(const T* __restrict a, const T* __restrict b, size_t n) {
    using ops = simd_ops<T>;
    constexpr size_t step = ops::lanes * 4;

    typename ops::vec acc0 = ops::setzero();
    typename ops::vec acc1 = ops::setzero();
    typename ops::vec acc2 = ops::setzero();
    typename ops::vec acc3 = ops::setzero();

    size_t i = 0;
    for (; i + step <= n; i += step) {                                                                  //EX:
        acc0 = ops::fmadd(ops::load(a + i + ops::lanes * 0), ops::load(b + i + ops::lanes * 0), acc0); //0-7
        acc1 = ops::fmadd(ops::load(a + i + ops::lanes * 1), ops::load(b + i + ops::lanes * 1), acc1); //8-15
        acc2 = ops::fmadd(ops::load(a + i + ops::lanes * 2), ops::load(b + i + ops::lanes * 2), acc2); //16-23
        acc3 = ops::fmadd(ops::load(a + i + ops::lanes * 3), ops::load(b + i + ops::lanes * 3), acc3); //24-31
    }
    // fold 4 → 1
    acc0 = ops::add(ops::add(acc0, acc1), ops::add(acc2, acc3));

    for (; i + ops::lanes <= n; i += ops::lanes)
        acc0 = ops::fmadd(ops::load(a + i), ops::load(b + i), acc0);

    T total = static_cast<T>(detail::hsum(acc0));
    for (; i < n; ++i) total += a[i] * b[i];
    return total;
}

// int32_t — mullo + add, no FMA
template <>
inline int32_t dot<int32_t>(const int32_t* __restrict a, const int32_t* __restrict b, size_t n) {
    using ops = simd_ops<int32_t>;
    constexpr size_t step = ops::lanes * 4;

    ops::vec acc0 = ops::setzero(), acc1 = ops::setzero();
    ops::vec acc2 = ops::setzero(), acc3 = ops::setzero();

    size_t i = 0;
    for (; i + step <= n; i += step) {
        acc0 = ops::add(acc0, ops::mul(ops::load(a + i + ops::lanes * 0), ops::load(b + i + ops::lanes * 0)));
        acc1 = ops::add(acc1, ops::mul(ops::load(a + i + ops::lanes * 1), ops::load(b + i + ops::lanes * 1)));
        acc2 = ops::add(acc2, ops::mul(ops::load(a + i + ops::lanes * 2), ops::load(b + i + ops::lanes * 2)));
        acc3 = ops::add(acc3, ops::mul(ops::load(a + i + ops::lanes * 3), ops::load(b + i + ops::lanes * 3)));
    }
    acc0 = ops::add(ops::add(acc0, acc1), ops::add(acc2, acc3));

    for (; i + ops::lanes <= n; i += ops::lanes)
        acc0 = ops::add(acc0, ops::mul(ops::load(a + i), ops::load(b + i)));

    int32_t total = detail::hsum(acc0);
    for (; i < n; ++i) total += a[i] * b[i];
    return total;
}

// int16_t — mullo + add, no FMA
template <>
inline int16_t dot<int16_t>(const int16_t* __restrict a, const int16_t* __restrict b, size_t n) {
    using ops = simd_ops<int16_t>;
    constexpr size_t step = ops::lanes * 4;

    ops::vec acc0 = ops::setzero(), acc1 = ops::setzero();
    ops::vec acc2 = ops::setzero(), acc3 = ops::setzero();

    size_t i = 0;
    for (; i + step <= n; i += step) {
        acc0 = ops::add(acc0, ops::mul(ops::load(a + i + ops::lanes * 0), ops::load(b + i + ops::lanes * 0)));
        acc1 = ops::add(acc1, ops::mul(ops::load(a + i + ops::lanes * 1), ops::load(b + i + ops::lanes * 1)));
        acc2 = ops::add(acc2, ops::mul(ops::load(a + i + ops::lanes * 2), ops::load(b + i + ops::lanes * 2)));
        acc3 = ops::add(acc3, ops::mul(ops::load(a + i + ops::lanes * 3), ops::load(b + i + ops::lanes * 3)));
    }
    acc0 = ops::add(ops::add(acc0, acc1), ops::add(acc2, acc3));

    for (; i + ops::lanes <= n; i += ops::lanes)
        acc0 = ops::add(acc0, ops::mul(ops::load(a + i), ops::load(b + i)));

    int16_t total = detail::hsum_epi16(acc0);
    for (; i < n; ++i) total += a[i] * b[i];
    return total;
}

// add — x4 unroll  (dst[i] += src[i])
template <typename T>
void add(T* __restrict dst, const T* __restrict src, size_t n) {
    using ops = simd_ops<T>;
    constexpr size_t step = ops::lanes * 4;

    size_t i = 0;
    for (; i + step <= n; i += step) {
        ops::store(dst + i + ops::lanes * 0, ops::add(ops::load(dst + i + ops::lanes * 0), ops::load(src + i + ops::lanes * 0)));
        ops::store(dst + i + ops::lanes * 1, ops::add(ops::load(dst + i + ops::lanes * 1), ops::load(src + i + ops::lanes * 1)));
        ops::store(dst + i + ops::lanes * 2, ops::add(ops::load(dst + i + ops::lanes * 2), ops::load(src + i + ops::lanes * 2)));
        ops::store(dst + i + ops::lanes * 3, ops::add(ops::load(dst + i + ops::lanes * 3), ops::load(src + i + ops::lanes * 3)));
    }
    for (; i + ops::lanes <= n; i += ops::lanes)
        ops::store(dst + i, ops::add(ops::load(dst + i), ops::load(src + i)));
    for (; i < n; ++i) dst[i] += src[i];
}

// sub — x4 unroll  (dst[i] -= src[i])
template <typename T>
void sub(T* __restrict dst, const T* __restrict src, size_t n) {
    using ops = simd_ops<T>;
    constexpr size_t step = ops::lanes * 4;

    size_t i = 0;
    for (; i + step <= n; i += step) {
        ops::store(dst + i + ops::lanes * 0, ops::sub(ops::load(dst + i + ops::lanes * 0), ops::load(src + i + ops::lanes * 0)));
        ops::store(dst + i + ops::lanes * 1, ops::sub(ops::load(dst + i + ops::lanes * 1), ops::load(src + i + ops::lanes * 1)));
        ops::store(dst + i + ops::lanes * 2, ops::sub(ops::load(dst + i + ops::lanes * 2), ops::load(src + i + ops::lanes * 2)));
        ops::store(dst + i + ops::lanes * 3, ops::sub(ops::load(dst + i + ops::lanes * 3), ops::load(src + i + ops::lanes * 3)));
    }
    for (; i + ops::lanes <= n; i += ops::lanes)
        ops::store(dst + i, ops::sub(ops::load(dst + i), ops::load(src + i)));
    for (; i < n; ++i) dst[i] -= src[i];
}

// scalar_mult — x4 unroll, scalar broadcast once  (data[i] *= scalar)
template <typename T>
void scalar_mult(T* __restrict data, size_t n, const T scalar) {
    using ops = simd_ops<T>;
    constexpr size_t step = ops::lanes * 4;
    typename ops::vec vs = ops::set1(scalar);

    size_t i = 0;
    for (; i + step <= n; i += step) {
        ops::store(data + i + ops::lanes * 0, ops::mul(ops::load(data + i + ops::lanes * 0), vs));
        ops::store(data + i + ops::lanes * 1, ops::mul(ops::load(data + i + ops::lanes * 1), vs));
        ops::store(data + i + ops::lanes * 2, ops::mul(ops::load(data + i + ops::lanes * 2), vs));
        ops::store(data + i + ops::lanes * 3, ops::mul(ops::load(data + i + ops::lanes * 3), vs));
    }
    for (; i + ops::lanes <= n; i += ops::lanes)
        ops::store(data + i, ops::mul(ops::load(data + i), vs));
    for (; i < n; ++i) data[i] *= scalar;
}

// copy — x4 unroll

template <typename T>
void copy(T* __restrict dst, const T* __restrict src, size_t n) {
    using ops = simd_ops<T>;
    constexpr size_t step = ops::lanes * 4;

    size_t i = 0;
    for (; i + step <= n; i += step) {
        ops::store(dst + i + ops::lanes * 0, ops::load(src + i + ops::lanes * 0));
        ops::store(dst + i + ops::lanes * 1, ops::load(src + i + ops::lanes * 1));
        ops::store(dst + i + ops::lanes * 2, ops::load(src + i + ops::lanes * 2));
        ops::store(dst + i + ops::lanes * 3, ops::load(src + i + ops::lanes * 3));
    }
    for (; i + ops::lanes <= n; i += ops::lanes)
        ops::store(dst + i, ops::load(src + i));
    for (; i < n; ++i) dst[i] = src[i];
}

// fill — x4 unroll, value broadcast once
template <typename T>
void fill(T* __restrict data, size_t n, const T val) {
    using ops = simd_ops<T>;
    constexpr size_t step = ops::lanes * 4;
    typename ops::vec v = ops::set1(val);

    size_t i = 0;
    for (; i + step <= n; i += step) {
        ops::store(data + i + ops::lanes * 0, v);
        ops::store(data + i + ops::lanes * 1, v);
        ops::store(data + i + ops::lanes * 2, v);
        ops::store(data + i + ops::lanes * 3, v);
    }
    for (; i + ops::lanes <= n; i += ops::lanes)
        ops::store(data + i, v);
    for (; i < n; ++i) data[i] = val;
}

//max of a vector/matrix
template <typename T>
    void max () {

}
//multiply vectors
template <typename T>
    void vecmul() {

}

// matmul — your turn
template <typename T>
    void matmul () {

}
} // namespace fastmath
