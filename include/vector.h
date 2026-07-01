//
// vector.h — fixed-capacity heap-allocated vector with SIMD ops
//
// Memory is always 32-byte aligned (required by AVX2 loads/stores).
// Size is fixed at construction — no push_back / resize.
//

#pragma once

#include <cassert>
#include <immintrin.h>

// _mm_malloc / _mm_free

#include "math_utils.h"

template <typename T>
class fixedvector {
public:
    //  constructors =

    fixedvector() : data_(nullptr), size_(0) {}

    explicit fixedvector(size_t n)
        : size_(n), data_(alloc(n)) {}

    // Construct from a raw array.
    fixedvector(size_t n, const T* source)
        : size_(n), data_(alloc(n)) {
        fastmath::copy(data_, source, n);
    }

    // Copy
    fixedvector(const fixedvector& other)
        : size_(other.size_), data_(alloc(other.size_)) {
        fastmath::copy(data_, other.data_, size_);
    }

    fixedvector& operator=(const fixedvector& other) {
        if (this == &other) return *this;
        T* nd = alloc(other.size_);
        fastmath::copy(nd, other.data_, other.size_);
        _mm_free(data_);
        data_ = nd;
        size_ = other.size_;
        return *this;
    }

    // Move
    fixedvector(fixedvector&& other) noexcept
        : data_(other.data_), size_(other.size_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }

    fixedvector& operator=(fixedvector&& other) noexcept {
        if (this == &other) return *this;
        _mm_free(data_);
        data_  = other.data_;
        size_  = other.size_;
        other.data_ = nullptr;
        other.size_ = 0;
        return *this;
    }

    ~fixedvector() { _mm_free(data_); }

    //  accessors

    T&       operator[](size_t i)       { return data_[i]; }
    const T& operator[](size_t i) const { return data_[i]; }

    T*       data()       { return data_; }
    const T* data() const { return data_; }

    size_t size() const { return size_; }

    //  arithmetic operators (in-place)

    fixedvector& operator+=(const fixedvector& other) {
        assert(size_ == other.size_);
        fastmath::add(data_, other.data_, size_);
        return *this;
    }

    fixedvector& operator-=(const fixedvector& other) {
        assert(size_ == other.size_);
        fastmath::sub(data_, other.data_, size_);
        return *this;
    }

    fixedvector& operator*=(const T scalar) {
        fastmath::scalar_mult(data_, size_, scalar);
        return *this;
    }

    //  arithmetic operators (value)

    fixedvector operator+(const fixedvector& other) const {
        fixedvector r(*this);
        r += other;
        return r;
    }

    fixedvector operator-(const fixedvector& other) const {
        fixedvector r(*this);
        r -= other;
        return r;
    }

    fixedvector operator*(const T scalar) const {
        fixedvector r(*this);
        r *= scalar;
        return r;
    }

    //  dot product  (operator|)
    T operator|(const fixedvector& other) const {
        assert(size_ == other.size_);
        return fastmath::dot(data_, other.data_, size_);
    }

private:
    T*     data_;
    size_t size_;

    static T* alloc(size_t n) {
        if (n == 0) return nullptr;
        T* p = static_cast<T*>(_mm_malloc(n * sizeof(T), 32));
        assert(p != nullptr);
        return p;
    }
};
