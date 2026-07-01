//
// matrix.h — row-major 2-D matrix with stride padding for AVX2 alignment
//
// Rows are padded so each row starts on a 32-byte boundary.
// Logical size is rows × cols; allocated size is rows × stride.
//

#pragma once

#include <cassert>
#include <immintrin.h>  // _mm_malloc / _mm_free


#include "math_utils.h"

template <typename T>
class matrix {
public:
    // --- constructors -------------------------------------------------------

    matrix() : data_(nullptr), rows_(0), cols_(0), stride_(0) {}

    matrix(size_t r, size_t c)
        : rows_(r), cols_(c),
          stride_(((c + simd_lanes - 1) / simd_lanes) * simd_lanes),
          data_(alloc(r, stride_)) {}

    // No copy — matrices can be large; force explicit clone() if needed.
    matrix(const matrix&)            = delete;
    matrix& operator=(const matrix&) = delete;

    matrix(matrix&& other) noexcept
        : data_(other.data_), rows_(other.rows_),
          cols_(other.cols_), stride_(other.stride_) {
        other.data_   = nullptr;
        other.rows_   = 0;
        other.cols_   = 0;
        other.stride_ = 0;
    }

    matrix& operator=(matrix&& other) noexcept {
        if (this == &other) return *this;
        _mm_free(data_);
        data_   = other.data_;
        rows_   = other.rows_;
        cols_   = other.cols_;
        stride_ = other.stride_;
        other.data_   = nullptr;
        other.rows_   = 0;
        other.cols_   = 0;
        other.stride_ = 0;
        return *this;
    }

    ~matrix() { _mm_free(data_); }

    // --- element access -----------------------------------------------------

    T&       operator()(size_t r, size_t c)       { return data_[r * stride_ + c]; }
    const T& operator()(size_t r, size_t c) const { return data_[r * stride_ + c]; }

    // Raw pointer to row r (aligned, length == stride_)
    T*       row(size_t r)       { return data_ + r * stride_; }
    const T* row(size_t r) const { return data_ + r * stride_; }

    // --- accessors ----------------------------------------------------------

    size_t rows()   const { return rows_; }
    size_t cols()   const { return cols_; }
    size_t stride() const { return stride_; }  // padded row width

    // Deep copy
    matrix clone() const {
        matrix out(rows_, cols_);
        for (size_t r = 0; r < rows_; ++r)
            fastmath::copy(out.row(r), row(r), cols_);
        return out;
    }



private:
    T*     data_;
    size_t rows_;
    size_t cols_;
    size_t stride_;

    // Pad column count to next multiple of the widest SIMD lane width (8 for float/int32)
    static constexpr size_t simd_lanes = 8;

    static T* alloc(size_t r, size_t stride) {
        if (r == 0 || stride == 0) return nullptr;
        T* p = static_cast<T*>(_mm_malloc(r * stride * sizeof(T), 32));
        assert(p != nullptr);
        return p;
    }
};
