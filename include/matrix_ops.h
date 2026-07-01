//
// matrix_ops.h — object-aware operations on matrix<T>
//
// Free functions
// object aware things that need more than one matrix for ex.
// because it was weird to have them in class a.mult(c,b) is stupid and in math is circular dependency.
//

#pragma once

#include <cassert>
#include <cstddef>
#include <type_traits>  // is_same_v (SIMD transpose float guard)

#include "matrix.h"
#include "math_utils.h"

namespace matops {
    constexpr size_t clamp_end(size_t x, size_t limit) {
        return x < limit ? x : limit;
    }


    namespace detail {

        // Scalar tiled transpose, works for all T, CS=32 for my L1 cache fit.
        template <typename T>
        void transpose_tiled(matrix<T>& out, const matrix<T>& a) {
            constexpr size_t CS = 32;
            for (size_t ii = 0; ii < a.rows(); ii += CS) {
                for (size_t jj = 0; jj < a.cols(); jj += CS) {
                    size_t i_end = clamp_end(ii + TS, M);
                    size_t j_end = clamp_end(jj + TS, N);
                    for (size_t i = ii; i < i_end; ++i)
                        for (size_t j = jj; j < j_end; ++j)
                            out(j, i) = a(i, j);
                }
            }
        }


        // simd transpose — float only, 8x8 tiles via fastmath::detail::transpose8x8_ps.
        // processes full 8x8 blocks in-register; scalar fallback for edge rows/cols.
        // for non-float T this overload is excluded via if constexpr at the call site.
        inline void transpose_simd(matrix<float>& out, const matrix<float>& a) {
            const size_t M = a.rows();
            const size_t N = a.cols();
            constexpr size_t TS = 8;    // register tile width == float lanes

            size_t ii = 0;
            for (; ii + TS <= M; ii += TS) {
                size_t jj = 0;
                for (; jj + TS <= N; jj += TS) {
                    // load 8 rows of 8 floats from a(ii..ii+7, jj..jj+7)
                    __m256 rows[8];
                    for (size_t r = 0; r < TS; ++r)
                        rows[r] = _mm256_loadu_ps(a.row(ii + r) + jj);

                    fastmath::detail::transpose8x8_ps(rows);

                    // store transposed rows into out(jj..jj+7, ii..ii+7)
                    for (size_t r = 0; r < TS; ++r)
                        _mm256_storeu_ps(out.row(jj + r) + ii, rows[r]);
                }
                // scalar tail for remaining cols
                for (; jj < N; ++jj)
                    for (size_t i = ii; i < ii + TS; ++i)
                        out(jj, i) = a(i, jj);
            }
            // scalar tail for remaining rows
            for (; ii < M; ++ii)
                for (size_t jj = 0; jj < N; ++jj)
                    out(jj, ii) = a(ii, jj);
        }

        // matmul_impl — single body, all variants selected at compile time.
        // bt must be b pre-transposed: bt.rows()==b.cols(), bt.cols()==b.rows()==a.cols()
        // T{} zero-init on accumulators — prevents compiler from assuming vectorization is safe.
        template <bool Tiled, bool SIMD, typename T>
        void matmul_impl(matrix<T>& c, const matrix<T>& a, const matrix<T>& bt) {
            const size_t M = a.rows();
            const size_t N = bt.rows();
            const size_t K = a.cols();
            constexpr size_t TS = 32;   // tile side — multiple of simd_lanes for alignment

            if constexpr (!Tiled) {
                for (size_t i = 0; i < M; ++i) {
                    for (size_t j = 0; j < N; ++j) {
                        if constexpr (SIMD) {
                            c(i, j) = fastmath::dot(a.row(i), bt.row(j), K);
                        } else {
                            T sum = T{};
                            for (size_t k = 0; k < K; ++k) sum += a(i, k) * bt(j, k);
                            c(i, j) = sum;
                        }
                    }
                }
            } else {
                // zero c — we accumulate across k-tiles
                for (size_t i = 0; i < M; ++i) fastmath::fill(c.row(i), N, T{});

                for (size_t ii = 0; ii < M; ii += TS) {
                    size_t i_end = clamp_end(ii + TS, M);
                    for (size_t jj = 0; jj < N; jj += TS) {
                        size_t j_end = clamp_end(jj + TS, N);
                        for (size_t kk = 0; kk < K; kk += TS) {
                            size_t k_end = clamp_end(kk + TS, K);
                            size_t k_len = k_end - kk;
                            for (size_t i = ii; i < i_end; ++i) {
                                for (size_t j = jj; j < j_end; ++j) {
                                    if constexpr (SIMD) {
                                        c(i, j) += fastmath::dot(a.row(i) + kk, bt.row(j) + kk, k_len);
                                    } else {
                                        T sum = T{};
                                        for (size_t k = kk; k < k_end; ++k) sum += a(i, k) * bt(j, k);
                                        c(i, j) += sum;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

    } // namespace detail


    // transpose<Tiled, SIMD> — all four combinations covered.
    //
    //   transpose<false, false>  scalar naive (falls through to tiled impl, same thing for small)
    //   transpose<true,  false>  scalar tiled, CS=32
    //   transpose<false, true >  SIMD 8x8 register tile (float only; other T falls back to tiled)
    //   transpose<true,  true >  SIMD 8x8 register tile — Tiled is implicit in the 8x8 blocking
    //
    // returns a new (cols x rows) matrix.
    template <bool Tiled, bool SIMD, typename T>
    matrix<T> transpose(const matrix<T>& a) {
        matrix<T> out(a.cols(), a.rows());
        if constexpr (SIMD && std::is_same_v<T, float>) {
            detail::transpose_simd(out, a);
        } else {
            detail::transpose_tiled(out, a);
        }
        return out;
    }

    // transpose_inplace — transposed copy then move-assign back.
    // actual in place is harder on non-square
    template <bool Tiled = true, bool SIMD = false, typename T>
    void transpose_inplace(matrix<T>& a) {
        matrix<T> temp = transpose<Tiled, SIMD>(a);
        a = (matrix<T>&&)temp;
    }


    //  matmul variants: each transposes b then calls the shared impl.

    template <typename T>
    void matmul_naive(matrix<T>& c, const matrix<T>& a, const matrix<T>& b) {
        assert(a.cols() == b.rows());
        matrix<T> bt = transpose<false, false>(b);
        detail::matmul_impl<false, false>(c, a, bt);
    }

    template <typename T>
    void matmul_simd(matrix<T>& c, const matrix<T>& a, const matrix<T>& b) {
        assert(a.cols() == b.rows());
        matrix<T> bt = transpose<true, false>(b);
        detail::matmul_impl<false, true>(c, a, bt);
    }

    template <typename T>
    void matmul_tiled(matrix<T>& c, const matrix<T>& a, const matrix<T>& b) {
        assert(a.cols() == b.rows());
        matrix<T> bt = transpose<true, false>(b);
        detail::matmul_impl<true, false>(c, a, bt);
    }

    template <typename T>
    void matmul_tiled_simd(matrix<T>& c, const matrix<T>& a, const matrix<T>& b) {
        assert(a.cols() == b.rows());
        matrix<T> bt = transpose<true, true>(b);
        detail::matmul_impl<true, true>(c, a, bt);
    }

    // matmul, default.
    template <typename T>
    void matmul(matrix<T>& c, const matrix<T>& a, const matrix<T>& b) {
        matmul_tiled_simd(c, a, b);
    }

    // matmul_test<Tiled, SIMD>  single call site, flags select path at compile time
    // bc of constexpr
    // transpose of b uses the matching SIMD flag so the whole pipeline is same
    //
    //   matmul_test<false, false>(c, a, b)  naive baseline
    //   matmul_test<true,  false>(c, a, b)  tiled scalar
    //   matmul_test<false, true >(c, a, b)  SIMD untiled
    //   matmul_test<true,  true >(c, a, b)  tiled + SIMD  (production)
    template <bool Tiled, bool SIMD, typename T>
    void matmul_test(matrix<T>& c, const matrix<T>& a, const matrix<T>& b) {
        assert(a.cols() == b.rows());
        matrix<T> bt = transpose<Tiled, SIMD>(b);
        detail::matmul_impl<Tiled, SIMD>(c, a, bt);
    }

} // namespace matops
