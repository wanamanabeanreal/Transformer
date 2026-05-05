//
// Created by anshu on 5/4/2026.
//

#ifndef TRANSFORMER_MATH_H
#define TRANSFORMER_MATH_H

#include <cstdint>

#include <immintrin.h> //intrinsics

//fastops, SIMD since somtimes compiler doesnt optimize dot, add/scalar just better, just float for now


namespace fastmath {
    //structs for datatypes
    template <typename T>
    struct simd_ops;

    template <>
    struct simd_ops<float> { //a float is 4 bytes
            static constexpr size_t lanes = 8;
            using vec = __m256;

            static vec set1(float x) { return _mm256_set1_ps(x); }
            static vec load(const float* p) { return _mm256_load_ps(p); }
            static void store(float* p, vec v) { _mm256_store_ps(p, v); }

            static vec add(vec a, vec b) { return _mm256_add_ps(a, b); }
            static vec mul(vec a, vec b) { return _mm256_mul_ps(a, b); }
        };


    template <>
    struct simd_ops<double> {//a double is 8 bytes
        static constexpr size_t lanes = 4;
        using vec = __m256d;

        static vec set1(double x) { return _mm256_set1_pd(x); }
        static vec load(const double* p) { return _mm256_load_pd(p); }
        static void store(double* p, vec v) { _mm256_store_pd(p, v); }

        static vec add(vec a, vec b) { return _mm256_add_pd(a, b); }
        static vec mul(vec a, vec b) { return _mm256_mul_pd(a, b); }
    };

    template <>
    struct simd_ops<int32_t> {//an int32 is 4 bytes
        static constexpr size_t lanes = 8;
        using vec = __m256i;

        static vec set1(int32_t x) { return _mm256_set1_epi32(x); }
        static vec load(const int32_t* p) { return _mm256_load_si256((const __m256i*)p); }
        static void store(int32_t* p, vec v) { _mm256_store_si256((__m256i*)p, v); }

        static vec add(vec a, vec b) { return _mm256_add_epi32(a, b); }
    };

    template <>
    struct simd_ops<int16_t> {//an int16 is 2 bytes
        static constexpr size_t lanes = 16;
        using vec = __m256i;

        static vec set1(int16_t x) { return _mm256_set1_epi16(x); }
        static vec load(const int16_t* p) { return _mm256_load_si256((const __m256i*)p); }
        static void store(int16_t* p, vec v) { _mm256_store_si256((__m256i*)p, v); }

        static vec add(vec a, vec b) { return _mm256_add_epi16(a, b); }
    };

    template <>
    struct simd_ops<int8_t> {//an int8 is 1 byte
        static constexpr size_t lanes = 32;
        using vec = __m256i;

        static vec set1(int8_t x) { return _mm256_set1_epi8(x); }
        static vec load(const int8_t* p) { return _mm256_load_si256((const __m256i*)p); }
        static void store(int8_t* p, vec v) { _mm256_store_si256((__m256i*)p, v); }

        static vec add(vec a, vec b) { return _mm256_add_epi8(a, b); }
    };

    //operations FLOAT ONLY FOR NOW, WIRE STRUCT ABOVE LATER.
    template <typename U>
         void scalar_mult(U* data, const size_t size, const U& scalar) {
            __m256 v_scalar = _mm256_set1_ps(static_cast<float>(scalar));
            size_t i = 0;
            for (;i +7< size ;i+=8) { //go by 8 to align with AVX2
                __m256 v_data = _mm256_load_ps((&data[i]));//write to reg
                __m256 v_res = _mm256_mul_ps(v_scalar, v_data);
                _mm256_store_ps(&data[i], v_res);
            }
            //tail
            for (; i<size;i+=1) {
                data[i] *= scalar;
            }
        }
    template <typename U> //needs special compile flags, eg -mavx2 -mfma -O3 -march=haswell for MINGW

         U dot(const U* data1, const U* data2, size_t size) { //can replace with 4 sum vars to unroll loop quicker
            __m256 v_sum = _mm256_setzero_ps(); //filled with zeroes
            size_t i = 0;
            for (;i +7< size ;i+=8) { //go by 8 to align with AVX2
                __m256 v_data1 = _mm256_load_ps((&data1[i]));//write to reg
                __m256 v_data2 = _mm256_load_ps((&data2[i]));
                v_sum = _mm256_fmadd_ps(v_data1,v_data2, v_sum);
        } //horizontal collapse, [] array can be replaced with more optimal register shuffle method
        alignas(32) float temp[8];
        _mm256_store_ps(temp, v_sum);
        float total = temp[0] + temp[1] + temp[2] + temp[3] +
                      temp[4] + temp[5] + temp[6] + temp[7];
        for (; i < size; ++i) {
            total += data1[i] * data2[i];
        }
        return total;
        }

    template <typename U>
        void add(U* dst, U* src, size_t size) {
            size_t i = 0;
            for (;i +7< size ;i+=8) {
                __m256 v_data1 = _mm256_load_ps((&dst[i]));//write to reg
                __m256 v_data2 = _mm256_load_ps((&src[i]));
                __m256 v_sum = _mm256_add_ps(v_data1,v_data2);
                _mm256_store_ps(&dst[i],v_sum); //to ram
            }
            for (; i < size; ++i) { //tail
                dst[i] = dst[i] + src[i];
            }
        }
    template <typename U>
        void copy(U* dst, const U* src, size_t n) {
                size_t i = 0;
                for (; i + 7 < n; i += 8) {
                    __m256 v = _mm256_load_ps(&src[i]);
                    _mm256_store_ps(&dst[i], v);
                }
                for (; i < n; ++i) {
                    dst[i] = src[i];
                }
            }


    //matrixOPS next:

}

#endif //TRANSFORMER_MATH_H