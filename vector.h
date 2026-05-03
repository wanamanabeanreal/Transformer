//
// Created by anshu on 5/1/2026.
//

#ifndef TRANSFORMER_VECTOR_H
#define TRANSFORMER_VECTOR_H

#include <cassert>
#include <cstdlib> //malloc indirect
#include <immintrin.h> //AVX2 intrinsics, to replace with assembly

//forward dec
namespace fastmath {
    template <typename U>
    void scalar_mult(U* data, const size_t size, const U& scalar);

    template <typename U>
    U dot(const U* data1, const U* data2, size_t size);
}

template <typename T>
class fixedvector { //EVENTUALLY: replace malloc(arena), rawptr with custom implementations(compress).
    private:
        T* data;
        size_t capacity; //size_t for large
        size_t size;
    public:
        fixedvector() : data(nullptr), capacity(0), size(0) {} //ctor
        fixedvector(size_t n): size(n), capacity(n) { //ctor with n (filled because usually we already know size in this code).
            data = (T*)_mm_malloc(n * sizeof(T),32); // 32 is good for my CPU arch
        }
        fixedvector(size_t n, const T* source): size(n),  capacity(n) { //ctor 3 with a source array & n
            data = (T*)_mm_malloc(n * sizeof(T),32);
            if (data != nullptr && source != nullptr) {
                for (size_t i = 0; i < n; i++) {
                    data[i] = source[i];
                }
            }
        }
        fixedvector (const fixedvector& other) { //copy ctor
                data = (T*)_mm_malloc(other.capacity * sizeof(T), 32);
                size = other.size;
                capacity = other.capacity;
                for (size_t i = 0; i < size; i++) {
                    data[i] = other.data[i];
                }
        }

        fixedvector& operator=(const fixedvector& other) {
            if (this == &other) return *this; //self check

            T* new_data = nullptr;
            if (other.capacity > 0) {
                new_data = static_cast<T*>(_mm_malloc(other.capacity * sizeof(T), 32));
                if (new_data && other.data) {
                    for (size_t i = 0; i < other.size; ++i) new_data[i] = other.data[i];
                }
            }

                _mm_free(data);
                data = new_data;
                size = other.size;
                capacity = other.capacity;
                return *this;
            }

        fixedvector(fixedvector&& other) noexcept { //move ctor //noexcept incase malloc throws
            data = other.data;
            size = other.size;
            capacity = other.capacity;
            other.data = nullptr;
            other.size = 0;
            other.capacity = 0;
        }

        fixedvector& operator=(fixedvector&& other) noexcept{   //move ctor assigment
            if (this == &other) return *this;
            _mm_free(data);
            data = other.data;
            size = other.size;
            capacity = other.capacity;
            other.data = nullptr;
            other.size = 0;
            other.capacity = 0;

            return *this;
        }


        ~fixedvector() {      //destructor
            _mm_free(data);
        }

        //other operators
        fixedvector operator+(const fixedvector& other) const { //addition
            assert(size == other.size);
            fixedvector result = fixedvector(size);
            const T* __restrict p1 = data; //tell compiler we are not sharing ptr, can trigger ptimization on GCC
            const T* __restrict p2 = other.data;
            for (size_t i = 0; i < size; i++) {
                result.data[i] = p1[i] + p2[i];
            }
            return result;
        }

        fixedvector& operator*(const T& scalar) { //scalar mult
            for (size_t i = 0; i < size; i++) {
                data[i] *= scalar;
            }
            return *this;
        }

        T operator|(const fixedvector& other) const { //dot product
            assert(size == other.size);
            T sum = 0;
            const T* __restrict p1 = data;
            const T* __restrict p2 = other.data;
            for (size_t i = 0; i < size; i++) {
                sum += p1[i] * p2[i];
            }
            return sum;
        }

        T& operator [](size_t i) {
            return data[i];
        }

        //acessors
        const T& operator [](size_t i) const {
            return data[i];
        }

        T* get_data() {
            return data;
        }
        const T* get_data() const{
            return data;
        }

        size_t get_size() const {
            return size;
        }

};

//fastops, SIMD in case compiler doesnt optimize dot, and also scalar is better this way.
namespace fastmath {
    template <typename U>
         void scalar_mult(U* data, const size_t size, const U& scalar) {   //fornow, float
            __m256 v_scalar = _mm256_set1_ps(static_cast<float>(scalar)); //make integer and double math versions later, for U
            //it writes scalar to 8 slots of 32 byte register (float = 4 byte).
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
    template <typename U> //needs special compile flags, eg -mavx2 -mfma -O3 -march=haswell
         U dot(const U* data1, const U* data2, size_t size) {
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
}





#endif //TRANSFORMER_VECTOR_H