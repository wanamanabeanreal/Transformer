//
// Created by anshu on 5/1/2026.
//

#ifndef TRANSFORMER_VECTOR_H
#define TRANSFORMER_VECTOR_H

#include <cassert>
#include "math.h"


//double forward dec
template <typename T> class fixedvector;

//forward dec
namespace fastmath {
    template <typename U>
    void scalar_mult(U* data, const size_t size, const U& scalar);

    template <typename U>
    U dot(const U* data1, const U* data2, size_t size);

    template <typename U>
    void add(U* dst, const U* src, size_t size);

    template <typename U>
    void copy(U* dst, const U* src, size_t size);
}

template <typename T> //one of cap/size
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
            fastmath::copy(data, other.data, size);
            }

    fixedvector(const fixedvector& other) {
            data = (T*)_mm_malloc(other.capacity * sizeof(T), 32);
            size = other.size;
            capacity = other.capacity;

            fastmath::copy(data, other.data, size);
        }

        fixedvector& operator=(const fixedvector& other) { //copy assign
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
            // operators

        fixedvector &operator+=(const fixedvector& other) { //addition
            assert(size == other.size);
            /*fixedvector result = fixedvector(size);
            const T* __restrict p1 = data; //tell compiler we are not sharing ptr, can trigger ptimization on GCC
            const T* __restrict p2 = other.data;
            for (size_t i = 0; i < size; i++) {
                result.data[i] = p1[i] + p2[i];
            }*/
            fastmath::add(& this, & other);
            return *this;
        }


         fixedvector operator+(const fixedvector& other) const {
            fixedvector result = *this;
            result += other;
            return result;
        }

        fixedvector& operator*=(const T& scalar) { //scalar mult
            /*for (size_t i = 0; i < size; i++) {
                data[i] *= scalar;
            }
            return *this;*/ //slow
            size_t s = this->get_size();
            return fastmath::scalar_mult(this->data,s,scalar);
        }

        // Non-modifying version
        fixedvector operator*(const T& scalar) const
            {
                fixedvector result = *this;  // copy
                result *= scalar;            // reuse the *= above
                return result;
            }

        T operator|(const fixedvector& other) const { //dot product
            assert(size == other.size);
            /*T sum = 0;
            const T* __restrict p1 = data;
            const T* __restrict p2 = other.data;
            for (size_t i = 0; i < size; i++) {
                sum += p1[i] * p2[i];
            }*/ //slow

            return fastmath::dot(this->data,other.data);
        }

};





#endif //TRANSFORMER_VECTOR_H