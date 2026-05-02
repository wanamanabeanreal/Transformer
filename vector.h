//
// Created by anshu on 5/1/2026.
//

#ifndef TRANSFORMER_VECTOR_H
#define TRANSFORMER_VECTOR_H

#include <cassert>
#include <cstdlib>
#include <immintrin.h> // Required for _mm_malloc

template <typename T>
class fixedvector { //EVENTUALLY: replace malloc, C++ ptr with custom implementations.
    private:
        T* data;
        size_t capacity; //big
        size_t size;
    public:
        fixedvector() : data(nullptr), capacity(0), size(0) {} //ctor
        fixedvector(size_t n): size(n), capacity(n) { //ctor with n (filled because usually we already know size in this code).
            data = (T*)_mm_malloc(n * sizeof(T),32); // 32 is good for my CPU arch
        }
        fixedvector(size_t n, T* source): size(n),  capacity(n) { //ctor 3 with a source array & n
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

        fixedvector& operator=(const fixedvector& other) { //copy ctor(assignment)
            if (this != &other) { //check self assign
                _mm_free(data);
                size = other.size;
                capacity = other.capacity;
                if (other.data != nullptr) {
                    data = (T*)_mm_malloc(capacity * sizeof(T), 32);
                    for (size_t i = 0; i < size; i++) {
                        data[i] = other.data[i];
                    }
                } else {
                    data = nullptr;
                }
            }
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
            if (this != &other) {
                _mm_free(data);
                data = other.data;
                size = other.size;
                capacity = other.capacity;
                other.data = nullptr;
                other.size = 0;
                other.capacity = 0;
            }
            return *this;
        }

        ~fixedvector() {      //destructor
            _mm_free(data);
        }
        //other operators
        fixedvector operator+(const fixedvector& other) const { //addition
            assert(size == other.size);
            fixedvector result = fixedvector(size);
            for (size_t i = 0; i < size; i++) {
                result.data[i] = data[i] + other.data[i];
            }
            return result;
        }

        fixedvector& operator*(size_t n) { //scalar mult
            for (size_t i = 0; i < size; i++) {
                data[i] *= n;
            }
            return *this;
        }

        T operator*(const fixedvector& other) const { //dot product
            assert(size == other.size);
            T sum = 0;
            for (size_t i = 0; i < size; i++) {
                sum += data[i] * other.data[i];
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


#endif //TRANSFORMER_VECTOR_H