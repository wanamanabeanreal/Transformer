//
// Created by anshu on 5/1/2026.
//

#ifndef TRANSFORMER_VECTOR_H
#define TRANSFORMER_VECTOR_H

#include <cstdlib>

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

                        //move ctor assigment
                        //destructor


};


#endif //TRANSFORMER_VECTOR_H