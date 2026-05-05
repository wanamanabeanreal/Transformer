//
// Created by anshu on 5/1/2026.
//

#ifndef TRANSFORMER_MATRIX_H
#define TRANSFORMER_MATRIX_H
#include <cstdlib>
#include <cassert>
#include "vector.h"
template <typename T> class matrix;
template <typename T>
class matrix {
    private:
        T* data;
        size_t rows;
        size_t cols;//logical width
        size_t stride;//practical width
    public:
        matrix (): data(nullptr), rows(0), cols(0) {
            //default ctor
        }
        matrix(const size_t r, const size_t c) : data(nullptr), rows(r), cols(c) { //r,c ctor
            stride = ((cols + 7) / 8) * 8;
            size_t total_elements = rows * stride;
            if (rows == 0 || cols == 0) return;
            data = static_cast<T*>(_mm_malloc(total_elements * sizeof(T), 32));
            assert(data != nullptr);
        }

        matrix(const matrix& other) = delete; //copy ctor, force delete, matrices are too big

        matrix& operator=(const matrix& other) = delete; //same for copy assignment

        matrix(const matrix&& other) {

        }


        T& operator ()(const size_t r, const size_t c) {
            return data[r * stride + c];
        }

        ~matrix() {
            if (data) _mm_free(data);
        }

};

#endif //TRANSFORMER_MATRIX_H