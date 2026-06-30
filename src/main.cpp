#include <iostream>
#include "math.h"
#include "vector.h"
#include "matrix.h"

int main() {
    // quick smoke-test
    fixedvector<float> a(8), b(8);
    for (size_t i = 0; i < 8; ++i) { a[i] = static_cast<float>(i); b[i] = 1.0f; }

    float d = a | b;  // dot product
    std::cout << "dot = " << d << "\n";  // expected: 0+1+2+…+7 = 28

    return 0;
}
