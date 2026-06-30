cmake_minimum_required(VERSION 3.20)
project(transformer CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --------------------------------------------------------------------------
# Compiler flags — AVX2 + FMA are required for the SIMD ops in math.h
# --------------------------------------------------------------------------
if (MSVC)
    add_compile_options(/arch:AVX2 /O2 /W4)
else()
    # GCC / Clang (CLion defaults to one of these)
    add_compile_options(-mavx2 -mfma -O3 -Wall -Wextra)
endif()

# --------------------------------------------------------------------------
# Include paths
# --------------------------------------------------------------------------
include_directories(include)

# --------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------
add_executable(transformer
    src/main.cpp
    src/layer.cpp
)
