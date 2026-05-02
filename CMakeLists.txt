cmake_minimum_required(VERSION 4.0)
project(transformer)

set(CMAKE_CXX_STANDARD 20)

add_executable(transformer main.cpp
        vector.h
        matrix.h
        vector.cpp
        matrix.cpp)
