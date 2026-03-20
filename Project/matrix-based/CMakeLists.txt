cmake_minimum_required(VERSION 3.12.0)
project(matrix_based LANGUAGES CXX C)

include(../common/cmake-common.cmake)

add_executable(matrix_based ./main.cpp ./MatrixBased.cpp)
deal_ii_setup_target(matrix_based)
