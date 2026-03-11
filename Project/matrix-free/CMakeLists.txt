cmake_minimum_required(VERSION 3.12.0)
project(matrix_free LANGUAGES CXX C)

include(../../common/cmake-common.cmake)

add_executable(matrix_free ./main.cpp ./MatrixFree.cpp)
deal_ii_setup_target(matrix_free)
