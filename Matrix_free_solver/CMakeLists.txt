cmake_minimum_required(VERSION 3.12.0)
project(Matrix_free_solver LANGUAGES CXX C)

include(../common/cmake-common.cmake)

add_executable(MatrixFree3D src/MatrixFree3D.cpp src/ADRMatrixFree3D.cpp)
deal_ii_setup_target(MatrixFree3D)
