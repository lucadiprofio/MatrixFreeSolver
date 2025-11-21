cmake_minimum_required(VERSION 3.12.0)
project(Matrix_based_solver LANGUAGES CXX C)

include(../common/cmake-common.cmake)

add_executable(MatrixBased3D src/MatrixBased3D.cpp src/ADRMatrixBased3D.cpp)
deal_ii_setup_target(MatrixBased3D)
