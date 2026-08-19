cmake_minimum_required(VERSION 3.12.0)

include(./cmake-common.cmake)

add_executable(fem-solver ./main.cpp ./solver.cpp)
deal_ii_setup_target(fem-solver)
