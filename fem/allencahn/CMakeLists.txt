cmake_minimum_required(VERSION 3.12.0)
project(parabolic LANGUAGES CXX C)

include(../common/cmake-common.cmake)

add_executable(parabolic src/parabolic_main.cpp src/Parabolic.cpp)
deal_ii_setup_target(parabolic)
