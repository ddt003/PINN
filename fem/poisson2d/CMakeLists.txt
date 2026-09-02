cmake_minimum_required(VERSION 3.12.0)
project(elliptic LANGUAGES CXX C)

include(../common/cmake-common.cmake)

add_executable(elliptic src/elliptic_main.cpp src/Elliptic.cpp)
deal_ii_setup_target(elliptic)
