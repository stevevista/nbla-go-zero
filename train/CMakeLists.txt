cmake_minimum_required(VERSION 2.8)
project(gendata)


if(MSVC)
  add_definitions(/D_CRT_SECURE_NO_WARNINGS)
else()
  add_compile_options(-std=c++11 -Wall -Wextra -O3 -DNDEBUG)
endif()

ADD_EXECUTABLE(gendata 
    board.cpp
    archive.cpp
    gendata.cpp
    sgf.cpp
    utils.cpp)
