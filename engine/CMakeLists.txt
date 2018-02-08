
if(MSVC)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2")
else()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.2")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

file(GLOB SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} 
  ./*.cpp
  leela/*.cpp
  leela1/*.cpp
  aq/*.cpp
  eng/*.cpp)

list(REMOVE_ITEM SOURCES ./game.cpp)

add_library(goengine SHARED ${SOURCES})


if(MSVC)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /STACK:10000000")
endif()


target_link_libraries(goengine lightmodel "-lpthread" "-lz")


set_property(TARGET goengine PROPERTY CXX_STANDARD 11)


include_directories(
../)

if(MSVC)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /STACK:10000000")
endif()

add_executable(game ./game.cpp)
set_property(TARGET game PROPERTY CXX_STANDARD 11)

target_link_libraries(game goengine)
