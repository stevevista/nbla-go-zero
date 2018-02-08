
include_directories(../)

file(GLOB SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} 
  ./*.cpp
  ./src/*.cpp)

add_library(trainmodel SHARED ${SOURCES})
target_link_libraries(trainmodel nnabla nblapp)
set_property(TARGET trainmodel PROPERTY CXX_STANDARD 11)
