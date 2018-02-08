
include_directories(../)


find_package(CUDA)
if (CUDA_FOUND)
  add_definitions(-DWITH_CUDNN)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.2")

file(GLOB SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} 
  ./*.cpp
  ./src/*.cpp)

add_library(gomodel SHARED ${SOURCES})
target_link_libraries(gomodel nnabla nblapp)
set_property(TARGET gomodel PROPERTY CXX_STANDARD 11)
