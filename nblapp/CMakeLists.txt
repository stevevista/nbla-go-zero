
include_directories(../nnabla/include)


find_package(CUDA)
if (CUDA_FOUND)
  add_definitions(-DWITH_CUDNN)
endif()


file(GLOB SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} 
  ./*.cpp)


add_library(nblapp SHARED ${SOURCES})
target_link_libraries(nblapp nnabla)
if (NOT MSVC)
target_compile_options(nblapp PRIVATE "-std=c++11")
endif()

