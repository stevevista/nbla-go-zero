
if(MSVC)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2")
else()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.2")
endif()

file(GLOB SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} 
  ./*.cpp)

set(target leela)
add_library(${target} SHARED ${SOURCES})


if(MSVC)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /STACK:10000000")
endif()


target_link_libraries(${target} simplemodel "-lpthread")


set_property(TARGET ${target} PROPERTY CXX_STANDARD 11)
