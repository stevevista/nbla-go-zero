
include_directories(../)

file(GLOB SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} 
  ./*.cpp)

list(REMOVE_ITEM SOURCES ./gendata.cpp)
list(REMOVE_ITEM SOURCES ./train.cpp)
list(REMOVE_ITEM SOURCES ./verify.cpp)

add_library(trainmodel STATIC ${SOURCES})
target_link_libraries(trainmodel nnabla nblapp)
if (NOT MSVC)
  target_compile_options(trainmodel PRIVATE "-std=c++11")
endif()



add_executable(gendata ./gendata.cpp)
target_link_libraries(gendata nblapp trainmodel)
target_compile_options(gendata PRIVATE "-std=c++11")

add_executable(train ./train.cpp)
target_link_libraries(train nblapp trainmodel)
target_compile_options(train PRIVATE "-std=c++11")

add_executable(verify ./verify.cpp)
target_link_libraries(verify nblapp trainmodel leela)
target_compile_options(verify PRIVATE "-std=c++11")
