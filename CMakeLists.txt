

#
cmake_minimum_required(VERSION 2.8)

## cmake .. -G "Visual Studio 14 2015 Win64" -T host=x64 
## cmake --build . --config Release

project (dnn-go C CXX)

  if(MSVC)
    # Force to always compile with W4
    if(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
      string(REGEX REPLACE "/W[0-4]" "/W0" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    else()
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W0")
    endif()
  endif()
  
# Setting output directory naively
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
  if (WIN32)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
  else()
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
  endif()

  
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O4 -fPIC -Wall -std=c++11")

add_subdirectory(leela)
add_subdirectory(nnabla)
add_subdirectory(nblapp)
add_subdirectory(model)



include_directories(./)
include_directories(../)

add_executable(selfplay 
                ./selfplay.cpp
                ./dlib/image_loader/libpng/png.c
               ./dlib/image_loader/libpng/pngerror.c
               ./dlib/image_loader/libpng/pngget.c
               ./dlib/image_loader/libpng/pngmem.c
               ./dlib/image_loader/libpng/pngpread.c
               ./dlib/image_loader/libpng/pngread.c
               ./dlib/image_loader/libpng/pngrio.c
               ./dlib/image_loader/libpng/pngrtran.c
               ./dlib/image_loader/libpng/pngrutil.c
               ./dlib/image_loader/libpng/pngset.c
               ./dlib/image_loader/libpng/pngtrans.c
               ./dlib/image_loader/zlib/adler32.c
               ./dlib/image_loader/zlib/crc32.c
               ./dlib/image_loader/zlib/infback.c
               ./dlib/image_loader/zlib/inffast.c
               ./dlib/image_loader/zlib/inflate.c
               ./dlib/image_loader/zlib/inftrees.c
               ./dlib/image_loader/zlib/trees.c
               ./dlib/image_loader/zlib/zutil.c
               ./dlib/image_loader/png_loader.cpp
               ./dlib/threads/threaded_object_extension.cpp
                ./dlib/threads/threads_kernel_1.cpp
                ./dlib/threads/threads_kernel_2.cpp
                ./dlib/threads/threads_kernel_shared.cpp
                ./dlib/gui_core/gui_core_kernel_1.cpp
                ./dlib/gui_core/gui_core_kernel_2.cpp
                ./dlib/ui.cpp)
               
target_link_libraries(selfplay nblapp trainmodel leela X11)
target_compile_options(selfplay PUBLIC "-std=c++14")

macro(add_example name)
   add_executable(${name} ./${name}.cpp)
   set_property(TARGET ${name} PROPERTY CXX_STANDARD 11)
   target_link_libraries(${name} nblapp trainmodel leela)
endmacro()

add_example(gendata)
add_example(train)
add_example(verify)


