


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

add_subdirectory(nnabla)
add_subdirectory(nblapp)
add_subdirectory(model)
add_subdirectory(leela)


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
               
set_property(TARGET selfplay PROPERTY CXX_STANDARD 11)
target_link_libraries(selfplay nblapp trainmodel leela X11)


macro(add_example name)
   add_executable(${name} ./${name}.cpp)
   set_property(TARGET ${name} PROPERTY CXX_STANDARD 11)
   target_link_libraries(${name} nblapp trainmodel lightmodel)
endmacro()



if(MSVC)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /STACK:10000000")
endif()

if (NOT WIN32)
add_example(gendata)
add_example(train)
add_example(verify)
add_example(board_train)
endif()

