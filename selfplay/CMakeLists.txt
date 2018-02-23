

#
cmake_minimum_required(VERSION 2.8)


include_directories(../)

add_executable(selfplay 
                ./selfplay.cpp
                ./image_loader/libpng/png.c
               ./image_loader/libpng/pngerror.c
               ./image_loader/libpng/pngget.c
               ./image_loader/libpng/pngmem.c
               ./image_loader/libpng/pngpread.c
               ./image_loader/libpng/pngread.c
               ./image_loader/libpng/pngrio.c
               ./image_loader/libpng/pngrtran.c
               ./image_loader/libpng/pngrutil.c
               ./image_loader/libpng/pngset.c
               ./image_loader/libpng/pngtrans.c
               ./image_loader/zlib/adler32.c
               ./image_loader/zlib/crc32.c
               ./image_loader/zlib/infback.c
               ./image_loader/zlib/inffast.c
               ./image_loader/zlib/inflate.c
               ./image_loader/zlib/inftrees.c
               ./image_loader/zlib/trees.c
               ./image_loader/zlib/zutil.c
               ./image_loader/png_loader.cpp
               ./threads/threaded_object_extension.cpp
                ./threads/threads_kernel_1.cpp
                ./threads/threads_kernel_2.cpp
                ./threads/threads_kernel_shared.cpp
                ./gui_core/gui_core_kernel_1.cpp
                ./gui_core/gui_core_kernel_2.cpp
                ./ui.cpp)
               
target_link_libraries(selfplay nblapp trainmodel leela X11)
target_compile_options(selfplay PUBLIC "-std=c++14")
