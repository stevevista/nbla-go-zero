
cmake_minimum_required(VERSION 2.8.12)
project(cudnn_test)

find_package(CUDA 7.5 REQUIRED)
set(CUDA_HOST_COMPILATION_CPP ON)
list(APPEND CUDA_NVCC_FLAGS "-arch=sm_30;-std=c++11;-D__STRICT_ANSI__")
add_definitions(-DDLIB_USE_CUDA)

include(find_cudnn.txt)

if (cudnn_include AND cudnn)
   include_directories(${cudnn_include})
   cuda_add_library(cudnn_test STATIC ../../nn/dnn/cudnn_dlibapi.cpp ${cudnn} )
   target_compile_options(cudnn_test PUBLIC "-std=c++11")
endif()
