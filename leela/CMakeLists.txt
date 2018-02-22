#
# This is a CMake makefile.  You can find the cmake utility and
# information about it at http://www.cmake.org
#
## cmake .. -G "Visual Studio 14 2015 Win64" -T host=x64 
## cmake --build . --config Release

cmake_minimum_required(VERSION 2.8)
project(leelaz-fox)


include(cmake_utils/set_compiler_specific_options.cmake)


if (USING_OLD_VISUAL_STUDIO_COMPILER)
  message("The visual studio compiler is too old.")
else()

# Adhere to GNU filesystem layout conventions
include(GNUInstallDirs)

# default to a Release build (except if CMAKE_BUILD_TYPE is set)
include(cmake_utils/release_build_by_default)


set(CPACK_PACKAGE_VERSION_MAJOR "19")
set(CPACK_PACKAGE_VERSION_MINOR "8")
set(CPACK_PACKAGE_VERSION_PATCH "0")
set(VERSION ${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH})
# Set DLIB_VERSION in the including CMake file so they can use it to do whatever they want. 

if (CMAKE_VERSION VERSION_LESS "3.9.0")
   # Set only because there are old target_link_libraries() statements in the
   # FindCUDA.cmake file that comes with CMake that error out if the new behavior
   # is used.  In newer versions of CMake we can instead set CUDA_LINK_LIBRARIES_KEYWORD which fixes this issue.
   cmake_policy(SET CMP0023 OLD)
else()
   set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)
endif()


macro (enable_preprocessor_switch option_name)
   list(APPEND active_preprocessor_switches "-D${option_name}")
endmacro()

macro (disable_preprocessor_switch option_name)
   if (active_preprocessor_switches)
      list(REMOVE_ITEM active_preprocessor_switches "-D${option_name}")
   endif()
endmacro()

macro (toggle_preprocessor_switch option_name)
   if (${option_name})
      enable_preprocessor_switch(${option_name})
   else()
      disable_preprocessor_switch(${option_name})
   endif()
endmacro()



# Suppress superfluous randlib warnings about libdlib.a having no symbols on MacOSX.
if (APPLE)
    set(CMAKE_C_ARCHIVE_CREATE   "<CMAKE_AR> Scr <TARGET> <LINK_FLAGS> <OBJECTS>")
    set(CMAKE_CXX_ARCHIVE_CREATE "<CMAKE_AR> Scr <TARGET> <LINK_FLAGS> <OBJECTS>")
    set(CMAKE_C_ARCHIVE_FINISH   "<CMAKE_RANLIB> -no_warning_for_no_symbols -c <TARGET>")
    set(CMAKE_CXX_ARCHIVE_FINISH "<CMAKE_RANLIB> -no_warning_for_no_symbols -c <TARGET>")
endif()

# Don't try to call add_library(dlib) and setup dlib's stuff if it has already
# been done by some other part of the current cmake project.  We do this
# because it avoids getting warnings/errors about cmake policy CMP0002.  This
# happens when a project tries to call add_subdirectory() on dlib more than
# once.  This most often happens when the top level of a project depends on two
# or more other things which both depend on dlib. 

   set (DLIB_ENABLE_STACK_TRACE_STR 
   "Enable this if you want to turn on the DLIB_STACK_TRACE macros" )
   set (DLIB_USE_BLAS_STR
   "Disable this if you don't want to use a BLAS library" )
   set (DLIB_USE_LAPACK_STR
   "Disable this if you don't want to use a LAPACK library" )
   set (DLIB_USE_CUDA_STR
   "Disable this if you don't want to use NVIDIA CUDA" )
   set (DLIB_ENABLE_ASSERTS_STR
   "Enable this if you want to turn on the DLIB_ASSERT macro" )


   option(DLIB_ENABLE_ASSERTS ${DLIB_ENABLE_ASSERTS_STR} OFF)
   option(DLIB_ENABLE_STACK_TRACE ${DLIB_ENABLE_STACK_TRACE_STR} OFF)
   toggle_preprocessor_switch(DLIB_ENABLE_STACK_TRACE)

   if(DLIB_ENABLE_ASSERTS)
      # Set these variables so they are set in the config.h.in file when dlib
      # is installed.
      set (DLIB_DISABLE_ASSERTS false)
      set (ENABLE_ASSERTS true) 
      enable_preprocessor_switch(ENABLE_ASSERTS)
      disable_preprocessor_switch(DLIB_DISABLE_ASSERTS)
   else()
      # Set these variables so they are set in the config.h.in file when dlib
      # is installed.
      set (DLIB_DISABLE_ASSERTS true)
      set (ENABLE_ASSERTS false) 
      disable_preprocessor_switch(ENABLE_ASSERTS)
      enable_preprocessor_switch(DLIB_DISABLE_ASSERTS)
   endif()

  option(DLIB_USE_BLAS ${DLIB_USE_BLAS_STR} ON)
  option(DLIB_USE_LAPACK ${DLIB_USE_LAPACK_STR} ON)
  option(DLIB_USE_CUDA ${DLIB_USE_CUDA_STR} ON)
 
  toggle_preprocessor_switch(DLIB_USE_BLAS)
  toggle_preprocessor_switch(DLIB_USE_LAPACK)
  toggle_preprocessor_switch(DLIB_USE_CUDA)


add_definitions(-DHAVE_PTHREAD)
file(GLOB SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} 
    ./*.cpp)

set(source_files 
    nn/dnn/cpu_dlib.cpp
    nn/dnn/tensor_tools.cpp
    ${SOURCES})

LIST(REMOVE_ITEM source_files src/leela/Leela.cpp)


include_directories(./stubs)

set(dlib_needed_libraries)
set(dlib_needed_includes)

if(UNIX)
  set(CMAKE_THREAD_PREFER_PTHREAD ON)
  find_package(Threads REQUIRED)
  set(dlib_needed_libraries ${dlib_needed_libraries} ${CMAKE_THREAD_LIBS_INIT})
endif()

      # we want to link to the right stuff depending on our platform.  
      if (WIN32 AND NOT CYGWIN) ###############################################################################
        set (dlib_needed_libraries ws2_32 winmm)
      endif () ##################################################################################

      INCLUDE (CheckFunctionExists)

      if (DLIB_USE_BLAS OR DLIB_USE_LAPACK)
          # Try to find BLAS, LAPACK and MKL
         include(cmake_utils/cmake_find_blas.txt)

         if (DLIB_USE_BLAS)
            if (blas_found)
               set (dlib_needed_libraries ${dlib_needed_libraries} ${blas_libraries})
            else()
               set(DLIB_USE_BLAS OFF CACHE STRING ${DLIB_USE_BLAS_STR} FORCE )
               toggle_preprocessor_switch(DLIB_USE_BLAS)
            endif()
         endif()

         if (DLIB_USE_LAPACK)
            if (lapack_found)
               set (dlib_needed_libraries ${dlib_needed_libraries} ${lapack_libraries})
               if (lapack_with_underscore)
                  set(LAPACK_FORCE_UNDERSCORE 1)
                  enable_preprocessor_switch(LAPACK_FORCE_UNDERSCORE)
               elseif (lapack_without_underscore)
                  set(LAPACK_FORCE_NOUNDERSCORE 1)
                  enable_preprocessor_switch(LAPACK_FORCE_NOUNDERSCORE)
               endif ()
            else()
               set(DLIB_USE_LAPACK OFF CACHE STRING ${DLIB_USE_LAPACK_STR} FORCE )
               toggle_preprocessor_switch(DLIB_USE_LAPACK)
            endif()
         endif()
      endif()


      if (DLIB_USE_CUDA)
         find_package(CUDA 7.5)

         if (CUDA_FOUND AND MSVC AND NOT CUDA_CUBLAS_LIBRARIES AND "${CMAKE_SIZEOF_VOID_P}" EQUAL "4")
            message(WARNING "You have CUDA installed, but we can't use it unless you put visual studio in 64bit mode.")
            set(CUDA_FOUND 0)
         endif()

         if (CUDA_FOUND AND (NOT USING_OLD_VISUAL_STUDIO_COMPILER))

            # There is some bug in cmake that causes it to mess up the
            # -std=c++11 option if you let it propagate it to nvcc in some
            # cases.  So instead we disable this and manually include
            # things from CMAKE_CXX_FLAGS in the CUDA_NVCC_FLAGS list below.
            if (APPLE)
               set(CUDA_PROPAGATE_HOST_FLAGS OFF)
               # Grab all the -D flags from CMAKE_CXX_FLAGS so we can pass them
               # to nvcc.
               string(REGEX MATCHALL "-D[^ ]*" FLAGS_FOR_NVCC "${CMAKE_CXX_FLAGS}")
            endif()


            set(CUDA_HOST_COMPILATION_CPP ON)
            # Note that we add __STRICT_ANSI__ to avoid freaking out nvcc with gcc specific
            # magic in the standard C++ header files (since nvcc uses gcc headers on
            # linux).
            list(APPEND CUDA_NVCC_FLAGS "-arch=sm_30;-D__STRICT_ANSI__;-D_MWAITXINTRIN_H_INCLUDED;-D_FORCE_INLINES;${FLAGS_FOR_NVCC}")
            list(APPEND CUDA_NVCC_FLAGS ${active_preprocessor_switches})
            if (NOT MSVC)
               list(APPEND CUDA_NVCC_FLAGS "-std=c++11")
            endif()
            list(APPEND CUDA_NVCC_FLAGS "-Xcompiler -fPIC")
            if (CMAKE_POSITION_INDEPENDENT_CODE)
               # sometimes this setting isn't propagated to NVCC, which then causes the
               # compile to fail.  So make sure it's propagated.
               if (NOT MSVC) # Visual studio doesn't have -fPIC so don't do it in that case.
                  list(APPEND CUDA_NVCC_FLAGS "-Xcompiler -fPIC")
               endif()
            endif()

            include(cmake_utils/test_for_cudnn/find_cudnn.txt)

            if (cudnn AND cudnn_include AND NOT DEFINED cuda_test_compile_worked AND NOT DEFINED cudnn_test_compile_worked)
               # make sure cuda is really working by doing a test compile
               message(STATUS "Building a CUDA test project to see if your compiler is compatible with CUDA...")

               set(CUDA_TEST_CMAKE_FLAGS "-DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}"
                                         "-DCMAKE_INCLUDE_PATH=${CMAKE_INCLUDE_PATH}"
                                         "-DCMAKE_LIBRARY_PATH=${CMAKE_LIBRARY_PATH}")

               if (NOT MSVC) # see https://github.com/davisking/dlib/issues/363
                  list(APPEND CUDA_TEST_CMAKE_FLAGS "-DCUDA_HOST_COMPILER=${CUDA_HOST_COMPILER}")
               endif()

               try_compile(cuda_test_compile_worked ${PROJECT_BINARY_DIR}/cuda_test_build 
                           ${PROJECT_SOURCE_DIR}/cmake_utils/test_for_cuda cuda_test
                           CMAKE_FLAGS ${CUDA_TEST_CMAKE_FLAGS}
                           )
               if (NOT cuda_test_compile_worked)
                  message(STATUS "*** CUDA was found but your compiler failed to compile a simple CUDA program so dlib isn't going to use CUDA. ***")
               else()
                  message(STATUS "Checking if you have the right version of cuDNN installed.")
                  try_compile(cudnn_test_compile_worked ${PROJECT_BINARY_DIR}/cudnn_test_build 
                              ${PROJECT_SOURCE_DIR}/cmake_utils/test_for_cudnn cudnn_test
                              CMAKE_FLAGS ${CUDA_TEST_CMAKE_FLAGS}
                                          )
                  if (NOT cudnn_test_compile_worked)
                     message(STATUS "*** Found cuDNN, but it looks like the wrong version so dlib will not use it.              ***")
                     message(STATUS "*** Dlib requires cuDNN V5.0 OR GREATER.  Since cuDNN is not found DLIB WILL NOT USE CUDA. ***")
                     message(STATUS "*** If you have cuDNN then set CMAKE_PREFIX_PATH to include cuDNN's folder.                ***")
                  endif()
               endif()
            endif()

            # Find where cuSOLVER is since the FindCUDA cmake package doesn't
            # bother to look for it.
            get_filename_component(cuda_blas_path "${CUDA_CUBLAS_LIBRARIES}" DIRECTORY)
            find_library(cusolver cusolver HINTS ${cuda_blas_path})
            mark_as_advanced(cusolver)
            # Also find OpenMP since cuSOLVER needs it.  Importantly, we only
            # look for one to link to if our use of BLAS, specifically the
            # Intel MKL, hasn't already decided what to use.  This is because
            # it makes the MKL bug out if you link to another openmp lib other
            # than Intel's when you use the MKL.
            if (NOT openmp_libraries AND NOT MSVC AND NOT XCODE)
               find_package(OpenMP)
               if (OPENMP_FOUND)
                  set(openmp_libraries ${OpenMP_CXX_FLAGS}) 
               else()
                  message(STATUS "*** Didn't find OpenMP, which is required to use CUDA. ***")
                  set(CUDA_FOUND 0)
               endif()
            endif()
         endif()

         if (CUDA_FOUND AND cudnn AND cuda_test_compile_worked AND cudnn_test_compile_worked AND cudnn_include)
            set(source_files ${source_files} 
               nn/dnn/cuda_dlib.cu
               nn/dnn/cudnn_dlibapi.cpp
               nn/dnn/cublas_dlibapi.cpp
               nn/dnn/cuda_data_ptr.cpp
               nn/dnn/gpu_data.cpp
               )
            set(dlib_needed_libraries ${dlib_needed_libraries} 
                                      ${CUDA_CUBLAS_LIBRARIES} 
                                      ${cudnn}
                                      ${CUDA_curand_LIBRARY}
                                      ${cusolver}
                                      )
            if(openmp_libraries)
               list(APPEND dlib_needed_libraries ${openmp_libraries})
            endif()
            
            include_directories(${cudnn_include})
            message(STATUS "Enabling CUDA support for dlib.  DLIB WILL USE CUDA")
         else()
            set(DLIB_USE_CUDA OFF CACHE STRING ${DLIB_USE_BLAS_STR} FORCE )
            toggle_preprocessor_switch(DLIB_USE_CUDA)
            if (USING_OLD_VISUAL_STUDIO_COMPILER)
               message(STATUS "*** Dlib CUDA support requires C++11 but your compiler doesn't support it. ***")
            endif()
            message(STATUS "Disabling CUDA support for dlib.  DLIB WILL NOT USE CUDA")
         endif()
      endif()
    

if (DLIB_USE_CUDA)
  include_directories(${dlib_needed_includes})
  cuda_add_library(leela STATIC ${source_files})
else()
  add_library(leela STATIC ${source_files})
endif()

target_include_directories(leela
                              INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
                              INTERFACE $<INSTALL_INTERFACE:include>
                              PUBLIC ${dlib_needed_includes}
                              )

target_link_libraries(leela PRIVATE ${dlib_needed_libraries})
target_compile_options(leela PRIVATE ${active_preprocessor_switches})
if (NOT MSVC)
target_compile_options(leela PUBLIC "-std=c++14" "-march=native" "-ffast-math")
endif()
target_compile_options(leela PUBLIC ${active_compile_opts} "-DNDEBUG")


endif()
