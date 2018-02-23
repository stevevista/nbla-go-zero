
###############################################################################
# General settings
###############################################################################
# Using cmake scripts and modules
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/build-tools/cmake/Modules)
list(APPEND CMAKE_FIND_ROOT_PATH $ENV{CMAKE_FIND_ROOT_PATH})
include(${CMAKE_CURRENT_SOURCE_DIR}/build-tools/cmake/Utils.cmake)

###############################################################################
option(BUILD_CPP_LIB "Build C++ Library" ON)
option(BUILD_CPP_TEST "Build testing" OFF)
option(BUILD_CPP_UTILS "Build C++ API and utilities" OFF)

option(BUILD_PYTHON_PACKAGE "Build python package" OFF)

option(NBLA_VERBOSE_MEMORY_USAGE "Enable debug print of memory usage" OFF)
if(NBLA_VERBOSE_MEMORY_USAGE)
  add_definitions(-DNBLA_VERBOSE_MEMORY_USAGE)
endif()

###############################################################################
# Settings
###############################################################################
if (NOT PYTHON_COMMAND_NAME)
  set(PYTHON_COMMAND_NAME "python${PYTHON_VERSION}")
endif()

find_program(PYTHON_COMMAND ${PYTHON_COMMAND_NAME})
if(NOT EXISTS ${PYTHON_COMMAND})
  message(FATAL_ERROR, "python ${PYTHON_COMMAND_NAME} not found. Please install python or check your settings.")
  error() # Place Unknown command to stop cmake process here.
endif()

###############################################################################
# Preprocessing
###############################################################################
execute_process(
  COMMAND ${PYTHON_COMMAND}
          ${CMAKE_CURRENT_SOURCE_DIR}/build-tools/cmake/get_setup_build_dir.py
          OUTPUT_VARIABLE NBLA_SETUP_LIB_DIR_RAW)
string(STRIP ${NBLA_SETUP_LIB_DIR_RAW} NBLA_SETUP_LIB_DIR)

###############################################################################
# Code generator
###############################################################################
file(MAKE_DIRECTORY  ${CMAKE_CURRENT_SOURCE_DIR}/src/nbla/proto)

execute_process(
  COMMAND
  ${PYTHON_COMMAND}
  ${CMAKE_CURRENT_SOURCE_DIR}/build-tools/code_generator/generate.py)


###############################################################################
# C++ Library
###############################################################################
if(BUILD_CPP_LIB)
  ###############################################################################
  # Third party libraries
  ###############################################################################
  add_subdirectory(third_party)
  
  if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Build type release is default on single-configuration build system like GnuMake."
      FORCE)
  endif()
  
  # Flags
  if(MSVC)
    # Force to always compile with W4
    if(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
      string(REGEX REPLACE "/W[0-4]" "/W0" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    else()
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W0")
    endif()
  endif()
  
  if(UNIX OR APPLE)
    find_program(CCACHE_FOUND ccache)
    if(CCACHE_FOUND)
      set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
      set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
    endif(CCACHE_FOUND)
    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O4 -fPIC -Wall -std=gnu++0x")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O4 -fPIC -Wall")
  endif()
  nbla_warnings_disable(CMAKE_CXX_FLAGS -Wno-sign-compare -Wno-uninitialized /wd4099)
  
  ###############################################################################
  # C++ core
  ###############################################################################
  # includes
  list(APPEND NBLA_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen-eigen-de14376a69f8
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/spdlog-0.13.0/include)
  
  if(BUILD_CPP_UTILS)
    list(APPEND NBLA_INCLUDE_DIRS
      ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cmdline-master)
  endif()

  include_directories(${NBLA_INCLUDE_DIRS};${PROJECT_BINARY_DIR})

  find_package(CUDA)

  find_package(CUDA)
  if (CUDA_FOUND)
    find_package(cuDNN REQUIRED)
    # Get CuDNN version
    set(file ${PROJECT_BINARY_DIR}/detect_cudnn_version.cpp)
    file(WRITE ${file} ""
      "#include <cudnn.h>\n"
      "#include <cstdio>\n"
      "int main()\n"
      "{\n"
      "  std::printf(\"%d\", CUDNN_VERSION / 100);\n"
      "  return 0;\n"
      "}\n")
    try_run(run_result compile_result ${PROJECT_BINARY_DIR} ${file}
      CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${CUDA_INCLUDE_DIRS}"
      LINK_LIBRARIES ${CUDA_LIBRARIES}
      RUN_OUTPUT_VARIABLE CUDNN_VERSION)


    add_definitions(-DWITH_CUDA)
    add_definitions(-DWITH_CUDNN)

    message("-- CUDA--")
    message("Version: " ${CUDA_VERSION})
    message("Runtime: " ${CUDA_CUDART_LIBRARY})
    message("CUBLAS: " ${CUDA_CUBLAS_LIBRARIES})
    message("CURAND: " ${CUDA_curand_LIBRARY})
    message("CUFFT: " ${CUDA_CUFFT_LIBRARIES})
    set(NBLA_CUDA_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS})
    set(NBLA_CUDA_LINKER_LIBS ${CUDA_CUDART_LIBRARY})
    list(APPEND NBLA_CUDA_LINKER_LIBS
      ${CUDA_CUBLAS_LIBRARIES}
      ${CUDA_curand_LIBRARY}
      )

    message("cuDNN-libs: " ${CUDNN_LIBRARIES})
    message("cuDNN-includes: " ${CUDNN_INCLUDE_DIRS})
    message("cuDNN version: " ${CUDNN_VERSION})
    list(APPEND NBLA_CUDA_INCLUDE_DIRS ${CUDNN_INCLUDE_DIRS})
    list(APPEND NBLA_CUDA_LINKER_LIBS ${CUDNN_LIBRARIES})

    # Distributed Training
    option(WITH_NCCL "Use nccl for distributed training" OFF)
    if(WITH_NCCL)
      add_definitions(-DFEATURE_DIST_TRAIN)
      find_package(NCCL REQUIRED)
      find_package(MPI REQUIRED)
      list(APPEND NBLA_CUDA_INCLUDE_DIRS 
        ${NCCL_INCLUDE_DIR}
        ${MPI_INCLUDE_PATH}
        )
      list(APPEND NBLA_CUDA_LINKER_LIBS
        ${NCCL_LIBRARIES}
        ${MPI_LIBRARIES}
        )
    endif()

    include_directories(${NBLA_CUDA_INCLUDE_DIRS})
    message("CUDA libs: ${NBLA_CUDA_LINKER_LIBS}")
    message("CUDA includes: ${NBLA_CUDA_INCLUDE_DIRS}")
    include(${CMAKE_CURRENT_SOURCE_DIR}/build-tools/cmake/select_compute_arch.cmake)
    CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS ${CUDA_SELECT_NVCC_ARCH_ARG})
    message("Arch: ${ARCH_FLAGS}")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};${ARCH_FLAGS}")

    if(MSVC)
      set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-Xcompiler /W0")
    else()
      #set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-std=c++11")
      set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}; --default-stream per-thread")
    endif()
      
    file(GLOB CU_SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} src/nbla/cuda/*/*.cu src/nbla/cuda/cudnn/*/*.cu)

    # Distributed Training
    if(NOT WITH_NCCL)
      list(REMOVE_ITEM CU_SOURCES src/nbla/cuda/communicator/data_parallel_communicator.cu)
      list(REMOVE_ITEM CU_SOURCES src/nbla/cuda/communicator/multi_process_data_parallel_communicator.cu)
    endif()

    cuda_compile(CU_OBJECTS
      ${CU_SOURCES}
      )

    list(APPEND NBLA_LINKER_LIBS ${NBLA_CUDA_LINKER_LIBS})
  endif()
  


  file(GLOB SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} 
    src/nbla/*.cpp
    src/nbla/array/*.cpp
    src/nbla/solver/*.cpp
    src/nbla/communicator/*.cpp
    src/nbla/function/*.cpp
    src/nbla/utils/*.cpp
    src/nbla/computation_graph/*.cpp)

  if (CUDA_FOUND)
    file(GLOB CUDA_SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} 
      src/nbla/cuda/*.cpp 
      src/nbla/cuda/array/*.cpp 
      src/nbla/cuda/cudnn/*.cpp 
      src/nbla/cuda/utils/*.cpp)

    list(APPEND SOURCES ${CUDA_SOURCES} ${CU_OBJECTS})
  endif()
  

  add_library(nnabla SHARED ${SOURCES})
    
  target_link_libraries(nnabla ${NBLA_LINKER_LIBS})

  if (NOT MSVC)
  target_compile_options(nnabla PRIVATE "-std=c++11")
  endif()

  set(NBLA_LIBRARY_NAME nnabla)
  
  ###############################################################################
  # C++ Utilities
  ###############################################################################
  if(BUILD_CPP_UTILS)
    add_subdirectory(src/nbla_utils)
    add_subdirectory(src/nbla_cli)
    add_subdirectory(examples/cpp)
  endif()
    
  ###############################################################################
  # C++ Testing
  ###############################################################################
  if(BUILD_TEST)
    if (WIN32)
      set(BUILD_SHARED_LIBS ON CACHE BOOL
        "Force to build gtest with dll on Windows.")
    endif()
    enable_testing()
    include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
    file(GLOB NBLA_TEST_SOURCES src/nbla/test/test_*.cpp)
    add_executable(cpplibtest ${NBLA_TEST_SOURCES})
    add_dependencies(cpplibtest ${NBLA_LIBRARY_NAME})
    target_link_libraries(cpplibtest gtest gtest_main)
    target_link_libraries(cpplibtest ${NBLA_LIBRARY_NAME})
    set_property(TARGET cpplibtest PROPERTY CXX_STANDARD 11)
    nbla_exclude_from_all(cpplibtest)  # Exclude target from all or default build
    add_test(NAME cpplibtest COMMAND cpplibtest)
  endif()

  ###############################################################################
  # Generate setup.cfg
  ###############################################################################
  if (NOT (DEFINED NBLA_PYTHON_BUILD_EXT_COMPILER))
    if (WIN32)
      set(NBLA_PYTHON_BUILD_EXT_COMPILER msvc)
    elseif(UNIX OR APPLE)
      set(NBLA_PYTHON_BUILD_EXT_COMPILER unix)
    else ()
      message(FATAL_ERROR
        "Python build_ext compiler inference is only supported on Win, Unix or Apple.")
    endif()
    message("Python build_ext compiler is infered as '${NBLA_PYTHON_BUILD_EXT_COMPILER}'.")
    message("You can specify a compiler manually setting a variable"
      " NBLA_PYTHON_BUILD_EXT_COMPILER. You can see a list of supported"
      " compiler by `python setup.py build_ext --help-compiler`.")
  endif()

  option(CONFIG_BUILD_DIR ${CMAKE_BINARY_DIR})

  
endif() # BUILD_CPP_LIB

###############################################################################
# Python API
###############################################################################
if(BUILD_PYTHON_PACKAGE)

  if(NOT BUILD_CPP_LIB)
    if(NOT IS_DIRECTORY ${CPPLIB_BUILD_DIR})
      message(FATAL_ERROR "Invalid CPPLIB_BUILD_DIR :[${CPPLIB_BUILD_DIR}]")
    endif()
    if(NOT EXISTS ${CPPLIB_LIBRARY})
      if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
	string(REPLACE .so .dylib CPPLIB_LIBRARY ${CPPLIB_LIBRARY})
	if(NOT EXISTS ${CPPLIB_LIBRARY})
	  message(FATAL_ERROR "Invalid CPPLIB_LIBRARY :[${CPPLIB_LIBRARY}]")
	endif()
      else()
	message(FATAL_ERROR "Invalid CPPLIB_LIBRARY :[${CPPLIB_LIBRARY}]")
      endif()
    endif()
    
    set(CPPLIB_LIBRARY_NAME "nnabla")
    set(CPPLIB_TARGET_FILE ${CPPLIB_LIBRARY})
    get_filename_component(CPPLIB_TARGET_FILE_NAME ${CPPLIB_TARGET_FILE} NAME)
    
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/python/setup.cfg.build-wheel.in
      ${CMAKE_CURRENT_SOURCE_DIR}/python/setup.cfg)
  endif()

  set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test/solver/__pycache__
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test/__pycache__
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test/function/__pycache__
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test/utils/__pycache__
    ${CMAKE_BINARY_DIR}/"build")
  # http://stackoverflow.com/questions/13298504/using-cmake-with-setup-py
  file(GLOB_RECURSE NBLA_PYTHON_DEPS
    ${CMAKE_CURRENT_SOURCE_DIR}/python/src/*.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/src/*.pyx
    ${CMAKE_CURRENT_SOURCE_DIR}/python/src/*.pxd)
  file(GLOB_RECURSE NBLA_PYTHON_BUILD_OUTPUT_CYTHON
    ${CMAKE_CURRENT_SOURCE_DIR}/python/src/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/python/src/*.h)
  
  set(NBLA_PYTHON_SETUP ${CMAKE_CURRENT_SOURCE_DIR}/python/setup.py)
  set(NBLA_PYTHON_OUTPUT ${CMAKE_BINARY_DIR}/build)
  set(NBLA_PYTHON_OUTPUT_WHEEL ${NBLA_PYTHON_OUTPUT}/.timestamp.wheel)
  
  # Wheel
  if(UNIX)
    add_custom_command(OUTPUT ${NBLA_PYTHON_OUTPUT_WHEEL}
      COMMAND ${CMAKE_COMMAND} -E remove -f ${NBLA_PYTHON_OUTPUT_CYTHON}
      COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_BINARY_DIR}/dist/*.whl
      COMMAND ${PYTHON_COMMAND} ${NBLA_PYTHON_SETUP} -q bdist_wheel
      COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/build-tools/auditwheel-nnabla ${CMAKE_BINARY_DIR}/dist/*-linux_*.whl
      COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_BINARY_DIR}/dist/*-linux*.whl
      COMMAND ${CMAKE_COMMAND} -E touch ${NBLA_PYTHON_OUTPUT_WHEEL}
      DEPENDS ${NBLA_PYTHON_DEPS} ${NBLA_LIBRARY_NAME} ${NBLA_PYTHON_OUTPUT_BUILD})
  else()
    add_custom_command(OUTPUT ${NBLA_PYTHON_OUTPUT_WHEEL}
      COMMAND ${CMAKE_COMMAND} -E remove -f ${NBLA_PYTHON_OUTPUT_CYTHON}
      COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_BINARY_DIR}/dist/*.whl
      COMMAND ${PYTHON_COMMAND} ${NBLA_PYTHON_SETUP} -q bdist_wheel
      COMMAND ${CMAKE_COMMAND} -E touch ${NBLA_PYTHON_OUTPUT_WHEEL}
      DEPENDS ${NBLA_PYTHON_DEPS} ${NBLA_LIBRARY_NAME} ${NBLA_PYTHON_OUTPUT_BUILD})
  endif()
  if(BUILD_CPP_LIB)
    add_custom_target(wheel ALL DEPENDS ${NBLA_PYTHON_OUTPUT_WHEEL} ${NBLA_PYTHON_OUTPUT_BUILD})
    add_dependencies(wheel ${NBLA_LIBRARY_NAME})
  else()
    add_custom_target(wheel DEPENDS ${NBLA_PYTHON_OUTPUT_WHEEL} ${NBLA_PYTHON_OUTPUT_BUILD})
  endif()

  if(UNIX OR APPLE)
    add_custom_target(benchmark
      COMMAND cd ${CMAKE_CURRENT_SOURCE_DIR}/python/benchmark &&
      LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/lib:$ENV{LD_LIBRARY_PATH}
      PYTHONPATH=${CMAKE_BINARY_DIR}/build/${NBLA_SETUP_LIB_DIR}:$ENV{PYTHONPATH}
      bash -c "py.test -x")
    add_dependencies(benchmark wheel)
  endif()
  
  #############################################################################
  # Document
  #############################################################################
  if(UNIX OR APPLE)
    add_custom_target(doc
      COMMAND rm -rf ${CMAKE_BINARY_DIR}/doc
      COMMAND mkdir -p ${CMAKE_BINARY_DIR}/doc/doxygen
      COMMAND cd ${CMAKE_CURRENT_SOURCE_DIR} &&
              cat build-tools/doxygen/config >Doxyfile &&
              echo OUTPUT_DIRECTORY  = ${CMAKE_BINARY_DIR}/doc/doxygen >>Doxyfile &&
              doxygen && rm -f Doxyfile
      COMMAND mkdir -p ${CMAKE_BINARY_DIR}/doc/sphinx
      COMMAND cd ${CMAKE_CURRENT_SOURCE_DIR} &&
              PYTHONPATH=${CMAKE_BINARY_DIR}/build/${NBLA_SETUP_LIB_DIR}:$ENV{PYTHONPATH}
              sphinx-build -M html doc ${CMAKE_BINARY_DIR}/doc/sphinx -c doc
      COMMAND mkdir -p ${CMAKE_BINARY_DIR}/doc/protoc
      COMMAND mv ${CMAKE_BINARY_DIR}/doc/sphinx/doctrees ${CMAKE_BINARY_DIR}/doc
      COMMAND mv ${CMAKE_BINARY_DIR}/doc/sphinx/html ${CMAKE_BINARY_DIR}/doc
      COMMAND rm -rf ${CMAKE_BINARY_DIR}/doc/sphinx
      COMMAND mv ${CMAKE_BINARY_DIR}/doc/doxygen/html ${CMAKE_BINARY_DIR}/doc/html/doxygen
      COMMAND rm -rf ${CMAKE_BINARY_DIR}/doc/doxygen)
    add_dependencies(doc wheel)
    nbla_exclude_from_all(doc)
  endif()

endif() # BUILD_PYTHON_PACKAGE

