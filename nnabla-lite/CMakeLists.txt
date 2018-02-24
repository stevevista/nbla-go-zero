
###############################################################################
# General settings
###############################################################################
# Using cmake scripts and modules
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/build-tools/cmake/Modules)
list(APPEND CMAKE_FIND_ROOT_PATH $ENV{CMAKE_FIND_ROOT_PATH})
include(${CMAKE_CURRENT_SOURCE_DIR}/build-tools/cmake/Utils.cmake)


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
# C++ Library
###############################################################################

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
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O4 -fPIC -Wall")
  endif()
  nbla_warnings_disable(CMAKE_CXX_FLAGS -Wno-sign-compare -Wno-uninitialized /wd4099)
  
  ###############################################################################
  # C++ core
  ###############################################################################
  # includes
  list(APPEND NBLA_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen-eigen-de14376a69f8
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/spdlog-0.13.0/include)


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
      set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-std=c++11")
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
  

  add_library(nnabla-lite SHARED ${SOURCES})
    
  target_link_libraries(nnabla-lite ${NBLA_LINKER_LIBS})

  if (NOT MSVC)
  target_compile_options(nnabla-lite PRIVATE "-std=c++11")
  endif()

  option(CONFIG_BUILD_DIR ${CMAKE_BINARY_DIR})




