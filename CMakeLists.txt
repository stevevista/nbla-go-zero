cmake_minimum_required (VERSION 2.8)
set(CMAKE_VERBOSE_MAKEFILE ON)

# NNabla project
project (nbla C CXX)

###############################################################################
# General settings
###############################################################################
# Using cmake scripts and modules
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/build-tools/cmake/Modules)
list(APPEND CMAKE_FIND_ROOT_PATH $ENV{CMAKE_FIND_ROOT_PATH})
include(${CMAKE_SOURCE_DIR}/build-tools/cmake/Utils.cmake)

###############################################################################
option(BUILD_CPP_LIB "Build C++ Library" ON)
option(BUILD_CPP_TEST "Build testing" OFF)
option(BUILD_CPP_UTILS "Build C++ API and utilities" OFF)

option(BUILD_PYTHON_PACKAGE "Build python package" ON)

option(NBLA_VERBOSE_MEMORY_USAGE "Enable debug print of memory usage" OFF)
if(NBLA_VERBOSE_MEMORY_USAGE)
  add_definitions(-DNBLA_VERBOSE_MEMORY_USAGE)
endif()

  
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
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O4 -fPIC -Wall -std=gnu++0x")
    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall")
  endif()
  nbla_warnings_disable(CMAKE_CXX_FLAGS -Wno-sign-compare -Wno-uninitialized /wd4099)
  
  # Setting output directory naively
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
  if (WIN32)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
  else()
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
  endif()
  
  ###############################################################################
  # C++ core
  ###############################################################################
  # includes
  list(APPEND NBLA_INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party/eigen-eigen-de14376a69f8)
  
  if(BUILD_CPP_UTILS)
    list(APPEND NBLA_INCLUDE_DIRS
      ${CMAKE_SOURCE_DIR}/third_party/cmdline-master)
  endif()
  
  include_directories(${NBLA_INCLUDE_DIRS};${PROJECT_BINARY_DIR})
  add_subdirectory(src/nbla)
  add_subdirectory(apps)

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
