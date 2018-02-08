
include_directories(
../)

file(GLOB SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} 
  ./*.cpp
  ./dnn/cpu_dlib.cpp
  ./dnn/tensor_tools.cpp)

list(REMOVE_ITEM SOURCES ./game.cpp)

option(DLIB_USE_BLAS "using blas" OFF)

find_package(CUDA)

if (CUDA_FOUND)
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

  ####################################################################################################
  # cuDNN

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
  add_definitions(-DDLIB_USE_CUDA)
  message("cuDNN-libs: " ${CUDNN_LIBRARIES})
  message("cuDNN-includes: " ${CUDNN_INCLUDE_DIRS})
  message("cuDNN version: " ${CUDNN_VERSION})
  list(APPEND NBLA_CUDA_INCLUDE_DIRS ${CUDNN_INCLUDE_DIRS})
  list(APPEND NBLA_CUDA_LINKER_LIBS ${CUDNN_LIBRARIES})

  include_directories(${NBLA_CUDA_INCLUDE_DIRS})
  message("CUDA libs: ${NBLA_CUDA_LINKER_LIBS}")
  message("CUDA includes: ${NBLA_CUDA_INCLUDE_DIRS}")
  include(${CMAKE_SOURCE_DIR}/build-tools/cmake/select_compute_arch.cmake)
  CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS ${CUDA_SELECT_NVCC_ARCH_ARG})
  message("Arch: ${ARCH_FLAGS}")
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};${ARCH_FLAGS}")

  if(MSVC)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-Xcompiler /W0")
  else()
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-std=c++11")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}; --default-stream per-thread")
  endif()

  cuda_compile(CU_OBJECTS
    ./dnn/cuda_dlib.cu
    )

  file(GLOB CUDA_SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} 
        ./dnn/gpu_data.cpp
        ./dnn/cublas_dlibapi.cpp
        ./dnn/cuda_data_ptr.cpp
        ./dnn/cudnn_dlibapi.cpp)

  list(APPEND SOURCES ${CUDA_SOURCES} ${CU_OBJECTS})
  list(APPEND NBLA_LINKER_LIBS ${NBLA_CUDA_LINKER_LIBS})
endif()

if (WIN32)
	set(mkl_search_path  "C:\\Program Files (x86)\\IntelSWTools\\compilers_and_libraries\\windows\\mkl\\lib\\intel64" )
	find_library(mkl_core mkl_core ${mkl_search_path})
	find_library(mkl_intel_lp64 mkl_intel_lp64 ${mkl_search_path})
	find_library(mkl_sequential mkl_sequential ${mkl_search_path})
	
	if (mkl_core)
		set(DLIB_USE_BLAS ON)
	endif()
else()
	find_library(openblas openblas /usr/lib64)
	if (openblas)
		set(DLIB_USE_BLAS ON)
	endif()
endif()

if (DLIB_USE_BLAS)
  add_definitions(-DDLIB_USE_BLAS)
  add_definitions(-DDLIB_USE_LAPACK)
  if (WIN32)
	list(APPEND NBLA_LINKER_LIBS ${mkl_core} ${mkl_intel_lp64} ${mkl_sequential})
  else()
	list(APPEND NBLA_LINKER_LIBS ${openblas})
  endif()
endif()


add_library(simplemodel SHARED ${SOURCES})
target_link_libraries(simplemodel nengine ${NBLA_LINKER_LIBS})
set_property(TARGET simplemodel PROPERTY CXX_STANDARD 11)



if(MSVC)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /STACK:10000000")
endif()

add_executable(game ./game.cpp)
set_property(TARGET game PROPERTY CXX_STANDARD 11)


target_link_libraries(game simplemodel nengine)

