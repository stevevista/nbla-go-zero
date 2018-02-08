// Copyright (c) 2017 Sony Corporation. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/** Utilities for CUDA
*/
#ifndef __NBLA_CUDA_COMMON_HPP__
#define __NBLA_CUDA_COMMON_HPP__

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>

#include <nbla/common.hpp>
#include <nbla/cuda/defs.hpp>
#include <nbla/cuda/init.hpp>
#include <nbla/exception.hpp>

#include <map>

namespace nbla {

using std::map;

/** Check Kernel Execution*/
#define NBLA_CUDA_KERNEL_CHECK() NBLA_CUDA_CHECK(cudaGetLastError())

/**
Check CUDA error for synchronous call
*/
#define NBLA_CUDA_CHECK(condition)                                             \
  {                                                                            \
    cudaError_t error = condition;                                             \
    if (error != cudaSuccess) {                                                \
      NBLA_ERROR(error_code::target_specific, "(%s) failed with \"%s\".",      \
                 #condition, cudaGetErrorString(error));                       \
    }                                                                          \
  }

inline string cublas_status_to_string(cublasStatus_t status) {
#define CASE_CUBLAS_STATUS(NAME)                                               \
  case CUBLAS_STATUS_##NAME:                                                   \
    return #NAME;

  switch (status) {
    CASE_CUBLAS_STATUS(SUCCESS);
    CASE_CUBLAS_STATUS(NOT_INITIALIZED);
    CASE_CUBLAS_STATUS(ALLOC_FAILED);
    CASE_CUBLAS_STATUS(INVALID_VALUE);
    CASE_CUBLAS_STATUS(ARCH_MISMATCH);
    CASE_CUBLAS_STATUS(MAPPING_ERROR);
    CASE_CUBLAS_STATUS(EXECUTION_FAILED);
    CASE_CUBLAS_STATUS(INTERNAL_ERROR);
#if CUDA_VERSION >= 6000
    CASE_CUBLAS_STATUS(NOT_SUPPORTED);
#endif
#if CUDA_VERSION >= 6050
    CASE_CUBLAS_STATUS(LICENSE_ERROR);
#endif
  }
  return "UNKNOWN";
#undef CASE_CUBLAS_STATUS
}

/**
*/
#define NBLA_CUBLAS_CHECK(condition)                                           \
  {                                                                            \
    cublasStatus_t status = condition;                                         \
    cudaGetLastError();                                                        \
    NBLA_CHECK(status == CUBLAS_STATUS_SUCCESS, error_code::target_specific,   \
               cublas_status_to_string(status));                               \
  }

inline string curand_status_to_string(curandStatus_t status) {
#define CASE_CURAND_STATUS(NAME)                                               \
  case CURAND_STATUS_##NAME:                                                   \
    return #NAME;

  switch (status) {
    CASE_CURAND_STATUS(SUCCESS);
    CASE_CURAND_STATUS(VERSION_MISMATCH);
    CASE_CURAND_STATUS(NOT_INITIALIZED);
    CASE_CURAND_STATUS(ALLOCATION_FAILED);
    CASE_CURAND_STATUS(TYPE_ERROR);
    CASE_CURAND_STATUS(OUT_OF_RANGE);
    CASE_CURAND_STATUS(LENGTH_NOT_MULTIPLE);
    CASE_CURAND_STATUS(DOUBLE_PRECISION_REQUIRED);
    CASE_CURAND_STATUS(LAUNCH_FAILURE);
    CASE_CURAND_STATUS(PREEXISTING_FAILURE);
    CASE_CURAND_STATUS(INITIALIZATION_FAILED);
    CASE_CURAND_STATUS(ARCH_MISMATCH);
    CASE_CURAND_STATUS(INTERNAL_ERROR);
  }
  return "UNKNOWN";
#undef CASE_CURAND_STATUS
}

#define NBLA_CURAND_CHECK(condition)                                           \
  {                                                                            \
    curandStatus_t status = condition;                                         \
    NBLA_CHECK(status == CURAND_STATUS_SUCCESS, error_code::target_specific,   \
               curand_status_to_string(status));                               \
  }

/** ceil(N/D) where N and D are integers */
#define NBLA_CEIL_INT_DIV(N, D)                                                \
  ((static_cast<int>(N) + static_cast<int>(D) - 1) / static_cast<int>(D))

/** Default num threads */
#define NBLA_CUDA_NUM_THREADS 512

/** Max number of blocks per dimension*/
#define NBLA_CUDA_MAX_BLOCKS 65536

/** Block size */
#define NBLA_CUDA_GET_BLOCKS(num) NBLA_CEIL_INT_DIV(num, NBLA_CUDA_NUM_THREADS)

/** Get an appropreate block size given a size of elements.

    The kernel is assumed to contain a grid-strided loop.
 */
inline int cuda_get_blocks_by_size(int size) {
  const int blocks = NBLA_CUDA_GET_BLOCKS(size);
  const int inkernel_loop = NBLA_CEIL_INT_DIV(blocks, NBLA_CUDA_MAX_BLOCKS);
  const int total_blocks = NBLA_CEIL_INT_DIV(blocks, inkernel_loop);
  return total_blocks;
}

/** Launch simple kernel */
#define NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, size, ...)                      \
  {                                                                            \
    (kernel)<<<cuda_get_blocks_by_size(size), NBLA_CUDA_NUM_THREADS>>>(        \
        (size), __VA_ARGS__);                                                  \
    NBLA_CUDA_KERNEL_CHECK();                                                  \
  }

/** Launch simple kernel */
#define NBLA_CUDA_LAUNCH_KERNEL_IN_STREAM(kernel, stream, size, ...)           \
  {                                                                            \
    (kernel)<<<cuda_get_blocks_by_size(size), NBLA_CUDA_NUM_THREADS, 0,        \
               (stream)>>>((size), __VA_ARGS__);                               \
    NBLA_CUDA_KERNEL_CHECK();                                                  \
  }

/** Cuda grid-strided loop */
#define NBLA_CUDA_KERNEL_LOOP(idx, num)                                        \
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < (num);           \
       idx += blockDim.x * gridDim.x)

/** Instantiate template CUDA functions */
#define NBLA_INSTANTIATE_CUDA_FUNCS(type, classname)                           \
  template void classname<type>::forward_impl(const Variables &inputs,         \
                                              const Variables &outputs);       \
  template void classname<type>::backward_impl(                                \
      const Variables &inputs, const Variables &outputs,                       \
      const vector<bool> &propagate_down);

/** CUDA device setter
@return index of device before change
*/
int cuda_set_device(int device);
/** Get current CUDA device.
@return index of device
*/
int cuda_get_device();
}
#endif
