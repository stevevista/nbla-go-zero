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

#ifndef __NBLA_CUDA_UTILS_SHUFFLE_DOWN_CUH__
#define __NBLA_CUDA_UTILS_SHUFFLE_DOWN_CUH__

#include <nbla/cuda/utils/types.cuh>

namespace nbla {
// Just a workaround. Performance is not optimal.
// <https://github.com/parallel-forall/code-samples/blob/master/posts/parallel_reduction_with_shfl/fake_shfl.h>
#define MAX_THREADS_PER_BLOCK 1024
template <typename T>
__inline__ __device__ T pre_fermi_shfl_down(T val, int offset, int width = 32) {
  static __shared__ T shared[MAX_THREADS_PER_BLOCK];
  int lane = threadIdx.x % 32;
  shared[threadIdx.x] = val;
  __syncthreads();
  val = (lane + offset < width) ? shared[threadIdx.x + offset] : 0;
  __syncthreads();
  return val;
}

template <typename T>
__forceinline__ __device__ T shuffle_down(T val, int offset, int width = 32) {
#if __CUDA_ARCH__ >= 300
#define SHFL_DOWN __shfl_down
#else
#define SHFL_DOWN pre_fermi_shfl_down
#endif
  return SHFL_DOWN(val, offset, width);
}

template <>
__forceinline__ __device__ float2 shuffle_down(float2 val, int offset,
                                               int width) {
  float2 buff;
  buff.x = shuffle_down(val.x, offset, width);
  buff.y = shuffle_down(val.y, offset, width);
  return buff;
}

template <>
__forceinline__ __device__ float3 shuffle_down(float3 val, int offset,
                                               int width) {
  float3 buff;
  buff.x = shuffle_down(val.x, offset, width);
  buff.y = shuffle_down(val.y, offset, width);
  buff.z = shuffle_down(val.z, offset, width);
  return buff;
}

template <>
__forceinline__ __device__ float4 shuffle_down(float4 val, int offset,
                                               int width) {
  float4 buff;
  buff.x = shuffle_down(val.x, offset, width);
  buff.y = shuffle_down(val.y, offset, width);
  buff.z = shuffle_down(val.z, offset, width);
  buff.w = shuffle_down(val.w, offset, width);
  return buff;
}

template <>
__forceinline__ __device__ floatint shuffle_down(floatint val, int offset,
                                                 int width) {
  floatint buff;
  buff.f = shuffle_down(val.f, offset, width);
  buff.i = shuffle_down(val.i, offset, width);
  return buff;
}
}
#endif
