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

// reduce_sum.cu

#include <algorithm>
#include <nbla/array.hpp>
#include <nbla/cuda/common.hpp>
#include <nbla/cuda/function/reduce_sum.hpp>
#include <nbla/variable.hpp>

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

namespace nbla {

template <typename T>
__global__ void kernel_reduce_sum_backward(const int num, T *dx, const T *dy) {
  NBLA_CUDA_KERNEL_LOOP(idx, num) { dx[idx] += *dy; }
}

template <class T>
void ReduceSumCuda<T>::forward_impl(const Variables &inputs,
                                    const Variables &outputs) {
  cuda_set_device(std::stoi(this->ctx_.device_id));
  T *y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_);
  const Size_t size = inputs[0]->size();
  thrust::device_ptr<const T> x(inputs[0]->get_data_pointer<T>(this->ctx_));
  T sum = thrust::reduce(x, x + size, (T)0, thrust::plus<T>());
  cudaMemcpy(y, &sum, sizeof(T), cudaMemcpyHostToDevice);
}

template <class T>
void ReduceSumCuda<T>::backward_impl(const Variables &inputs,
                                     const Variables &outputs,
                                     const vector<bool> &propagate_down,
                                     const vector<bool> &accum) {
  if (!propagate_down[0]) {
    return;
  }
  cuda_set_device(std::stoi(this->ctx_.device_id));
  const T *dy = outputs[0]->get_grad_pointer<T>(this->ctx_);
  T *dx = inputs[0]->cast_grad_and_get_pointer<T>(this->ctx_);
  const Size_t size = inputs[0]->size();
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_reduce_sum_backward, size, dx, dy);
}

template class ReduceSumCuda<float>;
}
