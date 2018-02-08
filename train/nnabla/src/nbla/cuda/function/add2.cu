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

// add2.cu

#include <algorithm>
#include <cmath>
#include <nbla/array.hpp>
#include <nbla/cuda/common.hpp>
#include <nbla/cuda/function/add2.hpp>
#include <nbla/variable.hpp>

namespace nbla {

template <typename T>
__global__ void kernel_add2_forward(const int num, T *y, const T *x0,
                                    const T *x1) {
  NBLA_CUDA_KERNEL_LOOP(idx, num) { y[idx] = x0[idx] + x1[idx]; }
}

template <typename T, bool accum>
__global__ void kernel_add2_backward(const int num, T *d, const T *dy) {
  NBLA_CUDA_KERNEL_LOOP(idx, num) { d[idx] = (accum ? d[idx] : 0) + dy[idx]; }
}

template <class T>
void Add2Cuda<T>::forward_impl(const Variables &inputs,
                               const Variables &outputs) {
  cuda_set_device(std::stoi(this->ctx_.device_id));
  const T *x0 = inputs[0]->get_data_pointer<T>(this->ctx_);
  const T *x1 = inputs[1]->get_data_pointer<T>(this->ctx_);
  T *y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_);
  size_t size = inputs[0]->size();
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_add2_forward, size, y, x0, x1);
}

template <class T>
void Add2Cuda<T>::backward_impl(const Variables &inputs,
                                const Variables &outputs,
                                const vector<bool> &propagate_down,
                                const vector<bool> &accum) {
  if (!(propagate_down[0] || propagate_down[1]))
    return;
  cuda_set_device(std::stoi(this->ctx_.device_id));
  const T *dy = outputs[0]->get_grad_pointer<T>(this->ctx_);
  size_t size = inputs[0]->size();
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      T *dx = inputs[i]->cast_grad_and_get_pointer<T>(this->ctx_);
      if (dx != dy) {
        if (accum[i]) {
          NBLA_CUDA_LAUNCH_KERNEL_SIMPLE((kernel_add2_backward<T, true>), size,
                                         dx, dy);
        } else {
          NBLA_CUDA_LAUNCH_KERNEL_SIMPLE((kernel_add2_backward<T, false>), size,
                                         dx, dy);
        }
      }
    }
  }
}

// Template instantiation
template class Add2Cuda<float>;
}
