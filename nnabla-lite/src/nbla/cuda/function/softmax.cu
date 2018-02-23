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

// softmax.cu

#include <algorithm>
#include <nbla/array.hpp>
#include <nbla/cuda/common.hpp>
#include <nbla/cuda/function/softmax.hpp>
#include <nbla/cuda/limits.hpp>
#include <nbla/variable.hpp>

namespace nbla {

template <typename T>
__global__ void kernel_softmax_forward(const int size0x2_, const int size1_,
                                       const int size2_, const T *x, T *y) {
  NBLA_CUDA_KERNEL_LOOP(idx, size0x2_) {
    const int i0 = idx / size2_;
    const int i2 = idx % size2_;
    // compute maximum
    T max_x = nbla::numeric_limits_cuda<T>::min();
    for (int i1 = 0; i1 < size1_; ++i1) {
      const int k = (i0 * size1_ + i1) * size2_ + i2;
      max_x = max(max_x, x[k]);
    }
    // Compute exponential and sum
    T exp_sum = T(0);
    for (int i1 = 0; i1 < size1_; ++i1) {
      const int k = (i0 * size1_ + i1) * size2_ + i2;
      const T tmp = exp(x[k] - max_x);
      y[k] = tmp;
      exp_sum += tmp;
    }
    // Compute softmax
    for (int i1 = 0; i1 < size1_; ++i1) {
      const int k = (i0 * size1_ + i1) * size2_ + i2;
      y[k] = y[k] / exp_sum;
    }
  }
}

template <class T>
void SoftmaxCuda<T>::forward_impl(const Variables &inputs,
                                  const Variables &outputs) {
  cuda_set_device(std::stoi(this->ctx_.device_id));
  // Setting up variables
  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);
  T *y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_);
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_softmax_forward,
                                 this->size0_ * this->size2_, this->size1_,
                                 this->size2_, x, y);
}


template class SoftmaxCuda<float>;
}