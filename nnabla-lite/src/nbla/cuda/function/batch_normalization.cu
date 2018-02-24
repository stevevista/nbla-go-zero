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

#include <nbla/array.hpp>
#include <nbla/variable.hpp>

#include <nbla/cuda/common.hpp>
#include <nbla/cuda/function/batch_normalization.hpp>
#include <nbla/cuda/limits.hpp>

namespace nbla {


/******************************************************************************/
/***                 Forward Global Kernel Implementation                     */
/******************************************************************************/

template <typename T>
__global__ void forward_global_kernel(const int size102_, const int size0_,
                                      const int size1_, const int size2_,
                                      const int size02_, const int size12_,
                                      const T *x, 
                                      const T *gamma, const T *beta, T *y) {
  NBLA_CUDA_KERNEL_LOOP(idx, size102_) {
    const int i1 = idx / size02_;
    const int i0 = (idx / size2_) % size0_;
    const int i2 = idx % size2_;
    const int i = i0 * size12_ + i1 * size2_ + i2;
    y[i] = x[i] * gamma[i1] + beta[i1];
  }
}

template <class T>
void BatchNormalizationCuda<T>::forward_impl(const Variables &inputs,
                                             Variable* output) {
  cuda_set_device(std::stoi(this->ctx_.device_id));
  forward_impl_global(inputs, output);
  
}

template <class T>
void BatchNormalizationCuda<T>::forward_impl_global(const Variables &inputs,
                                                    Variable* output) {
  // Inputs
  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);
  const T *beta = inputs[1]->get_data_pointer<T>(this->ctx_);
  const T *gamma = inputs[2]->get_data_pointer<T>(this->ctx_);
  // Output
  T *y = output->cast_data_and_get_pointer<T>(this->ctx_);

  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
      forward_global_kernel, this->size1_ * this->size02_, this->size0_,
      this->size1_, this->size2_, this->size02_, this->size12_,
      x, gamma, beta, y);
}


template class BatchNormalizationCuda<float>;
}
