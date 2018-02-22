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

#include <nbla/cuda/common.hpp>
#include <nbla/cuda/solver/rmsprop.hpp>

namespace nbla {

template <typename T>
__global__ void kernel_rmsprop_update(const int num, T *data, const T *grad,
                                      T *e_sqr_grad, const float lr,
                                      const float decay, const float eps) {
  NBLA_CUDA_KERNEL_LOOP(idx, num) {
    e_sqr_grad[idx] =
        e_sqr_grad[idx] * decay + grad[idx] * grad[idx] * (1 - decay);
    data[idx] -= lr * grad[idx] / (sqrt(e_sqr_grad[idx]) + eps);
  }
}

template <typename T>
__global__ void kernel_weight_decay(const int num, T *grad, const T *data,
                                    const float decay_rate) {
  NBLA_CUDA_KERNEL_LOOP(idx, num) { grad[idx] += decay_rate * data[idx]; }
}

template <typename T>
void RMSpropCuda<T>::update_impl(const string &key, VariablePtr param) {
  Size_t size = param->size();
  VariablePtr state = this->state_.at(key);
  T *e_sqr_grad = state->cast_data_and_get_pointer<T>(this->ctx_);
  const T *grad = param->get_grad_pointer<T>(this->ctx_);
  T *data = param->cast_data_and_get_pointer<T>(this->ctx_);
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_rmsprop_update, size, data, grad,
                                 e_sqr_grad, this->lr_, this->decay_,
                                 this->eps_);
}

NBLA_DEF_WEIGHT_DECAY(RMSpropCuda, weight_decay_cuda);

// Template instantiation
template class RMSpropCuda<float>;
}
