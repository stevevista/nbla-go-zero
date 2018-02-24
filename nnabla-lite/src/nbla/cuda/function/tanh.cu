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

// tanh.cpp

#include <nbla/cuda/function/tanh.hpp>
#include <nbla/cuda/common.hpp>
#include <nbla/cuda/cuda.hpp>

#include <cmath>

namespace nbla {

template <typename T>
__global__ void kernel_tanh_forward(int size, const T *x, T *y) {
  NBLA_CUDA_KERNEL_LOOP(idx, size) { y[idx] = tanh(x[idx]); }
}

template <typename T>                                                   
void TanhCuda<T>::forward_impl(const Variables &inputs,                   
                                   Variable* output) {                

    cuda_set_device(std::stoi(this->ctx_.device_id));
    const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);
    T *y = output->cast_data_and_get_pointer<T>(this->ctx_);
    int size = inputs[0]->size();
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_tanh_forward, size, x, y);
               
}                                                                           


// Template instantiation
template class TanhCuda<float>;
}
