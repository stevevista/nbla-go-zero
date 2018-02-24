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

// mul_scalar.cu

#include <nbla/cuda/function/mul_scalar.hpp>
#include <nbla/cuda/common.hpp>
#include <nbla/cuda/cuda.hpp>


namespace nbla {


template <typename T>
__global__ void kernel_mul_scalar_forward(const int num, const T *x1, const T *x1, T *y) {
  NBLA_CUDA_KERNEL_LOOP(idx, num) { y[idx] = x0[idx] * x1[0]; }
}


template <typename T>
void MulScalarCuda<T>::forward_impl(const Variables &inputs,
                                   Variable* output) {   

    const T *x0 = inputs[0]->get_data_pointer<T>(this->ctx_);
    const T *x1 = inputs[1]->get_data_pointer<T>(this->ctx_);
    T *y = output->cast_data_and_get_pointer<T>(this->ctx_);
    int size = output->size();
    cuda_set_device(std::stoi(this->ctx_.device_id));
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_mul_scalar_forward, size, x0, x1, y);
}


// Template instantiation
template class MulScalarCuda<float>;

}
