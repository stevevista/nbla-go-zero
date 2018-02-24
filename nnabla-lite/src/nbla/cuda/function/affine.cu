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
#include <nbla/common.hpp>
#include <nbla/cuda/common.hpp>
#include <nbla/cuda/function/affine.hpp>
#include <nbla/cuda/math.hpp>
#include <nbla/singleton_manager.hpp>
#include <nbla/variable.hpp>

namespace nbla {

template <class T>
void AffineCuda<T>::forward_impl(const Variables &inputs,
                                 Variable* output) {
  cuda_set_device(std::stoi(this->ctx_.device_id));
  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);
  const T *w = inputs[1]->get_data_pointer<T>(this->ctx_);
  T *y = output->cast_data_and_get_pointer<T>(this->ctx_);
  // y = x * w.
  cuda_gemm<T>(device_, y, true, x, this->i_col_, this->i_row_, true, w,
               this->w_col_, this->w_row_, true, (T)1,
               (T)0); // Note that arrays are row-major.
  if (inputs.size() == 3) {
    // With bias
    const T *b = inputs[2]->get_data_pointer<T>(this->ctx_);
    const T *ones =
        static_cast<const T *>(SingletonManager::get<NNabla>()->ones(
            this->o_row_, get_dtype<T>(), this->ctx_));
    // y = 1s * b^T + y
    cuda_gemm<T>(device_, y, true, ones, this->o_row_, 1, false, b, 1,
                 this->o_col_, false, (T)1, (T)1);
  }
}


// template instantiation
template class AffineCuda<float>;
}
