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

// relu.cpp

#include <nbla/array.hpp>
#include <nbla/function/relu.hpp>
#include <nbla/variable.hpp>

#include <algorithm>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(ReLU);

template <typename T>
void ReLU<T>::setup_impl(const Variables &inputs, Variable* output) {
    output->reshape(inputs[0]->shape(), true);
    output->set_array(inputs[0]->array());
}

template <class T>
void ReLU<T>::forward_impl(const Variables &inputs, Variable* output) {
  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);
  T *y = output->cast_data_and_get_pointer<T>(this->ctx_);
  for (int s = 0; s < inputs[0]->size(); s++) {
    y[s] = std::max(T(0), x[s]);
  }
}


template class ReLU<float>;
}
