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
#include <nbla/function/add2.hpp>
#include <nbla/variable.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(Add2);

template <typename T>
void Add2<T>::setup_impl(const Variables &inputs, Variable* output) {
  NBLA_CHECK(inputs[0]->shape() == inputs[1]->shape(), error_code::value,
                 "Shape of add2 variables mismatch. ");

    output->reshape(inputs[0]->shape(), true);
    output->set_array(inputs[0]->array());
}

template <class T>
void Add2<T>::forward_impl(const Variables &inputs, Variable* output) {
  const T *x0 = inputs[0]->get_data_pointer<T>(this->ctx_);
  const T *x1 = inputs[1]->get_data_pointer<T>(this->ctx_);
  T *y = output->cast_data_and_get_pointer<T>(this->ctx_);
  for (int s = 0; s < inputs[0]->size(); s++) {
    y[s] = x0[s] + x1[s];
  }
}


// Template instanciation
template class Add2<float>;
}
