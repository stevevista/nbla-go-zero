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
#include <nbla/function/batch_normalization.hpp>
#include <nbla/variable.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(BatchNormalization, int);

template <typename T>
void BatchNormalization<T>::setup_impl(const Variables &inputs,
                                       Variable* output) {

  // Check and parse shapes
  Shape_t shape_i = inputs[0]->shape();
  Size_t size = inputs[0]->size();
  Size_t size_axis = inputs[0]->size(axis_);
  size0_ = size / size_axis;       // Batch size.
  size1_ = shape_i[axis_];      // Size of specified axis.
  size2_ = size / size0_ / size1_; // Size of rest.
  size12_ = size1_ * size2_;
  size02_ = size0_ * size2_;
  NBLA_CHECK(size0_ * size1_ * size2_ == size, error_code::unclassified,
             "An error occurred during setup BatchNormalization function.");
  // Verify mean, var, beta and gamma dims.
  Shape_t shape_b = inputs[1]->shape();
  Shape_t shape_g = inputs[2]->shape();
  // Verify mean, var, beta and gamma shapes.
  Shape_t shape_check(shape_i.size(), 1);
  shape_check[axis_] = shape_i[axis_];
  NBLA_CHECK(shape_check == shape_b, error_code::value,
             "Shape of beta(inputs[1]) does not match. "
             "beta: (%s) != expected: (%s).",
             string_join(shape_b, string(", ")).c_str(),
             string_join(shape_check, string(", ")).c_str());
  NBLA_CHECK(shape_check == shape_g, error_code::value,
             "Shape of gamma(inputs[2]) does not match. "
             "gamma: (%s) != expected: (%s).",
             string_join(shape_g, string(", ")).c_str(),
             string_join(shape_check, string(", ")).c_str());

  // Reshape outputs and/or temporary buffers.
  output->reshape(shape_i, true);
}

template <class T>
void BatchNormalization<T>::forward_impl(const Variables &inputs,
                                         Variable* output) {
  forward_impl_global(inputs, output);
}

template <class T>
void BatchNormalization<T>::forward_impl_global(const Variables &inputs,
                                                Variable* output) {
  // Inputs
  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);
  const T *beta = inputs[1]->get_data_pointer<T>(this->ctx_);
  const T *gamma = inputs[2]->get_data_pointer<T>(this->ctx_);
  // Output
  T *y = output->cast_data_and_get_pointer<T>(this->ctx_);

  // Subtract mean and divide by std, and apply beta and gamma.
  for (int i1 = 0; i1 < size1_; ++i1) {
    for (int i02 = 0; i02 < size02_; ++i02) {
      const int i0 = i02 / size2_;
      const int i2 = i02 % size2_;
      const int i = i0 * size12_ + i1 * size2_ + i2;
      y[i] = x[i] * gamma[i1] + beta[i1];
    }
  }
}


template class BatchNormalization<float>;
}
