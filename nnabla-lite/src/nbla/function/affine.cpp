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

// affine.cpp

#include <nbla/array.hpp>
#include <nbla/function/affine.hpp>
#include <nbla/utils/eigen.hpp>
#include <nbla/variable.hpp>

#include <algorithm>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(Affine, int);

template <typename T>
void Affine<T>::setup_impl(const Variables &inputs, const Variables &outputs) {
  Shape_t shape_data = inputs[0]->shape();
  Shape_t shape_weights = inputs[1]->shape();
  NBLA_CHECK(shape_weights.size() >= 2, error_code::value,
             "Weights(inputs[1]) must be matrix or tensor.");
  NBLA_CHECK(base_axis_ < shape_data.size(), error_code::value,
             "Base_axis must be less than ndim of input data(inputs[0]). "
             "base_axis: %d >= ndim of input: %d.",
             base_axis_, shape_data.size());
  NBLA_CHECK(inputs[0]->size(base_axis_) == shape_weights[0], error_code::value,
             "Size of input data(inputs[0]) and weights(inputs[1]) mismatch. "
             "size of input: %d != size of weights: %d.",
             inputs[0]->size(base_axis_), shape_weights[0]);
  i_col_ = inputs[0]->size(base_axis_);
  i_row_ = inputs[0]->size() / i_col_;
  w_row_ = shape_weights[0];
  w_col_ = inputs[1]->size() / w_row_;
  o_row_ = i_row_;
  o_col_ = w_col_;
  Shape_t shape_out;
  for (int i = 0; i < base_axis_; ++i) {
    shape_out.push_back(shape_data[i]);
  }
  for (int i = 1; i < shape_weights.size(); ++i) {
    shape_out.push_back(shape_weights[i]);
  }
  outputs[0]->reshape(shape_out, true);
  NBLA_CHECK(i_col_ == w_row_, error_code::value, "Shape mismatch.");
  NBLA_CHECK(outputs[0]->size() == o_row_ * o_col_, error_code::value,
             "Shape mismatch. %ld %ld %ld", outputs[0]->size(), o_row_, o_col_);
  if (inputs.size() == 3) {
    // With bias
    Shape_t shape_bias = inputs[2]->shape();
    NBLA_CHECK(shape_bias.size() == shape_weights.size() - 1, error_code::value,
               "Length of bias(inputs[2]) and weights(inputs[1]) mismatch. "
               "bias length: %d != weights length-1: %d.",
               shape_bias.size(), shape_weights.size() - 1);
    for (int i = 0; i < shape_bias.size(); ++i) {
      NBLA_CHECK(shape_bias[i] == shape_weights[i + 1], error_code::value,
                 "Shape of bias(inputs[2]) and weights(inputs[1]) mismatch. "
                 "shape_bias[%d]: %d != shape_weights[%d + 1]: %d.",
                 i, shape_bias[i], i, shape_weights[i + 1]);
    }
  }
}

template <class T>
void Affine<T>::forward_impl(const Variables &inputs,
                             const Variables &outputs) {
  using namespace ::nbla::eigen;
  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);
  const T *w = inputs[1]->get_data_pointer<T>(this->ctx_);
  T *y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_);
  ConstMatrixMap<T> mx(x, i_row_, i_col_);
  ConstMatrixMap<T> mw(w, w_row_, w_col_);
  MatrixMap<T> my(y, o_row_, o_col_);
  my = mx * mw;
  if (inputs.size() == 3) {
    // With bias
    const T *b = inputs[2]->get_data_pointer<T>(this->ctx_);
    my.rowwise() += ConstRowVectorMap<T>(b, o_col_);
  }
}


template class Affine<float>;
}
