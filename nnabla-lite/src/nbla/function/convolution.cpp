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

// convolution.cpp

#include <nbla/array.hpp>
#include <nbla/function/convolution.hpp>
#include <nbla/utils/eigen.hpp>
#include <nbla/utils/im2col.hpp>
#include <nbla/variable.hpp>

#include <algorithm>
#include <cstring>
#include <memory>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(Convolution, int,    // base_axis
                              const vector<int> &, // pad
                              const vector<int> &, // stride
                              const vector<int> &, // dilation
                              int);                // group

template <typename T>
void Convolution<T>::setup_impl(const Variables &inputs,
                                Variable* output) {
  // Shape check
  Shape_t shape_data = inputs[0]->shape();
  Shape_t shape_weights = inputs[1]->shape();
  NBLA_CHECK(base_axis_ < shape_data.size() - 1, error_code::unclassified,
             "base_axis must be less than ndim - 1 of inputs[0]. "
             "base_axis: %d >= ndim of inputs[0] - 1: %d.",
             base_axis_, shape_data.size() - 1);
  spatial_dims_ = shape_data.size() - base_axis_ - 1;
  NBLA_CHECK(shape_weights.size() == 2 + spatial_dims_, error_code::value,
             "Weights must be a tensor more than 3D.");
  // Storing shape variables
  channels_i_ = shape_data[base_axis_];
  channels_o_ = shape_weights[0];
  channels_g_ = shape_weights[1];
  inner_size_k_ = channels_g_;
  const int channels_i_mod_group = channels_i_ % group_;
  NBLA_CHECK(channels_i_mod_group == 0, error_code::value,
             "Number of input channel needs to be divisible by group. "
             "Input channel: %d, group: %d.",
             channels_i_, group_);
  const int channels_o_mod_group = channels_o_ % group_;
  NBLA_CHECK(channels_o_mod_group == 0, error_code::value,
             "Number of output channel needs to be divisible by group. "
             "Output channel: %d, group: %d.",
             channels_o_, group_);
  NBLA_CHECK(channels_i_ / group_ == channels_g_, error_code::value,
             "Number of grouped channel mismatch."
             "Input: %d != Weights[1]: %d.",
             channels_i_ / group_, channels_g_);
  NBLA_CHECK(pad_.size() == spatial_dims_, error_code::value,
             "pad size mismatch. pad size: %d != spatial dims: %d.",
             pad_.size(), spatial_dims_);
  NBLA_CHECK(stride_.size() == spatial_dims_, error_code::value,
             "stride size mismatch. stride size: %d != spatial dims: %d.",
             stride_.size(), spatial_dims_);
  NBLA_CHECK(dilation_.size() == spatial_dims_, error_code::value,
             "dilation size mismatch. dilation size: %d != spatial dims: %d.",
             dilation_.size(), spatial_dims_);
  for (int i = 0; i < spatial_dims_; ++i) {
    kernel_.push_back(shape_weights[2 + i]);
    inner_size_k_ *= kernel_[i];
    spatial_shape_i_.push_back(shape_data[base_axis_ + 1 + i]);
    const int k = dilation_[i] * (kernel_[i] - 1) + 1;
    const int o = (spatial_shape_i_[i] + 2 * pad_[i] - k) / stride_[i] + 1;
    NBLA_CHECK(
        o > 0, error_code::value,
        "Invalid configuration of convolution at %d-th spatial dimension.  "
        "{input:%d, kernel:%d, pad:%d, stride:%d, dilation:%d}.",
        i, spatial_shape_i_[i], kernel_[i], pad_[i], stride_[i], dilation_[i]);
    spatial_shape_o_.push_back(o);
  }

  // Reshaping output
  Shape_t shape_out;
  outer_size_ = 1;
  for (int i = 0; i < base_axis_; ++i) { // Fill shapes up to base axis
    shape_out.push_back(shape_data[i]);
    outer_size_ *= shape_data[i];
  }
  shape_out.push_back(channels_o_); // output channels
  inner_size_i_ = channels_i_;
  inner_size_o_ = channels_o_;
  for (int i = 0; i < spatial_dims_; ++i) {
    shape_out.push_back(spatial_shape_o_[i]);
    inner_size_i_ *= spatial_shape_i_[i];
    inner_size_o_ *= spatial_shape_o_[i];
  }
  output->reshape(shape_out, true);

  // Reshaping col buffer
  // Actual memory is not allocated until it is used.
  col_.reshape(Shape_t{inner_size_k_ * group_, inner_size_o_ / channels_o_},
               true);

  // Check for with bias
  if (inputs.size() == 3) {
    NBLA_CHECK(inputs[2]->shape().size() == 1, error_code::value,
               "Bias(inputs[2]) must be a 1d tensor.");
    NBLA_CHECK(inputs[2]->shape()[0] == channels_o_, error_code::value,
               "Shape of bias(inputs[2]) and weights(inputs[1]) mismatch. "
               "bias shape[0]: %d != weights shape[0]: %d.",
               inputs[2]->shape()[0], channels_o_);
  }

  // Set variables for convolution by matrix multiplication
  // In 2D case:
  // K: in maps, H: in height, W: in width
  // K': out maps, H': out height, W': out height
  // M: kernel height, N: kernel width
  row_w_ = channels_o_ / group_;          // K'
  col_w_ = inner_size_k_;                 // KMN
  row_col_ = col_w_;                      // KMN
  col_col_ = inner_size_o_ / channels_o_; // H'W'
  row_y_ = channels_o_ / group_;          // K'
  col_y_ = col_col_;                      // H'W'
}

template <class T>
void Convolution<T>::forward_impl(const Variables &inputs,
                                  Variable* output) {
  using namespace ::nbla::eigen;
  // Getting variable pointers
  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);
  const T *w = inputs[1]->get_data_pointer<T>(this->ctx_);
  T *col = col_.cast_data_and_get_pointer<T>(this->ctx_);
  T *y = output->cast_data_and_get_pointer<T>(this->ctx_);
  const T *b;
  if (inputs.size() == 3) {
    b = inputs[2]->get_data_pointer<T>(this->ctx_);
  }
  // Sample loop
  for (int n = 0; n < outer_size_; ++n) {
    // Im2col
    if (spatial_dims_ == 2) {
      im2col<T>(x + n * inner_size_i_, channels_i_, spatial_shape_i_.data(),
                kernel_.data(), pad_.data(), stride_.data(), dilation_.data(),
                col);
    } else {
      im2col_nd<T>(x + n * inner_size_i_, channels_i_, spatial_dims_,
                   spatial_shape_i_.data(), kernel_.data(), pad_.data(),
                   stride_.data(), dilation_.data(), col);
    }
    // Convolution by matrix multiplication
    T *y_n = y + n * inner_size_o_;
    for (int g = 0; g < group_; ++g) {
      MatrixMap<T> mcol(col + g * row_col_ * col_col_, row_col_, col_col_);
      ConstMatrixMap<T> mk(w + g * row_w_ * col_w_, row_w_, col_w_);
      MatrixMap<T> my(y_n + g * row_y_ * col_y_, row_y_, col_y_);
      my = mk * mcol;
    }
    // Adding bias
    if (inputs.size() == 3) {
      MatrixMap<T> my(y_n, channels_o_, col_y_);
      my.colwise() += ConstColVectorMap<T>(b, channels_o_);
    }
  }
}

template class Convolution<float>;
}
