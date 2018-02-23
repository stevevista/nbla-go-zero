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

// convolution.cu

#include <nbla/array.hpp>
#include <nbla/cuda/common.hpp>
#include <nbla/cuda/function/convolution.hpp>
#include <nbla/cuda/math.hpp>
#include <nbla/singleton_manager.hpp>
#include <nbla/variable.hpp>

#include <nbla/cuda/utils/col2im.hpp>
#include <nbla/cuda/utils/im2col.hpp>

#include <algorithm>

namespace nbla {

template <typename T>
void ConvolutionCuda<T>::setup_impl(const Variables &inputs,
                                    const Variables &outputs) {
  Convolution<T>::setup_impl(inputs, outputs);
}

template <class T>
void ConvolutionCuda<T>::forward_impl(const Variables &inputs,
                                      const Variables &outputs) {
  cuda_set_device(std::stoi(this->ctx_.device_id));
  // Getting variable pointers
  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);
  const T *w = inputs[1]->get_data_pointer<T>(this->ctx_);
  Variable *vcol = &this->col_;
  T *col = vcol->cast_data_and_get_pointer<T>(this->ctx_);
  float *y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_);
  const T *b;
  if (inputs.size() == 3) {
    b = inputs[2]->get_data_pointer<T>(this->ctx_);
  }
  // Sample loop
  for (int n = 0; n < this->outer_size_; ++n) {
    // Im2col
    if (this->spatial_dims_ == 2) {
      im2col_cuda<T>(x + n * this->inner_size_i_, this->channels_i_,
                     this->spatial_shape_i_.data(), this->kernel_.data(),
                     this->pad_.data(), this->stride_.data(),
                     this->dilation_.data(), col);
    } else {
      im2col_nd_cuda<T>(x + n * this->inner_size_i_, this->channels_i_,
                        this->spatial_dims_, this->spatial_shape_i_.data(),
                        this->kernel_.data(), this->pad_.data(),
                        this->stride_.data(), this->dilation_.data(), col);
    }
    // Convolution by matrix multiplication
    T *y_n = y + n * this->inner_size_o_;
    for (int g = 0; g < this->group_; ++g) {
      // y = x * w
      cuda_gemm<T>(device_, y_n + g * this->row_y_ * this->col_y_, false,
                   col + g * this->row_col_ * this->col_col_, this->col_col_,
                   this->row_col_, false, w + g * this->row_w_ * this->col_w_,
                   this->col_w_, this->row_w_, false, (T)1, (T)0);
    }
    // Adding bias
    if (inputs.size() == 3) {
      const T *ones =
          static_cast<const T *>(SingletonManager::get<NNabla>()->ones(
              this->col_y_, get_dtype<T>(), this->ctx_));
      // y = 1s * b^T + y
      cuda_gemm<T>(device_, y_n, false, ones, 1, this->col_y_, true, b,
                   this->channels_o_, 1, true, (T)1, (T)1);
    }
  }
}

template class ConvolutionCuda<float>;
}
