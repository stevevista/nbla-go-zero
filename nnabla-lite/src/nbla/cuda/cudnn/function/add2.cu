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

/** Add2
 */

#include <algorithm>
#include <nbla/array.hpp>
#include <nbla/cuda/common.hpp>
#include <nbla/cuda/cudnn/cudnn.hpp>
#include <nbla/cuda/cudnn/function/add2.hpp>
#include <nbla/variable.hpp>

namespace nbla {

template <typename T>
void Add2CudaCudnn<T>::setup_impl(const Variables &inputs,
                                  Variable* output) {

  Add2<T>::setup_impl(inputs, output);
  cudnn_handle_ = SingletonManager::get<CudnnHandleManager>()->handle(device_);
  NBLA_CUDNN_CHECK(cudnnSetTensor4dDescriptor(input_desc_, CUDNN_TENSOR_NCHW,
                                              cudnn_data_type<T>::type(), 1, 1,
                                              1, inputs[0]->size()));
  NBLA_CUDNN_CHECK(cudnnSetTensor4dDescriptor(output_desc_, CUDNN_TENSOR_NCHW,
                                              cudnn_data_type<T>::type(), 1, 1,
                                              1, output->size()));
}

template <typename T>
void Add2CudaCudnn<T>::forward_impl(const Variables &inputs,
                                    Variable* output) {
  cuda_set_device(std::stoi(this->ctx_.device_id));
  const T *x0 = inputs[0]->get_data_pointer<T>(this->ctx_);
  const T *x1 = inputs[1]->get_data_pointer<T>(this->ctx_);
  T *y = output->cast_data_and_get_pointer<T>(this->ctx_);
  T alpha = 1;
  T beta = 1;

#if CUDNN_VERSION >= 4000
    NBLA_CUDNN_CHECK(cudnnAddTensor(cudnn_handle_, &alpha, input_desc_, x1,
                                    &beta, output_desc_, y));
#else
    NBLA_CUDNN_CHECK(cudnnAddTensor(cudnn_handle_, CUDNN_ADD_FULL_TENSOR,
                                    &alpha, input_desc_, x1, &beta,
                                    output_desc_, y));
#endif
  } else if (x1 == y) {
#if CUDNN_VERSION >= 4000
    NBLA_CUDNN_CHECK(cudnnAddTensor(cudnn_handle_, &alpha, input_desc_, x0,
                                    &beta, output_desc_, y));
#else
    NBLA_CUDNN_CHECK(cudnnAddTensor(cudnn_handle_, CUDNN_ADD_FULL_TENSOR,
                                    &alpha, input_desc_, x0, &beta,
                                    output_desc_, y));
#endif

}

template class Add2CudaCudnn<float>;
}
