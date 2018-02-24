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

#ifndef __NBLA_CUDA_CUDNN_FUNCTION_CONVOLUTION_HPP__
#define __NBLA_CUDA_CUDNN_FUNCTION_CONVOLUTION_HPP__

#include <nbla/cuda/common.hpp>
#include <nbla/cuda/cuda.hpp>
#include <nbla/cuda/cudnn/cudnn.hpp>
#include <nbla/cuda/function/convolution.hpp>
#include <nbla/function/convolution.hpp>

#include <iostream>

namespace nbla {

template <typename T> class ConvolutionCudaCudnn : public Convolution<T> {
public:
  explicit ConvolutionCudaCudnn(const Context &ctx, int base_axis,
                                const vector<int> &pad,
                                const vector<int> &stride,
                                const vector<int> &dilation, int group)
      : Convolution<T>(ctx, base_axis, pad, stride, dilation, group),
        device_(std::stoi(ctx.device_id)) {
  }

  virtual string name() { return "ConvolutionCudaCudnn"; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cuda>()->array_classes();
  }


protected:
  int device_;
  cudnnHandle_t cudnn_handle_;
  int x_offset_;
  int w_offset_;
  int b_offset_;
  int y_offset_;
  shared_ptr<CudnnConv2dResource> rsc2d_;
  virtual void setup_impl(const Variables &inputs, Variable* output);
  void setup_impl_2d(const Variables &inputs, Variable* output);
  void setup_impl_nd(const Variables &inputs, Variable* output);
  virtual void forward_impl(const Variables &inputs, Variable* output);
};
}
#endif
