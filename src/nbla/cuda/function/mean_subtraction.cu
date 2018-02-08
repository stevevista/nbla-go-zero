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
#include <nbla/cuda/common.hpp>
#include <nbla/cuda/function/mean_subtraction.hpp>
#include <nbla/cuda/math.hpp>
#include <nbla/variable.hpp>

namespace nbla {

template <typename T>
void MeanSubtractionCuda<T>::forward_impl(const Variables &inputs,
                                          const Variables &outputs) {
  cuda_set_device(std::stoi(this->ctx_.device_id));
  if (this->update_runing_mean_) { // Training mode.
    forward_impl_batch(inputs, outputs);
  } else { // Testing mode.
    forward_impl_global(inputs, outputs);
  }
}

template <typename T>
__global__ void kernel_mean_subtraction_inc_t(T *t, const int max) {
  if (t[0] < max) {
    t[0] = t[0] + 1;
  }
}

template <typename T>
__global__ void kernel_mean_subtraction_forward_batch(const int size1_,
                                                      const int size0_,
                                                      const T *x, T *m, T *rm,
                                                      T *y, const int *t) {
  NBLA_CUDA_KERNEL_LOOP(i1, size1_) {
    T coef = 1.0 / ((*t) + 1);

    // Batch mean
    T mean = 0;
    for (int i0 = 0; i0 < size0_; ++i0) {
      mean += x[i1 + i0 * size1_];
    }
    m[i1] = mean / size0_;

    // Moving mean
    rm[i1] = rm[i1] + (m[i1] - rm[i1]) * coef;

    // Output
    for (int i0 = 0; i0 < size0_; ++i0) {
      y[i1 + i0 * size1_] = x[i1 + i0 * size1_] - rm[i1];
    }
  }
}

template <class T>
void MeanSubtractionCuda<T>::forward_impl_batch(const Variables &inputs,
                                                const Variables &outputs) {
  // Inputs
  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);
  // Output
  T *y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_);
  Variable *batch_mean = &this->mean_;
  T *m = batch_mean->cast_data_and_get_pointer<T>(this->ctx_); // batch mean

  // Inputs/Outputs
  T *rm = inputs[1]->cast_data_and_get_pointer<T>(this->ctx_); // running mean
  int *t =
      inputs[2]->cast_data_and_get_pointer<int>(this->ctx_); // running count

  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_mean_subtraction_forward_batch,
                                 this->size1_, this->size0_, x, m, rm, y, t);

  kernel_mean_subtraction_inc_t<<<1, 1>>>(t, std::numeric_limits<int>::max());
}

template <typename T>
__global__ void
kernel_mean_subtraction_forward_global(const int size1_, const int size0_,
                                       const T *x, const T *rm, T *y) {
  NBLA_CUDA_KERNEL_LOOP(i1, size1_) {
    for (int i0 = 0; i0 < size0_; ++i0) {
      y[i1 + i0 * size1_] = x[i1 + i0 * size1_] - rm[i1];
    }
  }
}

template <class T>
void MeanSubtractionCuda<T>::forward_impl_global(const Variables &inputs,
                                                 const Variables &outputs) {
  // Inputs
  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);
  const T *rm = inputs[1]->get_data_pointer<T>(this->ctx_); // running mean

  // Output
  T *y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_);

  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_mean_subtraction_forward_global,
                                 this->size1_, this->size0_, x, rm, y);
}

template <typename T>
void MeanSubtractionCuda<T>::backward_impl(const Variables &inputs,
                                           const Variables &outputs,
                                           const vector<bool> &propagate_down,
                                           const vector<bool> &accum) {
  cuda_set_device(std::stoi(this->ctx_.device_id));
  if (this->update_runing_mean_) { // Training mode.
    backward_impl_batch(inputs, outputs, propagate_down, accum);
  } else { // Testing mode.
    backward_impl_global(inputs, outputs, propagate_down, accum);
  }
}

template <typename T, bool accum>
__global__ void
kernel_mean_subtraction_backward_batch(const int num, T *dx, const T *dy,
                                       const int *t, const int size0_) {
  const T factor = (T)1.0 / ((*t) * size0_);
  NBLA_CUDA_KERNEL_LOOP(idx, num) {
    dx[idx] = (accum ? dx[idx] : 0) + dy[idx] * (1 - factor);
  }
}

template <class T>
void MeanSubtractionCuda<T>::backward_impl_batch(
    const Variables &inputs, const Variables &outputs,
    const vector<bool> &propagate_down, const vector<bool> &accum) {
  if (!propagate_down[0]) {
    return;
  }

  const T *dy = outputs[0]->get_grad_pointer<T>(this->ctx_);
  T *dx = inputs[0]->cast_grad_and_get_pointer<T>(this->ctx_);
  const int *t = inputs[2]->get_data_pointer<int>(this->ctx_);
  size_t size = inputs[0]->size();
  if (accum[0]) {
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
        (kernel_mean_subtraction_backward_batch<T, true>), size, dx, dy, t,
        this->size0_);
  } else {
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
        (kernel_mean_subtraction_backward_batch<T, false>), size, dx, dy, t,
        this->size0_);
  }
}

template <typename T, bool accum>
__global__ void kernel_mean_subtraction_backward_global(const int num, T *dx,
                                                        const T *dy) {
  NBLA_CUDA_KERNEL_LOOP(idx, num) { dx[idx] = (accum ? dx[idx] : 0) + dy[idx]; }
}

template <class T>
void MeanSubtractionCuda<T>::backward_impl_global(
    const Variables &inputs, const Variables &outputs,
    const vector<bool> &propagate_down, const vector<bool> &accum) {
  if (!propagate_down[0]) {
    return;
  }

  const T *dy = outputs[0]->get_grad_pointer<T>(this->ctx_);
  T *dx = inputs[0]->cast_grad_and_get_pointer<T>(this->ctx_);
  size_t size = inputs[0]->size();
  if (accum[0]) {
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
        (kernel_mean_subtraction_backward_global<T, true>), size, dx, dy);
  } else {
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
        (kernel_mean_subtraction_backward_global<T, false>), size, dx, dy);
  }
}

// template instantiation
template class MeanSubtractionCuda<float>;
}
