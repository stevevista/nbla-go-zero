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

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
/* UNDER REVIEW.

   NOTE: cudaMemcpy and kernel execution bat setup_impl.
*/
////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

#include <nbla/array.hpp>
#include <nbla/cuda/common.hpp>
#include <nbla/cuda/function/slice.hpp>
#include <nbla/cuda/math.hpp>
#include <nbla/variable.hpp>

namespace nbla {

__global__ void kernel_slice_create_table(const int num, const int dim,
                                          int *addr_table_buf,
                                          const int *shape_info_buf) {
  NBLA_CUDA_KERNEL_LOOP(idx, num) {
    int addr = 0;
    for (int id = 0; id < dim; id++) {
      const int shape_info_offset = id * 5;
      const int o = (idx / shape_info_buf[shape_info_offset + 1]) // stride_y
                    % shape_info_buf[shape_info_offset];          // shape_y
      const int i = shape_info_buf[shape_info_offset + 3]         // start
                    + o * shape_info_buf[shape_info_offset + 4];  // step
      addr += i * shape_info_buf[shape_info_offset + 2];          // stride_x
    }
    addr_table_buf[idx] = addr;
  }
}

template <typename T>
void SliceCuda<T>::setup_impl(const Variables &inputs,
                              const Variables &outputs) {
  Slice<T>::setup_impl(inputs, outputs);

  // Prepare address table
  const Shape_t shape_y = outputs[0]->shape();
  const Shape_t stride_y = outputs[0]->strides();
  const Shape_t stride_x = inputs[0]->strides();
  size_t size = outputs[0]->size();
  this->addr_table_.reshape(shape_y, true);
  const int shape_info_size = shape_y.size() * 5;
  // out_size, out_stride, in_stride, start, step
  int *shape_info = new int[shape_info_size];
  for (int i = 0; i < shape_y.size(); i++) {
    shape_info[i * 5] = shape_y[i];
    shape_info[i * 5 + 1] = stride_y[i];
    shape_info[i * 5 + 2] = stride_x[i];
    shape_info[i * 5 + 3] = this->start_[0][i];
    shape_info[i * 5 + 4] = this->step_[0][i];
  }
  Shape_t shape_info_shape;
  shape_info_shape.push_back(shape_info_size);
  Variable shape_info_variable;
  shape_info_variable.reshape(shape_info_shape, true);
  int *shape_info_buf =
      shape_info_variable.cast_data_and_get_pointer<int>(this->ctx_);
  cudaMemcpy(shape_info_buf, shape_info, sizeof(int) * shape_info_size,
             cudaMemcpyHostToDevice);
  delete[] shape_info;
  Variable *addr_table_ = &this->addr_table_;
  int *addr_table_buf = addr_table_->cast_data_and_get_pointer<int>(this->ctx_);
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_slice_create_table, size,
                                 shape_y.size(), addr_table_buf,
                                 shape_info_buf);
}

template <typename T>
__global__ void kernel_slice_forward(const int num, T *y, const T *x,
                                     const int *addr_table_buf) {
  NBLA_CUDA_KERNEL_LOOP(idx, num) { y[idx] = x[addr_table_buf[idx]]; }
}

template <typename T>
void SliceCuda<T>::forward_impl(const Variables &inputs,
                                const Variables &outputs) {
  cuda_set_device(std::stoi(this->ctx_.device_id));
  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);
  const int *addr_table_buf =
      this->addr_table_.get_data_pointer<int>(this->ctx_);
  T *y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_);
  size_t size = outputs[0]->size();
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_slice_forward, size, y, x,
                                 addr_table_buf);
}

template <typename T>
__global__ void kernel_slice_backward(const int num, T *dx, const T *dy,
                                      const int *addr_table_buf) {
  NBLA_CUDA_KERNEL_LOOP(idx, num) { dx[addr_table_buf[idx]] += dy[idx]; }
}

template <typename T>
void SliceCuda<T>::backward_impl(const Variables &inputs,
                                 const Variables &outputs,
                                 const vector<bool> &propagate_down,
                                 const vector<bool> &accum) {
  if (!propagate_down[0]) {
    return;
  }
  cuda_set_device(std::stoi(this->ctx_.device_id));
  if (!accum[0])
    inputs[0]->grad()->zero(); // TODO: optimize?
  T *dx = inputs[0]->cast_grad_and_get_pointer<T>(this->ctx_);
  const int *addr_table_buf =
      this->addr_table_.get_data_pointer<int>(this->ctx_);
  const T *dy = outputs[0]->get_grad_pointer<T>(this->ctx_);
  size_t size = outputs[0]->size();
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_slice_backward, size, dx, dy,
                                 addr_table_buf);
}

// template instantiation
template class SliceCuda<float>;
}
