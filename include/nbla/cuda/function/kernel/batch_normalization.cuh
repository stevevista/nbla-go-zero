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

#include <nbla/cuda/common.hpp>
#include <nbla/cuda/function/kernel/transpose.cuh>
#include <nbla/cuda/utils/block_reduce.cuh>

namespace nbla {

#define NBLA_CUDA_1D_GRID_STRIDE_LOOP(idx, num)                                \
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; i < num;               \
       idx += blockDim.x * gridDim.x)

// prototype
//#define TEST_FEATURE_MEAN_VARIANCE_KERNEL
#ifdef TEST_FEATURE_MEAN_VARIANCE_KERNEL
template <typename T>
__global__ void mean_variance_kernel(const T *in, T *tmp_m, T *tmp_v, T *m,
                                     T *v, int N, int blockNums) {
  float2 mean_variance;
  mean_variance.x = 0;
  mean_variance.y = 0;
  NBLA_CUDA_1D_GRID_STRIDE_LOOP(i, N) {
    const T value = in[i];
    mean_variance.x += value;
    mean_variance.y += value * value;
  }
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  mean_variance = blockReduceSumOfFloat2(mean_variance);
  if (threadIdx.x == 0) {
    tmp_m[blockIdx.x] = mean_variance.x;
    tmp_v[blockIdx.x] = mean_variance.y;
  }
  __syncthreads();
  mean_variance.x = 0;
  mean_variance.y = 0;
  if (i < blockNums) {
    mean_variance.x = tmp_m[i];
    mean_variance.y = tmp_v[i];
  }
  mean_variance = blockReduceSumOfFloat2(mean_variance);
  if (threadIdx.x == 0) {
    m[blockIdx.x] = mean_variance.x;
    v[blockIdx.x] = mean_variance.y;
  }
}
#endif

// prototype
//#define TEST_FEATURE_MEAN_VARIANCE_AXIS_REDUCTION_KERNEL
#ifdef TEST_FEATURE_MEAN_VARIANCE_AXIS_REDUCTION_KERNEL
template <typename T>
__global__ void mean_variance_with_axis_kernel(const T *in_trans, T *tmp_m,
                                               T *tmp_v, T *m, T *v, int N,
                                               int blockNums, int axis_size) {
  float2 mean_variance;
  for (int idx = 0; idx < axis_size; ++idx) {
    mean_variance.x = 0;
    mean_variance.y = 0;
    NBLA_CUDA_1D_GRID_STRIDE_LOOP(i, N) {
      const T value = in_trans[i + idx * N];
      mean_variance.x += value;
      mean_variance.y += value * value;
    }
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    mean_variance = blockReduceSumOfFloat2(mean_variance);
    if (threadIdx.x == 0) {
      tmp_m[blockIdx.x] = mean_variance.x;
      tmp_v[blockIdx.x] = mean_variance.y;
    }
    __syncthreads();
    mean_variance.x = 0;
    mean_variance.y = 0;
    if (i < blockNums) {
      mean_variance.x = tmp_m[i];
      mean_variance.y = tmp_v[i];
    }
    mean_variance = blockReduceSumOfFloat2(mean_variance);
    if (threadIdx.x == 0) {
      m[idx] = mean_variance.x;
      v[idx] = mean_variance.y;
    }
    __syncthreads();
  }
}
#endif

/******************************************************************************/
/***                 Forward Batch Kernel Implementation                      */
/******************************************************************************/

template <typename T>
__global__ void forward_batch_mean_variance_kernel(
    const int size1, const int size2, const int size02, const int size12,
    const float decay_rate, const float eps, const T *x, const T *gamma,
    const T *beta, T *m, T *v, T *rm, T *rv) {
  NBLA_CUDA_KERNEL_LOOP(i1, size1) {
    T tmp_m = 0;
    T tmp_v = 0;
    for (int i02 = 0; i02 < size02; ++i02) {
      const int i0 = i02 / size2;
      const int i2 = i02 % size2;
      const int i = i0 * size12 + i1 * size2 + i2;
      const T value = x[i];
      tmp_m += value;
      tmp_v += value * value;
    }
    tmp_m /= size02;
    m[i1] = tmp_m;
    tmp_v = tmp_v / size02 - tmp_m * tmp_m;
    v[i1] = tmp_v;

    rm[i1] = decay_rate * rm[i1] + (1. - decay_rate) * tmp_m;
    rv[i1] =
        decay_rate * rv[i1] + (1. - decay_rate) * tmp_v * size02 / (size02 - 1);
  }
}

template <typename T>
__global__ void forward_batch_gamma_beta_kernel(
    const int size102, const int size0, const int size2, const int size02,
    int const size12, const float decay_rate, const float eps, const T *x,
    const T *m, const T *v, const T *rm, const T *rv, const T *gamma,
    const T *beta, T *y) {
  NBLA_CUDA_KERNEL_LOOP(idx, size102) {
    const int i1 = idx / size02;
    const int i0 = (idx / size2) % size0;
    const int i2 = idx % size2;
    const int i = i0 * size12 + i1 * size2 + i2;
    const T stdvar = sqrt(v[i1] + eps);
    y[i] = (x[i] - m[i1]) * gamma[i1] / stdvar + beta[i1];
  }
}

template <typename T>
__global__ void forward_batch_kernel_mean_variance_preprocess(const T *x,
                                                              const int N, T *m,
                                                              T *v) {
  float2 mean_variance;
  mean_variance.x = 0;
  mean_variance.y = 0;
  NBLA_CUDA_1D_GRID_STRIDE_LOOP(i, N) {
    const T value = x[i];
    mean_variance.x += value;
    mean_variance.y += value * value;
  }
  mean_variance = blockReduceSumOfFloat2(mean_variance);
  if (threadIdx.x == 0) {
    m[blockIdx.x] = mean_variance.x;
    v[blockIdx.x] = mean_variance.y;
  }
}

__global__ void forward_batch_kernel_mean_variance_postprocess(
    const float *block_m, const float *block_v, const int block_nums,
    const float decay_rate, const float inv_N, const float svar, float *m,
    float *v, float *rm, float *rv) {
  float2 mean_variance;
  mean_variance.x = 0;
  mean_variance.y = 0;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < block_nums) {
    mean_variance.x = block_m[i];
    mean_variance.y = block_v[i];
  }
  mean_variance = blockReduceSumOfFloat2(mean_variance);
  if (threadIdx.x == 0) {
    const float mean = mean_variance.x * inv_N;
    const float variance = mean_variance.y * inv_N - mean * mean;
    m[blockIdx.x] = mean;
    v[blockIdx.x] = variance;
    rm[blockIdx.x] = decay_rate * rm[blockIdx.x] + (1. - decay_rate) * mean;
    rv[blockIdx.x] =
        decay_rate * rv[blockIdx.x] + (1. - decay_rate) * variance * svar;
  }
}

__global__ void forward_batch_kernel_gamma_beta_trans(
    const int shape_size, const int N, const float *x, const float *gamma,
    const float *beta, float *m, float *v, float decay_rate, float eps,
    const int ndim, const int *axes, const int *x_strides, const int *y_strides,
    const int *y_shape, float *y, float *inv_sqrt_variance) {
  NBLA_CUDA_KERNEL_LOOP(o, shape_size) {
    int i = 0;
    for (int d = 0; d < ndim; ++d) {
      const int k = int(o / y_strides[d]) % y_shape[d];
      i += k * x_strides[axes[d]];
    }
    const int axis_idx = int(i / N);
    const float inv_stdvar = 1. / sqrt(v[axis_idx] + eps);
    inv_sqrt_variance[axis_idx] = inv_stdvar;
    y[o] = (x[i] - m[axis_idx]) * gamma[axis_idx] * inv_stdvar + beta[axis_idx];
  }
}

/******************************************************************************/
/***                 Forward Global Kernel Implementation                     */
/******************************************************************************/

template <typename T>
__global__ void forward_global_kernel(const int size102_, const int size0_,
                                      const int size1_, const int size2_,
                                      const int size02_, const int size12_,
                                      const float decay_rate_, const float eps_,
                                      const T *x, const T *rm, const T *rv,
                                      const T *gamma, const T *beta, T *y) {
  NBLA_CUDA_KERNEL_LOOP(idx, size102_) {
    const int i1 = idx / size02_;
    const int i0 = (idx / size2_) % size0_;
    const int i2 = idx % size2_;
    const int i = i0 * size12_ + i1 * size2_ + i2;
    const T mean = rm[i1];
    const T stdvar = sqrt(rv[i1] + eps_);
    y[i] = (x[i] - mean) * gamma[i1] / stdvar + beta[i1];
  }
}

/******************************************************************************/
/***             Backward Batch Data Kernel Implementation                    */
/******************************************************************************/

template <typename T>
__global__ void backward_batch_data_mean_variance_kernel(
    const int size1, const int size2, const int size02, const int size12,
    const float decay_rate, const float eps, const T *dy, const T *m,
    const T *v, const T *x, const T *g, const T *dm, const T *dv, T *dmean,
    T *dvar) {
  NBLA_CUDA_KERNEL_LOOP(i1, size1) {
    T tmp_dvar = 0;
    T tmp_dmean = 0;
    T tmp = 0;
    for (int i02 = 0; i02 < size02; ++i02) {
      const int i0 = i02 / size2;
      const int i2 = i02 % size2;
      const int i = i0 * size12 + i1 * size2 + i2;
      const T dxh = dy[i] * g[i1]; // Grad of x hat.
      const T cx = x[i] - m[i1];   // x - mean
      tmp_dvar += dxh * cx;
      tmp_dmean += dxh;
      tmp += cx;
    }
    T tmp_v = v[i1];
    dvar[i1] = tmp_dvar * -0.5 * pow(tmp_v + eps, (T)-1.5) + (dv ? dv[i1] : 0);
    dmean[i1] = tmp_dmean * (-1. / sqrt(tmp_v + eps)) +
                dvar[i1] * (-2) * tmp / (size02) + (dm ? dm[i1] : 0);
  }
}

template <typename T>
__global__ void backward_batch_data_dx_kernel(
    const int size102, const int size0, const int size1, const int size2,
    const int size02, const int size12, const float decay_rate, const float eps,
    const T *dy, const T *m, const T *v, const T *x, const T *g, const T *dm,
    const T *dv, const T *dmean, const T *dvar, T *dx) {
  NBLA_CUDA_KERNEL_LOOP(idx, size102) {
    const int i1 = idx / size02;
    const int i0 = (idx / size2) % size0;
    const int i2 = idx % size2;
    const int i = i0 * size12 + i1 * size2 + i2;
    dx[i] += dy[i] * g[i1] / sqrt(v[i1] + eps) +
             dvar[i1] * 2 * (x[i] - m[i1]) / (size02) + dmean[i1] / (size02);
  }
}

__global__ void backward_batch_data_kernel_mean_variance_preprocess(
    const int N, const float *dy, const float *x, const float *g,
    const float *m, float *block_m, float *block_v, float *block_t) {
  float3 mean_variance;
  mean_variance.x = 0; // dmean
  mean_variance.y = 0; // dvar
  mean_variance.z = 0; // tmp
  NBLA_CUDA_1D_GRID_STRIDE_LOOP(i, N) {
    const float dxh = dy[i] * g[0];
    const float cx = x[i] - m[0];
    mean_variance.y += dxh * cx;
    mean_variance.x += dxh;
    mean_variance.z += cx;
  }
  mean_variance = blockReduceSumOfFloat3(mean_variance);
  if (threadIdx.x == 0) {
    block_m[blockIdx.x] = mean_variance.x;
    block_v[blockIdx.x] = mean_variance.y;
    block_t[blockIdx.x] = mean_variance.z;
  }
}

__global__ void backward_batch_data_kernel_mean_variance_postprocess(
    const float *block_m, const float *block_v, const float *block_t,
    const int block_nums, const float inv_N, const float *v, const float *dm,
    const float *dv, const float eps, const int N,
    const float *inv_sqrt_variance, const int axis_idx, float *dmean,
    float *dvar, float *t) {
  float3 mean_variance;
  mean_variance.x = 0; // dmean
  mean_variance.y = 0; // dvar
  mean_variance.z = 0; // tmp
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < block_nums) {
    mean_variance.x += block_m[i];
    mean_variance.y += block_v[i];
    mean_variance.z += block_t[i];
  }
  mean_variance = blockReduceSumOfFloat3(mean_variance);
  if (threadIdx.x == 0) {
    const float tmp_dvar =
        mean_variance.y * (float)-0.5 * pow(v[0] + eps, (float)-1.5) +
        (dv ? dv[axis_idx] : 0);
    dvar[0] = tmp_dvar;
    dmean[0] = mean_variance.x * (-inv_sqrt_variance[0]) +
               tmp_dvar * (-2) * mean_variance.z * inv_N +
               (dm ? dm[axis_idx] : 0);
  }
}

__global__ void backward_batch_data_kernel_gamma_beta_trans(
    const int shape_size, const float inv_N, const float *dy, const float *x,
    const float *g, const float *v, const float *m, const float *dmean,
    const float *dvar, const int ndim, const int *axes, const int *y_strides,
    const int *x_strides, const int *x_shape, const float *inv_sqrt_variance,
    float *dx) {
  NBLA_CUDA_KERNEL_LOOP(o, shape_size) {
    int i = 0;
    for (int d = 0; d < ndim; ++d) {
      const int k = int(o / x_strides[d]) % x_shape[d];
      i += k * y_strides[axes[d]];
    }
    int axis_idx = (int)(i * inv_N);
    dx[o] += dy[i] * g[axis_idx] * inv_sqrt_variance[axis_idx] +
             dvar[axis_idx] * 2 * (x[i] - m[axis_idx]) * inv_N +
             dmean[axis_idx] * inv_N;
  }
}

/******************************************************************************/
/***             Backward Batch Gamma Beta Kernel Implementation              */
/******************************************************************************/

template <typename T>
__global__ void
backward_batch_gamma_beta_kernel(const int size1_, const int size2_,
                                 const int size02_, const int size12_,
                                 const float eps_, const T *dy, const T *m,
                                 const T *v, const T *x, T *db, T *dg) {
  NBLA_CUDA_KERNEL_LOOP(i1, size1_) {
    const T mean = m[i1];
    const T inv_sqrt_variance = (T)1 / sqrt(v[i1] + eps_);
    T dbeta = (T)0;
    T dgamma = (T)0;
    for (int i02 = 0; i02 < size02_; ++i02) {
      const int i0 = i02 / size2_;
      const int i2 = i02 % size2_;
      const int i = i0 * size12_ + i1 * size2_ + i2;
      const T value = dy[i];
      dbeta += value;
      dgamma += value * (x[i] - mean);
    }
    db[i1] += dbeta;
    dg[i1] += dgamma * inv_sqrt_variance;
  }
}

__global__ void backward_batch_kernel_gamma_beta_preprocess(
    const int N, const float *dy, const float *x, const float *m,
    float *tmp_gamma_buffer_per_block, float *tmp_beta_buffer_per_block,
    float *inv_sqrt_variance) {
  float2 gamma_beta;
  gamma_beta.x = 0; // gamma
  gamma_beta.y = 0; // beta
  NBLA_CUDA_1D_GRID_STRIDE_LOOP(i, N) {
    const float value = dy[i];
    gamma_beta.x += value * (x[i] - m[0]) * inv_sqrt_variance[0];
    gamma_beta.y += value;
  }
  gamma_beta = blockReduceSumOfFloat2(gamma_beta);
  if (threadIdx.x == 0) {
    tmp_gamma_buffer_per_block[blockIdx.x] = gamma_beta.x;
    tmp_beta_buffer_per_block[blockIdx.x] = gamma_beta.y;
  }
}

__global__ void backward_batch_kernel_gamma_beta_postprocess(
    const float *tmp_gamma_buffer_per_block,
    const float *tmp_beta_buffer_per_block, const int N, float *dg, float *db) {
  float2 gamma_beta;
  gamma_beta.x = 0; // gamma
  gamma_beta.y = 0; // beta
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    gamma_beta.x += tmp_gamma_buffer_per_block[i];
    gamma_beta.y += tmp_beta_buffer_per_block[i];
  }
  gamma_beta = blockReduceSumOfFloat2(gamma_beta);
  if (threadIdx.x == 0) {
    dg[blockIdx.x] += gamma_beta.x;
    db[blockIdx.x] += gamma_beta.y;
  }
}
}
