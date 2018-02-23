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

/** Utilities for CUDA CUDNN
*/
#ifndef __NBLA_CUDA_CUDNN_HPP__
#define __NBLA_CUDA_CUDNN_HPP__

#include <cudnn.h>

#include <nbla/common.hpp>
#include <nbla/cuda/defs.hpp>
#include <nbla/singleton_manager.hpp>

#include <iostream>
#include <map>
#include <memory>
#include <unordered_map>

namespace nbla {

using std::map;
using std::shared_ptr;
using std::unordered_map;
using std::hash;

#if CUDNN_VERSION >= 5000
#define NBLA_CUDNN_USE_WORKSPACE
#else
#endif

template <class T> class cudnn_data_type;

template <> class cudnn_data_type<float> {
public:
  static cudnnDataType_t type() { return CUDNN_DATA_FLOAT; }
};
template <> class cudnn_data_type<double> {
public:
  static cudnnDataType_t type() { return CUDNN_DATA_DOUBLE; }
};
template <> class cudnn_data_type<unsigned short> {
public:
  static cudnnDataType_t type() { return CUDNN_DATA_HALF; }
};

inline string cudnn_status_to_string(cudnnStatus_t status) {
#define CASE_CUDNN_STATUS(NAME)                                                \
  case CUDNN_STATUS_##NAME:                                                    \
    return #NAME;

  switch (status) {
    CASE_CUDNN_STATUS(SUCCESS);
    CASE_CUDNN_STATUS(NOT_INITIALIZED);
    CASE_CUDNN_STATUS(ALLOC_FAILED);
    CASE_CUDNN_STATUS(BAD_PARAM);
    CASE_CUDNN_STATUS(INTERNAL_ERROR);
    CASE_CUDNN_STATUS(INVALID_VALUE);
    CASE_CUDNN_STATUS(ARCH_MISMATCH);
    CASE_CUDNN_STATUS(MAPPING_ERROR);
    CASE_CUDNN_STATUS(EXECUTION_FAILED);
    CASE_CUDNN_STATUS(NOT_SUPPORTED);
    CASE_CUDNN_STATUS(LICENSE_ERROR);
#if CUDNN_VERSION >= 6000
    CASE_CUDNN_STATUS(RUNTIME_PREREQUISITE_MISSING);
#endif
  }
  return "UNKNOWN";
#undef CASE_CUDNN_STATUS
}

#define NBLA_CUDNN_CHECK(condition)                                            \
  {                                                                            \
    cudnnStatus_t status = condition;                                          \
    NBLA_CHECK(status == CUDNN_STATUS_SUCCESS, error_code::target_specific,    \
               cudnn_status_to_string(status));                                \
  }

/** cuDNN Convolution 2d Descriptor used as a key to find previously used
 * (cached) config.
*/
struct NBLA_CUDA_API CudnnConv2dDesc {
  int device;            ///< Device ID.
  cudnnDataType_t dtype; ///< Data type.
  cudnnConvolutionMode_t
      mode;      ///< CUDNN_CONVOLUTION or CUDNN_CROSS_CORRELATION;
  int n;         ///< Batch size.
  int c;         ///< Channels of input.
  int h;         ///< Height of input.
  int w;         ///< Width of input.
  int o;         ///< Channels of output
  int kh;        ///< Height of kernel.
  int kw;        ///< Width of kernel.
  int padh;      ///< Padding height of input.
  int padw;      ///< Padding width of input.
  int strideh;   ///< Padding height of input.
  int stridew;   ///< Padding width of input.
  int group;     ///< Number of groups.
  int dilationh; ///< Dilation height of filter.
  int dilationw; ///< Dilation width of filter.

  /// Operator == compares all elements.
  bool operator==(const CudnnConv2dDesc &right) const;

  /** Custom hash function for CudnnConv2dDesc.
   */
  class Hash {
  public:
    std::size_t operator()(const CudnnConv2dDesc &x) const {
      size_t h = hash<int>{}(x.device);
      hash_combine(h, static_cast<int>(x.dtype));
      hash_combine(h, static_cast<int>(x.mode));
      hash_combine(h, x.n);
      hash_combine(h, x.c);
      hash_combine(h, x.h);
      hash_combine(h, x.w);
      hash_combine(h, x.o);
      hash_combine(h, x.kh);
      hash_combine(h, x.kw);
      hash_combine(h, x.padh);
      hash_combine(h, x.padw);
      hash_combine(h, x.strideh);
      hash_combine(h, x.stridew);
      hash_combine(h, x.group);
      hash_combine(h, x.dilationw);
      hash_combine(h, x.dilationh);
      return h;
    }
  };
};

std::ostream &operator<<(std::ostream &os, const CudnnConv2dDesc &desc);

/** cuDNN Convolution 2D resource cache.
 */
struct NBLA_CUDA_API CudnnConv2dResource {
  int device;                             ///< Device ID.
  cudnnTensorDescriptor_t x_desc;         ///< Input desc.
  cudnnTensorDescriptor_t y_desc;         ///< Output desc.
  cudnnTensorDescriptor_t b_desc;         ///< Bias desc.
  cudnnTensorDescriptor_t b_desc_deconv;  ///< Bias desc for deconvolution.
  cudnnFilterDescriptor_t w_desc;         ///< Wegiht desc.
  cudnnConvolutionDescriptor_t conv_desc; ///< Conv desc.
  cudnnConvolutionFwdAlgo_t fwd_algo;     ///< Best forward algorithm found.
  cudnnConvolutionBwdFilterAlgo_t
      bwd_filter_algo; ///< Best Backward filter algorithm found.
  cudnnConvolutionBwdDataAlgo_t
      bwd_data_algo; ///< Best backward data algorithm found.
#ifdef NBLA_CUDNN_USE_WORKSPACE
  size_t fwd_workspace_size;        ///< Forward workspace size.
  size_t bwd_filter_workspace_size; ///< Backward filter workspace size.
  size_t bwd_data_workspace_size;   ///< Backward data workspace size.
#endif
  CudnnConv2dResource(const CudnnConv2dDesc &desc);
  ~CudnnConv2dResource();

#ifdef NBLA_CUDNN_USE_WORKSPACE
  /** Get maximum workspace size.
   */
  size_t workspace_size() const;
#endif

private:
  void find_best_algorithms();
};

/**
Singleton class for storing cudnn handle for CUDA CUDNN Computation.
*/
class NBLA_CUDA_API CudnnHandleManager {
public:
  ~CudnnHandleManager();

  /**
     Get cuDNN handle for devive.
   */
  cudnnHandle_t handle(int device = -1);

  /** Hash map for CudnnConv2dResource.
   */
  unordered_map<CudnnConv2dDesc, shared_ptr<CudnnConv2dResource>,
                typename CudnnConv2dDesc::Hash>
      conv2d_resource;

protected:
  map<int, cudnnHandle_t> handles_;

private:
  friend SingletonManager;
  CudnnHandleManager();
  DISABLE_COPY_AND_ASSIGN(CudnnHandleManager);
};
}
#endif
