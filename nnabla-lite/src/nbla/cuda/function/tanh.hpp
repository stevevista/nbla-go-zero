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

#ifndef __NBLA_CUDA_FUNCTION_TANH_HPP__
#define __NBLA_CUDA_FUNCTION_TANH_HPP__


#include <nbla/function/tanh.hpp>

namespace nbla {


template <typename T> class TanhCuda : public Tanh<T> {  
protected:
  int device_;
    public:                                                                        \
  virtual ~TanhCuda() {}                                                     \
  virtual string name() { return "TanhCuda"; }                     \
    explicit TanhCuda(const Context &ctx) : Tanh<T>(ctx), 
    device_(std::stoi(ctx.device_id)) {}    \
    virtual shared_ptr<Function> copy() const { 
      return create_Tanh(this->ctx_); 
    } 
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cuda>()->array_classes();
  }
    protected:                                                                     \
  virtual void forward_impl(const Variables &inputs,   
                            Variable* output); 
};


}
#endif
