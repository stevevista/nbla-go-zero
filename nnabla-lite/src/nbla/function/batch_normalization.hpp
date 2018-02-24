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

/** Batch Normalization
 */
#ifndef __NBLA_FUNCTION_BATCHNORM_HPP__
#define __NBLA_FUNCTION_BATCHNORM_HPP__

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>

#include <vector>

using std::vector;

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(BatchNormalization, int);


template <typename T>
class BatchNormalization
    : public BaseFunction<int> {
protected:
  int axis_;
  int size0_, size1_, size2_, size02_, size12_;

public:
  BatchNormalization(const Context &ctx, int axis)
      : BaseFunction(ctx, axis), axis_(axis) {}
  virtual ~BatchNormalization() {}
  virtual shared_ptr<Function> copy() const {
    return create_BatchNormalization(ctx_, axis_);
  }
  virtual int min_inputs() { return 2; }
  virtual string name() { return "BatchNormalization"; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cpu>()->array_classes();
  }

protected:
  NBLA_API virtual void setup_impl(const Variables &inputs,
                                   Variable* output);
  NBLA_API virtual void forward_impl(const Variables &inputs,
                                     Variable* output);
  NBLA_API virtual void forward_impl_global(const Variables &inputs,
                                            Variable* output);
};
}
#endif
